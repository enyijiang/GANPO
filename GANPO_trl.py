

import argparse
import os
import torch
from accelerate import logging
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoProcessor
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch import autocast
from contextlib import contextmanager, nullcontext
from transformers.optimization import get_scheduler
from torch.autograd import Variable
from accelerate.utils.memory import clear_device_cache
from transformers.training_args import OptimizerNames
import torch.nn.utils.spectral_norm as spectral_norm
from transformers import AutoModel
from peft import get_peft_model, LoraConfig, TaskType
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType, FullStateDictConfig
from accelerate.utils import DistributedType
from torch.nn.parallel import DistributedDataParallel as DDP
import gc

from trl import (
    DatasetMixtureConfig,
    DPOConfig,
    DPOTrainer,
    ModelConfig,
    ScriptArguments,
    TrlParser,
    get_dataset,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)


logger = logging.get_logger(__name__)



def save_dual_discriminators(accelerator, disc_chosen, disc_rejected, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    
    print("Processing disc_chosen...")
    with FSDP.state_dict_type(disc_chosen, StateDictType.FULL_STATE_DICT, save_policy):
        chosen_state = disc_chosen.state_dict()
        
    if accelerator.is_main_process:
        torch.save(chosen_state, os.path.join(output_dir, "disc_chosen.pt"))
        print("Saved disc_chosen.pt")
        del chosen_state 

    print("Processing disc_rejected...")
    with FSDP.state_dict_type(disc_rejected, StateDictType.FULL_STATE_DICT, save_policy):
        rejected_state = disc_rejected.state_dict()
        
    if accelerator.is_main_process:
        torch.save(rejected_state, os.path.join(output_dir, "disc_rejected.pt"))
        print("Saved disc_rejected.pt")
        del rejected_state
        
    accelerator.wait_for_everyone()

def get_hidden_size(model):
    """
    Safely extract hidden_size from any HuggingFace transformer model.
    Works for Qwen, LLaMA, GPT-2, Falcon, Mistral, etc.
    """
    cfg = model.config

    # Standard name
    if hasattr(cfg, "hidden_size"):
        return cfg.hidden_size

    # Some models use "dim"
    if hasattr(cfg, "dim"):
        return cfg.dim

    # GPT-2 / GPT-NeoX style
    if hasattr(cfg, "n_embd"):
        return cfg.n_embd

    raise ValueError("Could not infer hidden size from model.config.")
    
def selective_log_softmax(logits, index) -> torch.Tensor:
    """
    A memory-efficient implementation of the common `log_softmax -> gather` operation.

    This function is equivalent to the following naive implementation:
    ```python
    logps = torch.gather(logits.log_softmax(-1), dim=-1, index=index.unsqueeze(-1)).squeeze(-1)
    ```

    Args:
        logits (`torch.Tensor`):
            Logits tensor of shape `(..., num_classes)`.
        index (`torch.Tensor`):
            Index tensor of shape `(...)`, specifying the positions to gather from the log-softmax output.

    Returns:
        `torch.Tensor`:
            Gathered log probabilities with the same shape as `index`.
    """
    if logits.dtype in [torch.float32, torch.float64]:
        selected_logits = torch.gather(logits, dim=-1, index=index.unsqueeze(-1)).squeeze(-1)
        # loop to reduce peak mem consumption
        logsumexp_values = torch.stack([torch.logsumexp(lg, dim=-1) for lg in logits])
        per_token_logps = selected_logits - logsumexp_values  # log_softmax(x_i) = x_i - logsumexp(x)
    else:
        # logsumexp approach is unstable with bfloat16, fall back to slightly less efficient approach
        per_token_logps = []
        for row_logits, row_labels in zip(logits, index, strict=True):  # loop to reduce peak mem consumption
            row_logps = F.log_softmax(row_logits, dim=-1)
            row_per_token_logps = row_logps.gather(dim=-1, index=row_labels.unsqueeze(-1)).squeeze(-1)
            per_token_logps.append(row_per_token_logps)
        per_token_logps = torch.stack(per_token_logps)
    return per_token_logps

def pad_to_length(tensor: torch.Tensor, length: int, pad_value: int | float, dim: int = -1) -> torch.Tensor:
    if tensor.size(dim) >= length:
        return tensor
    else:
        pad_size = list(tensor.shape)
        pad_size[dim] = length - tensor.size(dim)
        return torch.cat(
            [
                tensor,
                pad_value * torch.ones(*pad_size, dtype=tensor.dtype, device=tensor.device),
            ],
            dim=dim,
        )

def flush_left(mask: torch.Tensor, *tensors: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, ...]:
    """
    Shift non-zero elements in the mask and corresponding tensors to the left.

    This function operates on a binary mask and any number of additional tensors with the same dimensions as the mask.
    For each row, non-zero values are shifted to the leftmost positions. Then, columns that contain only zeros across
    all rows are truncated from the mask and tensors. Visually, this operation can be represented as follows:

    ```
    [[0, 0, x, x, x, x],  ->  [[x, x, x, x],
     [0, x, x, x, 0, 0]]       [x, x, x, 0]]
    ```

    Args:
        mask (`torch.Tensor`):
            2D tensor (binary mask) with shape `(N, M)`.
        *tensors (`torch.Tensor`):
            One or more 2D tensors with the same shape as `mask`. These tensors will be processed alongside `mask`,
            with non-zero values shifted and excess zero columns truncated in the same manner.

    Returns:
        `torch.Tensor`:
            Updated binary mask with non-zero values flushed to the left and trailing zero columns removed.
        `*torch.Tensor`
            Updated tensors, processed in the same way as the mask.

    Example:
    ```python
    >>> mask = torch.tensor([[0, 0, 1, 1, 1], [0, 1, 1, 0, 0]])
    >>> tensor = torch.tensor([[9, 9, 2, 3, 4], [9, 5, 6, 9, 9]])
    >>> new_mask, new_tensor = flush_left(mask, tensor)
    >>> print(new_mask)
    tensor([[1, 1, 1],
            [1, 1, 0]])

    >>> print(new_tensor)
    tensor([[2, 3, 4],
            [5, 6, 0]])
    ```
    """
    _, M = mask.shape

    # Create copy of mask and tensors
    mask_copy = mask.clone()
    tensors = [t.clone() for t in tensors]

    # Shift non-zero values to the left
    first_non_zero = mask_copy.argmax(dim=1)
    pos = torch.arange(M, device=mask_copy.device).unsqueeze(0)
    idx_roll = (pos + first_non_zero.unsqueeze(1)) % M
    mask_roll = mask_copy.gather(1, idx_roll)
    rolled_tensors = [t.gather(1, idx_roll) for t in tensors]

    # Truncate trailing columns that are all zeros in mask_roll
    col_sums = mask_roll.sum(dim=0)
    empty_cols = col_sums == 0
    first_empty_col = int(empty_cols.to(torch.int8).argmax()) if empty_cols.any() else M
    flushed_mask = mask_roll[:, :first_empty_col]
    flushed_tensors = [t[:, :first_empty_col] for t in rolled_tensors]

    if not flushed_tensors:
        return flushed_mask
    return flushed_mask, *flushed_tensors


def flush_right(mask: torch.Tensor, *tensors: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, ...]:
    """
    Shift non-zero elements in the mask and corresponding tensors to the right. See `flush_left` for details.
    """
    _, M = mask.shape

    # Create copy of mask and tensors
    mask_copy = mask.clone()
    tensors = [t.clone() for t in tensors]

    # Shift non-zero values to the right
    flipped_mask = torch.fliplr(mask_copy)
    first_non_zero = flipped_mask.argmax(dim=1)
    pos = torch.arange(M, device=mask_copy.device).unsqueeze(0)
    idx_roll = (pos - first_non_zero.unsqueeze(1)) % M
    mask_roll = mask_copy.gather(1, idx_roll)
    rolled_tensors = [t.gather(1, idx_roll) for t in tensors]

    # Truncate leading columns that are all zeros in mask_roll
    col_sums = mask_roll.sum(dim=0)
    non_empty_cols = col_sums != 0
    first_non_empty_col = int(non_empty_cols.to(torch.int8).argmax()) if non_empty_cols.any() else M
    flushed_mask = mask_roll[:, first_non_empty_col:]
    flushed_tensors = [t[:, first_non_empty_col:] for t in rolled_tensors]

    if not flushed_tensors:
        return flushed_mask
    return flushed_mask, *flushed_tensors
    
class TransformerDiscriminator(nn.Module):
    def __init__(
        self, 
        input_dim=4096,     
        hidden_dim=512,    
        num_layers=2,       
        num_heads=8,        
        max_seq_len=2048,  
        dropout=0.1
    ):
        super().__init__()

        self.project_in = spectral_norm(nn.Linear(input_dim, hidden_dim))
        

        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len, hidden_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True 
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.head = nn.Sequential(
            spectral_norm(nn.Linear(hidden_dim, hidden_dim)),
            nn.GELU(),
            spectral_norm(nn.Linear(hidden_dim, 1))
        )
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, hidden_states, mask=None):
        """
        hidden_states: [Batch, Seq_Len, Input_Dim] 
        mask: [Batch, Seq_Len] 
        """
        batch_size, seq_len, _ = hidden_states.size()
        

        x = self.project_in(hidden_states) # -> [B, S, hidden_dim]

        if seq_len < 2048:
            x = x + self.pos_embedding[:, :seq_len, :]
        else:
            x = x[:, :2048, :] + self.pos_embedding
        

        src_key_padding_mask = None
        if mask is not None:
            src_key_padding_mask = (mask == 0) 
        
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)

        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).expand_as(x) # [B, S, Dim]
            sum_embeddings = torch.sum(x * mask_expanded, dim=1)
            sum_mask = mask.sum(dim=1, keepdim=True).clamp(min=1e-9)
            pooled_output = sum_embeddings / sum_mask
        else:
            pooled_output = x.mean(dim=1) 
        score = self.head(pooled_output) # -> [B, 1]
        
        return score

class QuadGANDPOTrainer(DPOTrainer):
    """
    QuadGAN DPO
    """
    def __init__(self, disc_chosen, disc_rejected, projector, best_model, disc_lr=1e-5, adv_weight=1.0, momentum=0.9,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.disc_chosen = disc_chosen
        self.disc_rejected = disc_rejected
        self.projector = projector
        self.best_model = best_model
        self.adv_weight = adv_weight
        self.disc_criterion = nn.BCEWithLogitsLoss()
        self.disc_lr = self.args.learning_rate / 2.0

        self.disc_chosen = self.disc_chosen.to(self.accelerator.device)
        # self.disc_chosen = self.accelerator.prepare_model(self.disc_chosen)
        self.disc_rejected = self.disc_rejected.to(self.accelerator.device)
        # self.disc_rejected = self.accelerator.prepare_model(self.disc_rejected)

        # print("✅ Initializing QuadGANDPOTrainer with adversarial critics.")
            
        self.disc_chosen_optimizer = None
        self.disc_chosen_scheduler = None
        self.disc_rejected_optimizer = None
        self.disc_rejected_scheduler = None
        self.projector_optimizer = None
        self.projector_scheduler = None
        self.momentum = momentum 

        self.running_mean_pos = None
        self.running_mean_neg = None
        self.running_mean_policy = None
        self.running_mean_policy_rejected = None

        self.running_mean_pos_rej = None
        self.running_mean_neg_rej = None
        self.running_mean_policy_rej = None
        self.running_mean_policy_rejected_rej = None
    
    def _update_mean(self, running_mean, curr_mean):
        curr = curr_mean.detach()
        # 如果是 None (第一次运行)，直接赋值
        if running_mean is None:
            return curr
        else:
            return self.momentum * running_mean + (1 - self.momentum) * curr
    
    def concatenated_forward(
        self, model: nn.Module, batch: dict[str, list | torch.LongTensor], is_ref_model: bool = False
    ) -> dict[str, torch.Tensor]:
        """
        Runs the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

        We do this to avoid doing two forward passes, because it's faster for FSDP.

        Args:
            model:
                Model to run the forward pass on.
            batch:
                Batch of input data.
            is_ref_model:
                Whether this method is being called for the reference model. If `True`, length desensitization is not
                applied.
        """
        num_examples = batch["prompt_input_ids"].shape[0]

        concatenated_batch = self.concatenated_inputs(batch, padding_value=self.pad_token_id)

        model_kwargs = {"use_cache": False}
        if self.aux_loss_enabled:
            model_kwargs["output_router_logits"] = True

        # Add the pixel values and attention masks for vision models
        if "pixel_values" in concatenated_batch:
            model_kwargs["pixel_values"] = concatenated_batch["pixel_values"]
        if "pixel_attention_mask" in concatenated_batch:
            model_kwargs["pixel_attention_mask"] = concatenated_batch["pixel_attention_mask"]
        if "image_sizes" in concatenated_batch:
            model_kwargs["image_sizes"] = concatenated_batch["image_sizes"]

        prompt_input_ids = concatenated_batch["prompt_input_ids"]
        prompt_attention_mask = concatenated_batch["prompt_attention_mask"]
        completion_input_ids = concatenated_batch["completion_input_ids"]
        completion_attention_mask = concatenated_batch["completion_attention_mask"]
        if self.is_encoder_decoder:
            labels = completion_input_ids
            labels[completion_attention_mask == 0] = self.label_pad_token_id
            outputs = model(
                input_ids=prompt_input_ids,
                attention_mask=prompt_attention_mask,
                labels=labels,  # we need the labels for the logits to be returned
                **model_kwargs,
            )
            logits = outputs.logits
            loss_mask = completion_attention_mask.bool()
        else:
            # Concatenate the prompt and completion inputs
            input_ids = torch.cat((prompt_input_ids, completion_input_ids), dim=1)
            attention_mask = torch.cat((prompt_attention_mask, completion_attention_mask), dim=1)
            if "token_type_ids" in concatenated_batch:
                prompt_token_type_ids = concatenated_batch["token_type_ids"]
                token_type_ids = pad_to_length(prompt_token_type_ids, input_ids.shape[1], 0)
            # Mask the prompt but not the completion for the loss
            loss_mask = torch.cat(
                (torch.zeros_like(prompt_attention_mask), completion_attention_mask),
                dim=1,
            )

            # Flush and truncate
            if self.max_length is not None and self.max_length < attention_mask.size(1):
                if self.truncation_mode == "keep_start":
                    # Flush left to reduce the memory usage
                    # [[0, 0, x, x, x, x],  ->  [[x, x, x, x],
                    #  [0, x, x, x, 0, 0]]       [x, x, x, 0]]
                    if "token_type_ids" in concatenated_batch:
                        attention_mask, input_ids, loss_mask, token_type_ids = flush_left(
                            attention_mask, input_ids, loss_mask, token_type_ids
                        )
                    else:
                        attention_mask, input_ids, loss_mask = flush_left(attention_mask, input_ids, loss_mask)
                    attention_mask = attention_mask[:, : self.max_length]
                    input_ids = input_ids[:, : self.max_length]
                    loss_mask = loss_mask[:, : self.max_length]
                elif self.truncation_mode == "keep_end":
                    # Flush right before truncating left, then flush left
                    # [[0, 0, x, x, x, x],  ->  [[0, 0, x, x],
                    #  [0, x, x, x, 0, 0]]       [0, x, x, x]]
                    if "token_type_ids" in concatenated_batch:
                        attention_mask, input_ids, loss_mask, token_type_ids = flush_left(
                            attention_mask, input_ids, loss_mask, token_type_ids
                        )
                        token_type_ids = token_type_ids[:, -self.max_length :]
                    else:
                        attention_mask, input_ids, loss_mask = flush_right(attention_mask, input_ids, loss_mask)
                    input_ids = input_ids[:, -self.max_length :]
                    attention_mask = attention_mask[:, -self.max_length :]
                    loss_mask = loss_mask[:, -self.max_length :]
                    if "token_type_ids" in concatenated_batch:
                        attention_mask, input_ids, loss_mask, token_type_ids = flush_left(
                            attention_mask, input_ids, loss_mask, token_type_ids
                        )
                    else:
                        attention_mask, input_ids, loss_mask = flush_left(attention_mask, input_ids, loss_mask)
                else:
                    raise ValueError(
                        f"Unknown truncation mode: '{self.truncation_mode}'. Should be one of ['keep_end', "
                        "'keep_start']."
                    )
            else:
                # Flush left to reduce the memory usage
                # [[0, 0, x, x, x, x],  ->  [[x, x, x, x],
                #  [0, x, x, x, 0, 0]]       [x, x, x, 0]]
                if "token_type_ids" in concatenated_batch:
                    attention_mask, input_ids, loss_mask, token_type_ids = flush_left(
                        attention_mask, input_ids, loss_mask, token_type_ids
                    )
                else:
                    attention_mask, input_ids, loss_mask = flush_left(attention_mask, input_ids, loss_mask)

            if "token_type_ids" in concatenated_batch:
                model_kwargs["token_type_ids"] = token_type_ids

            if self.use_logits_to_keep:
                # Compute logits_to_keep based on loss_mask pattern:
                # [[0, 0, 0, x, x, x, x],
                #  [0, 0, 0, x, x, x, 0]]
                #         ^ start computing logits from here ([:, -(7-3+1):])
                first_compute_index = loss_mask.nonzero(as_tuple=True)[1].min()
                logits_to_keep = (loss_mask.shape[1] - first_compute_index).item() + 1  # +1 for the first label
                model_kwargs["logits_to_keep"] = logits_to_keep

            model_kwargs["output_hidden_states"] = True

            if self.padding_free:
                # Flatten the input_ids, position_ids, and loss_mask
                # input_ids = [[a, b, c, 0], ->     input_ids = [[a, b, c, d, e, f, g]]
                #              [d, e, f, g]]     position_ids = [[0, 1, 2, 0, 1, 2, 3]]
                input_ids = input_ids[attention_mask.bool()].unsqueeze(0)
                loss_mask = loss_mask[attention_mask.bool()].unsqueeze(0)
                position_ids = attention_mask.cumsum(1)[attention_mask.bool()].unsqueeze(0) - 1
                model_kwargs["position_ids"] = position_ids
            else:
                model_kwargs["attention_mask"] = attention_mask

            outputs = model(input_ids, **model_kwargs)
            logits = outputs.logits

            # Offset the logits by one to align with the labels
            labels = torch.roll(input_ids, shifts=-1, dims=1)
            loss_mask = torch.roll(loss_mask, shifts=-1, dims=1).bool()

            if self.use_logits_to_keep:
                # Align labels with logits
                # logits:    -,  -, [x2, x3, x4, x5, x6]
                #                     ^ --------- ^       after logits[:, :-1, :]
                # labels:   [y0, y1, y2, y3, y4, y5, y6]
                #                         ^ --------- ^   with logits_to_keep=4, [:, -4:]
                # loss_mask: [0,  0,  0,  1,  1,  1,  1]
                labels = labels[:, -logits_to_keep:]
                loss_mask = loss_mask[:, -logits_to_keep:]

        if logits.shape[:2] != labels.shape[:2]:
            # for LLaVA, the returned logits include the image tokens (placed before the text tokens)
            seq_len = labels.shape[1]
            logits = logits[:, -seq_len:]

        # Compute the log probabilities of the labels
        labels[~loss_mask] = 0  # dummy token; we'll ignore the losses on these tokens later
        per_token_logps = selective_log_softmax(logits, labels)
        per_token_logps[~loss_mask] = 0
        per_token_logps = torch.roll(per_token_logps, shifts=1, dims=1)

        if self.padding_free:
            # Unflatten the per_token_logps (shape: [1, sum_seq_len] -> [batch_size, seq_len])
            batch_size, seq_len = attention_mask.shape
            per_token_logps_ = torch.zeros(
                batch_size, seq_len, device=outputs.logits.device, dtype=outputs.logits.dtype
            )
            per_token_logps_[attention_mask.bool()] = per_token_logps
            per_token_logps = per_token_logps_

        all_logps = per_token_logps[:, 1:].sum(-1)

        output = {}

        if self.use_weighting:
            with torch.no_grad():
                # Eq (2) of the WPO paper: https://huggingface.co/papers/2406.11827
                logprobs = F.log_softmax(logits, dim=-1)
                weights_adjustment_factor = torch.logsumexp(2 * logprobs, dim=-1)  # same as sum(probs**2) in log space
                per_token_logps_adjusted = per_token_logps - weights_adjustment_factor
                all_weights = (per_token_logps_adjusted * loss_mask).sum(-1) / loss_mask.sum(-1)
                chosen_weights = all_weights[:num_examples]
                rejected_weights = all_weights[num_examples:]
                output["policy_weights"] = torch.clamp(torch.exp(chosen_weights + rejected_weights), max=1)

        if self.args.rpo_alpha is not None or "sft" in self.loss_type:
            # Only use the chosen logits for the RPO loss or SFT loss
            chosen_logits = logits[:num_examples, :-1] if not self.is_encoder_decoder else logits[:num_examples]
            chosen_labels = labels[:num_examples, :-1] if not self.is_encoder_decoder else labels[:num_examples]

            # Compute the log probabilities of the labels
            output["nll_loss"] = F.cross_entropy(
                torch.flatten(chosen_logits, end_dim=1), torch.flatten(chosen_labels, end_dim=1), ignore_index=0
            )

        if "ipo" in self.loss_type:
            all_logps = all_logps / loss_mask.sum(-1)

        if self.args.ld_alpha is not None and not is_ref_model:
            # Compute response lengths based on loss_mask
            completion_lengths = loss_mask.sum(dim=1)

            chosen_lengths = completion_lengths[:num_examples]
            rejected_lengths = completion_lengths[num_examples:]
            public_lengths = torch.min(chosen_lengths, rejected_lengths)  # l_p in the paper
            public_lengths = torch.cat([public_lengths, public_lengths], dim=0)

            seq_len = per_token_logps.size(1)
            position_ids = torch.arange(seq_len, device=per_token_logps.device).expand_as(per_token_logps)

            ld_mask = position_ids < public_lengths.unsqueeze(1)
            mask = position_ids < completion_lengths.unsqueeze(1)

            front_mask = (ld_mask & mask).float()
            rear_mask = (~ld_mask & mask).float()
            front_logps = (per_token_logps * front_mask).sum(dim=1)
            rear_logps = (per_token_logps * rear_mask).sum(dim=1)

            all_logps = front_logps + self.args.ld_alpha * rear_logps

        output["chosen_logps"] = all_logps[:num_examples]
        output["rejected_logps"] = all_logps[num_examples:]

        # Compute the mean logits
        if self.padding_free:
            # position_ids contains a sequence of range identifiers (e.g., [[0, 1, 2, 0, 1, 2, 3, ...]]).
            # There are 2*num_examples ranges in total: the first half corresponds to the chosen tokens,
            # and the second half to the rejected tokens.
            # To find the start of the rejected tokens, we look for the num_examples+1-th zero in pos_id.
            split_idx = (position_ids == 0).nonzero(as_tuple=True)[1][num_examples]
            mean_chosen_logits = logits[0, :split_idx][loss_mask[0, :split_idx]].mean()
            mean_rejected_logits = logits[0, split_idx:][loss_mask[0, split_idx:]].mean()
        else:
            mean_chosen_logits = logits[:num_examples][loss_mask[:num_examples]].mean()
            mean_rejected_logits = logits[num_examples:][loss_mask[num_examples:]].mean()

        output["mean_chosen_logits"] = mean_chosen_logits
        output["mean_rejected_logits"] = mean_rejected_logits

        if self.aux_loss_enabled:
            output["aux_loss"] = outputs.aux_loss

        return output, outputs.hidden_states[-1]    

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        super().create_optimizer_and_scheduler(num_training_steps)
        
        print("Initializing Disc Chosen Optimizer...")
        self.disc_chosen_optimizer = AdamW(self.disc_chosen.parameters(), lr=self.disc_lr)
        
        self.disc_chosen_scheduler = get_scheduler(
            "cosine", 
            optimizer=self.disc_chosen_optimizer,
            num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
            num_training_steps=num_training_steps,
        )

        print("Initializing Disc Chosen Optimizer...")
        self.disc_rejected_optimizer = AdamW(self.disc_rejected.parameters(), lr=self.disc_lr)

        self.disc_rejected_scheduler = get_scheduler(
            "cosine", 
            optimizer=self.disc_rejected_optimizer,
            num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
            num_training_steps=num_training_steps,
        )
        
        if self.accelerator is not None:
            self.disc_chosen, self.disc_chosen_optimizer, self.disc_chosen_scheduler = \
                self.accelerator.prepare(self.disc_chosen, self.disc_chosen_optimizer, self.disc_chosen_scheduler)
            self.disc_rejected, self.disc_rejected_optimizer, self.disc_rejected_scheduler = \
                self.accelerator.prepare(self.disc_rejected, self.disc_rejected_optimizer, self.disc_rejected_scheduler)

        # print("✅ Disc Chosen Optimizer Created.")
        print("Initializing Projector Optimizer...")
        if self.projector is not None:
            self.projector_optimizer = AdamW(self.projector.parameters(), lr=5e-7)
            self.projector_scheduler = get_scheduler(
                "cosine", 
                optimizer=self.projector_optimizer,
                num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                num_training_steps=num_training_steps,
            )       

        if self.accelerator is not None:
            self.projector, self.projector_optimizer, self.projector_scheduler = \
                self.accelerator.prepare(self.projector, self.projector_optimizer, self.projector_scheduler)

    def get_batch_loss_metrics(
        self,
        model,
        batch: dict[str, list | torch.LongTensor],
        train_eval,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute the DPO loss and other metrics for the given batch of inputs for train or test."""
        metrics = {}

        if self.args.use_liger_kernel:
            model_output = self._compute_loss_liger(model, batch)
            losses = model_output["loss"]
            chosen_rewards = model_output["chosen_rewards"]
            rejected_rewards = model_output["rejected_rewards"]
        else:
            model_output, _= self.concatenated_forward(model, batch)

            # if ref_chosen_logps and ref_rejected_logps in batch use them, otherwise use the reference model
            if "ref_chosen_logps" in batch and "ref_rejected_logps" in batch:
                ref_chosen_logps = batch["ref_chosen_logps"]
                ref_rejected_logps = batch["ref_rejected_logps"]
            else:
                ref_chosen_logps, ref_rejected_logps = self.compute_ref_log_probs(batch)

            # Initialize combined losses
            losses = 0
            chosen_rewards = 0
            rejected_rewards = 0

            # Compute losses for each loss type
            for idx, loss_type in enumerate(self.loss_type):
                # Compute individual loss using standard DPO loss function
                _losses, _chosen_rewards, _rejected_rewards = self.dpo_loss(
                    model_output["chosen_logps"],
                    model_output["rejected_logps"],
                    ref_chosen_logps,
                    ref_rejected_logps,
                    loss_type,
                    model_output,
                )

                # Add weighted contributions
                weight = self.loss_weights[idx] if self.loss_weights else 1.0
                losses = losses + _losses * weight
                chosen_rewards = chosen_rewards + _chosen_rewards * weight
                rejected_rewards = rejected_rewards + _rejected_rewards * weight

        reward_accuracies = (chosen_rewards > rejected_rewards).float()

        if self.args.rpo_alpha is not None:
            losses = losses + self.args.rpo_alpha * model_output["nll_loss"]  # RPO loss from V3 of the paper

        if self.use_weighting:
            losses = losses * model_output["policy_weights"]

        if self.aux_loss_enabled:
            losses = losses + self.aux_loss_coef * model_output["aux_loss"]

        prefix = "eval_" if train_eval == "eval" else ""
        metrics[f"{prefix}rewards/chosen"] = self.accelerator.gather_for_metrics(chosen_rewards).mean().item()
        metrics[f"{prefix}rewards/rejected"] = self.accelerator.gather_for_metrics(rejected_rewards).mean().item()
        metrics[f"{prefix}rewards/accuracies"] = self.accelerator.gather_for_metrics(reward_accuracies).mean().item()
        metrics[f"{prefix}rewards/margins"] = (
            self.accelerator.gather_for_metrics(chosen_rewards - rejected_rewards).mean().item()
        )
        metrics[f"{prefix}logps/chosen"] = (
            self.accelerator.gather_for_metrics(model_output["chosen_logps"]).detach().mean().item()
        )
        metrics[f"{prefix}logps/rejected"] = (
            self.accelerator.gather_for_metrics(model_output["rejected_logps"]).detach().mean().item()
        )
        metrics[f"{prefix}logits/chosen"] = (
            self.accelerator.gather_for_metrics(model_output["mean_chosen_logits"]).detach().mean().item()
        )
        metrics[f"{prefix}logits/rejected"] = (
            self.accelerator.gather_for_metrics(model_output["mean_rejected_logits"]).detach().mean().item()
        )
        if self.args.rpo_alpha is not None or "sft" in self.loss_type:
            metrics[f"{prefix}nll_loss"] = (
                self.accelerator.gather_for_metrics(model_output["nll_loss"]).detach().mean().item()
            )
        if self.aux_loss_enabled:
            metrics[f"{prefix}aux_loss"] = (
                self.accelerator.gather_for_metrics(model_output["aux_loss"]).detach().mean().item()
            )

        return losses.mean(), metrics
    
    def compute_ref_log_probs(self, batch: dict[str, torch.LongTensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Computes log probabilities of the reference model for a single padded batch of a DPO specific dataset."""
        compte_ref_context_manager = (
            autocast(self.accelerator.device.type) if self._peft_has_been_casted_to_bf16 else nullcontext()
        )
        with torch.no_grad(), compte_ref_context_manager:
            if self.ref_model is None:
                with self.null_ref_context():
                    ref_model_output, _ = self.concatenated_forward(self.model, batch, is_ref_model=True)
            else:
                ref_model_output, _ = self.concatenated_forward(self.ref_model, batch, is_ref_model=True)
        return ref_model_output["chosen_logps"], ref_model_output["rejected_logps"]

    def compute_loss(self, model, inputs, h_policy, h_policy_rejected, outputs_policy, outputs_ref, num_items_in_batch=None,return_outputs=False):

        if self.projector is not None:
            self.projector.train()
            for p in self.projector.parameters():
                p.requires_grad_(True)
        
        # caculate discriminator loss 
        # Freeze discriminator params but DO NOT detach last_hidden so model gets gradients
        self.disc_chosen.eval()
        for p in self.disc_chosen.parameters():
            p.requires_grad_(False)

        self.disc_rejected.eval()
        for p in self.disc_rejected.parameters():
            p.requires_grad_(False)

        model.train()
        for p in model.parameters():
            p.requires_grad_(True)

        policy_chosen_logps = outputs_policy["chosen_logps"]
        policy_rejected_logps = outputs_policy["rejected_logps"]

        ref_chosen_logps = outputs_ref["chosen_logps"]
        ref_rejected_logps = outputs_ref["rejected_logps"]

        # calculate DPO deltas
        delta_chosen = (policy_chosen_logps - ref_chosen_logps) 
        delta_rejected = (policy_rejected_logps - ref_rejected_logps) 

        # 5) DPO-style logistic loss without KL
        diff = self.beta * (delta_chosen - delta_rejected)
        losses = -F.logsigmoid(diff)

        dpo_loss = losses.mean()

        
        s_policy = self.disc_chosen(
            h_policy,
            # inputs["chosen_attention_mask"],
        )
        s_policy_rejected = self.disc_rejected(
            h_policy_rejected,
            # inputs["rejected_attention_mask"],
        )

        ref_pos = self.running_mean_pos
        ref_neg_rejected = self.running_mean_neg_rej

        

        ones = Variable(torch.ones_like(s_policy), requires_grad=False)

        g_loss_mimic = self.disc_criterion(s_policy - ref_pos, ones)

        g_loss_mimic_rej = self.disc_criterion(s_policy_rejected - ref_neg_rejected, ones)

        disc_loss = (g_loss_mimic + g_loss_mimic_rej) / 2.0
        loss = dpo_loss + self.adv_weight * disc_loss

        if return_outputs:
            return loss, {
                "dpo_loss": dpo_loss,
                "g_loss_mimic": g_loss_mimic,
                "g_loss_mimic_rej": g_loss_mimic_rej,
            }
        return loss
    
    # ------------------------------------------------------------------
    # Loss 2: update DISCRIMINATOR (backbone frozen, only reps used)
    # ------------------------------------------------------------------
    def compute_loss_disc(
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor],
        h_policy: torch.Tensor,
        h_policy_rejected: torch.Tensor,
        h_pos: torch.Tensor,
        h_neg: torch.Tensor,
        num_items_in_batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Gradients go ONLY to discriminator. Backbone is a frozen feature extractor.
        """
        batch = inputs

        # Backbone: frozen
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)

        if self.projector is not None:
            self.projector.eval()
            for p in self.projector.parameters():
                p.requires_grad_(False)

        # Discriminator: train
        self.disc_chosen.train()
        for p in self.disc_chosen.parameters():
            p.requires_grad_(True)

        self.disc_rejected.train()
        for p in self.disc_rejected.parameters():
            p.requires_grad_(True)

        s_policy = self.disc_chosen(
            h_policy,
            # inputs["chosen_attention_mask"],
        )

        s_pos = self.disc_chosen(
            h_pos,
            # inputs["chosen_attention_mask"],
        )
        s_neg = self.disc_chosen(
            h_neg,
            # inputs["rejected_attention_mask"],
        )

        s_policy_rejected_rej = self.disc_rejected(
            h_policy_rejected,
            # inputs["rejected_attention_mask"],
        )
        s_pos_rej = self.disc_rejected(
            h_pos,
            # inputs["chosen_attention_mask"],
        )
        s_neg_rej = self.disc_rejected(
            h_neg,
            # inputs["rejected_attention_mask"],
        )

        # 2. 计算当前 Batch 均值
        curr_mean_pos = s_pos.mean()
        curr_mean_neg = s_neg.mean()
        curr_mean_policy = s_policy.mean()

        curr_mean_pos_rej = s_pos_rej.mean()
        curr_mean_neg_rej = s_neg_rej.mean()
        curr_mean_policy_rejected_rej = s_policy_rejected_rej.mean()


        if self.running_mean_pos is None:
            self.running_mean_pos = curr_mean_pos.detach()
            self.running_mean_neg = curr_mean_neg.detach()
            self.running_mean_policy = curr_mean_policy.detach()

            self.running_mean_pos_rej = curr_mean_pos_rej.detach()
            self.running_mean_neg_rej = curr_mean_neg_rej.detach()
            self.running_mean_policy_rejected_rej = curr_mean_policy_rejected_rej.detach()
        else:
            # 后续运行，正常更新
            self.running_mean_pos = self._update_mean(self.running_mean_pos, curr_mean_pos)
            self.running_mean_neg = self._update_mean(self.running_mean_neg, curr_mean_neg)
            self.running_mean_policy = self._update_mean(self.running_mean_policy, curr_mean_policy)

            self.running_mean_pos_rej = self._update_mean(self.running_mean_pos_rej, curr_mean_pos_rej)
            self.running_mean_neg_rej = self._update_mean(self.running_mean_neg_rej, curr_mean_neg_rej)
            self.running_mean_policy_rejected_rej = self._update_mean(self.running_mean_policy_rejected_rej, curr_mean_policy_rejected_rej)

        ref_pos = self.running_mean_pos
        ref_neg = self.running_mean_neg
        ref_policy = self.running_mean_policy

        ref_pos_rej = self.running_mean_pos_rej 
        ref_neg_rej = self.running_mean_neg_rej
        ref_policy_rejected_rej = self.running_mean_policy_rejected_rej

        ones = Variable(torch.ones_like(s_pos), requires_grad=False)
        zeros = Variable(torch.zeros_like(s_pos), requires_grad=False)

        d_loss_real_policy = self.disc_criterion(s_pos - ref_policy, ones) + self.disc_criterion(s_policy - ref_pos, zeros)
        

        d_loss_policy_neg = self.disc_criterion(s_policy - ref_neg, ones) + self.disc_criterion(s_neg - ref_policy, zeros)


        d_loss_real_policy_rej = self.disc_criterion(s_neg_rej - ref_policy_rejected_rej, ones) + self.disc_criterion(s_policy_rejected_rej - ref_neg_rej, zeros)

        d_loss_policy_neg_rej = self.disc_criterion(s_policy_rejected_rej - ref_pos_rej, ones) + self.disc_criterion(s_pos_rej - ref_policy_rejected_rej, zeros)

        disc_loss_pos = (d_loss_real_policy + d_loss_policy_neg) / 4.0
        disc_loss_neg = (d_loss_real_policy_rej + d_loss_policy_neg_rej) / 4.0

        return disc_loss_pos, disc_loss_neg
    
    def training_step(
        self,
        model: nn.Module,
        inputs,
        num_items_in_batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs, updating BOTH Critic and Policy.
        """
        
        # 1. [Standard] Prepare buffers for context parallelism
        cp_context, inputs = self._prepare_context_parallel_inputs(model, inputs)

        # 2. [Standard] Context manager is no-op if CP isn't enabled
        with cp_context():
            model.train()
            if hasattr(self, "disc_chosen"):
                self.disc_chosen.train()

            if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
                self.optimizer.train()

            inputs = self._prepare_inputs(inputs)

            outputs_policy, h_policy_all = self.concatenated_forward(model, inputs, is_ref_model=False)

            if self.projector is not None:
                h_policy_all = self.projector(h_policy_all)
            h_policy, h_policy_rejected = torch.chunk(h_policy_all, 2, dim=0)

            # --- 2. Reference Forward (一次性 & inference_mode) ---
            with torch.no_grad():
                outputs_ref, best_hidden_outputs = self.concatenated_forward(self.ref_model, inputs, is_ref_model=True)
                h_ref_all = best_hidden_outputs
                h_pos, h_neg = torch.chunk(h_ref_all, 2, dim=0)
            

            if hasattr(self, "disc_chosen") and self.disc_chosen_optimizer is not None:

                critic_loss_pos, critic_loss_neg = self.compute_loss_disc(model, inputs, h_policy.detach(), h_policy_rejected.detach(), h_pos.detach(), h_neg.detach(), num_items_in_batch=num_items_in_batch)


                if self.args.n_gpu > 1:
                    critic_loss_pos = critic_loss_pos.mean()
                    critic_loss_neg = critic_loss_neg.mean()

                if (not self.model_accepts_loss_kwargs or num_items_in_batch is None) and self.compute_loss_func is None:
                    critic_loss_pos = critic_loss_pos / self.current_gradient_accumulation_steps
                    critic_loss_neg = critic_loss_neg / self.current_gradient_accumulation_steps

                self.log({'disc_loss_pos': critic_loss_pos.detach().item(),
                          'disc_loss_neg': critic_loss_neg.detach().item()})


                self.accelerator.backward(critic_loss_pos)
                self.accelerator.backward(critic_loss_neg)

                is_gradient_accumulation_boundary = (
                    (self.state.global_step + 1) % self.args.gradient_accumulation_steps == 0
                )
                
                if is_gradient_accumulation_boundary:

                    self.disc_chosen_optimizer.step()
                    self.disc_chosen_scheduler.step()
                    self.disc_chosen_optimizer.zero_grad()

                    self.disc_rejected_optimizer.step()
                    self.disc_rejected_scheduler.step()
                    self.disc_rejected_optimizer.zero_grad()



            with self.compute_loss_context_manager():

                loss, specs = self.compute_loss(model, inputs, h_policy, h_policy_rejected, outputs_policy, outputs_ref, num_items_in_batch=num_items_in_batch, return_outputs=True)

            # -------- log to wandb --------
            if specs:
            #     # add step info
                logs = {f"{k}": v.detach().item() if isinstance(v, torch.Tensor) else v for k, v in specs.items()}
                self.log(logs)   # Trainer logs to wandb automatically
            
            del inputs
            if (
                self.args.torch_empty_cache_steps is not None
                and self.state.global_step % self.args.torch_empty_cache_steps == 0
            ):
                clear_device_cache()

            kwargs = {}

            # [Standard] LOMO Optimizer support
            if self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
                kwargs["learning_rate"] = self._get_learning_rate()

            # [Standard] Multi-GPU Loss Averaging
            if self.args.n_gpu > 1:
                loss = loss.mean()

            # [Standard] Gradient Accumulation Normalization
            # if (not self.model_accepts_loss_kwargs or num_items_in_batch is None) and self.compute_loss_func is None:
            loss = loss / self.current_gradient_accumulation_steps

            # [Standard] DeepSpeed special handling
            if self.accelerator.distributed_type == DistributedType.DEEPSPEED:
                kwargs["scale_wrt_gas"] = False

            # [Standard] Policy Backward

            self.accelerator.backward(loss)

            is_gradient_accumulation_boundary = (
                    (self.state.global_step + 1) % self.args.gradient_accumulation_steps == 0
                )
                
            if is_gradient_accumulation_boundary:
                if self.projector is not None:
                    if self.args.max_grad_norm is not None and self.args.max_grad_norm > 0:
                        self.accelerator.clip_grad_norm_(self.projector.parameters(), self.args.max_grad_norm)
                    
                    self.projector_optimizer.step()
                    self.projector_scheduler.step()
                    self.projector_optimizer.zero_grad()

            return loss.detach()

def main(script_args, training_args, model_args, dataset_args):
    ################
    # Model
    ###################
    dtype = model_args.dtype if model_args.dtype in ["auto", None] else getattr(torch, model_args.dtype)
    model_kwargs = dict(
        revision=model_args.model_revision,
        attn_implementation=model_args.attn_implementation,
        dtype=dtype,
    )
    quantization_config = get_quantization_config(model_args)
    if quantization_config is not None:
        # Passing None would not be treated the same as omitting the argument, so we include it only when valid.
        model_kwargs["quantization_config"] = quantization_config
        model_kwargs["device_map"] = get_kbit_device_map()
    # model_kwargs["device_map"] = {'':device_string}

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code,**model_kwargs
    )
    peft_config = get_peft_config(model_args)
    if peft_config is None:
        best_model = None
        ref_model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code, **model_kwargs
        )
    else:
        ref_model = None
    if script_args.ignore_bias_buffers:
        # torch distributed hack
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]

    # Load the dataset
    if dataset_args.datasets and script_args.dataset_name:
        logger.warning(
            "Both `datasets` and `dataset_name` are provided. The `datasets` argument will be used to load the "
            "dataset and `dataset_name` will be ignored."
        )
        dataset = get_dataset(dataset_args)
    elif dataset_args.datasets and not script_args.dataset_name:
        dataset = get_dataset(dataset_args)
    elif not dataset_args.datasets and script_args.dataset_name:
        dataset = load_dataset(
            script_args.dataset_name, name=script_args.dataset_config, streaming=script_args.dataset_streaming
        )
    else:
        raise ValueError("Either `datasets` or `dataset_name` must be provided.")

    if 'meta' in model_args.model_name_or_path.lower():
        num_layer = 2
    else:
        num_layer = 2
    
    hidden_size = get_hidden_size(model)
    hidden_size_teacher = get_hidden_size(ref_model)
    disc_chosen = TransformerDiscriminator(input_dim=hidden_size_teacher, num_layers=num_layer)
    disc_chosen = disc_chosen.to('cuda' if torch.cuda.is_available() else 'cpu')


    disc_rejected = TransformerDiscriminator(input_dim=hidden_size_teacher, num_layers=num_layer)
    disc_rejected = disc_rejected.to('cuda' if torch.cuda.is_available() else 'cpu')

    # 初始化
    if hidden_size != hidden_size_teacher:
        projector = None
    else:
        print("Hidden size of student and teacher are the same, no need for projector.")
        projector = None

    print("✅ Model and dataset loaded.")

    # Initialize the DPO trainer
    trainer = QuadGANDPOTrainer(
        disc_chosen=disc_chosen,
        disc_rejected=disc_rejected,
        disc_lr=1e-7, 
        adv_weight=1.0,
        model=model,
        ref_model=ref_model,
        best_model=best_model,
        projector=projector,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        peft_config=peft_config,
    )

    # Train the model
    trainer.train()

    # Log training complete
    trainer.accelerator.print("✅ Training completed.")

    if training_args.eval_strategy != "no":
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Save and push to Hub
    trainer.save_model(training_args.output_dir)
    trainer.accelerator.print(f"💾 Model saved to {training_args.output_dir}.")

    save_dual_discriminators(
        accelerator=trainer.accelerator,
        disc_chosen=trainer.disc_chosen,
        disc_rejected=trainer.disc_rejected,
        output_dir='disc_ckpts')

    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)
        trainer.accelerator.print(f"🤗 Model pushed to the Hub in https://huggingface.co/{trainer.hub_model_id}.")


def make_parser(subparsers: argparse._SubParsersAction | None = None):
    dataclass_types = (ScriptArguments, DPOConfig, ModelConfig, DatasetMixtureConfig)
    if subparsers is not None:
        parser = subparsers.add_parser("dpo", help="Run the DPO training script", dataclass_types=dataclass_types)
    else:
        parser = TrlParser(dataclass_types)
    return parser


if __name__ == "__main__":
    parser = make_parser()
    # When using the trl cli, this script may be run with additional arguments, corresponding accelerate arguments.
    # To ensure that their parsing does not interfere with the script arguments, parse the arguments with
    # `return_remaining_strings=True`, then ignore the remaining strings.
    script_args, training_args, model_args, dataset_args, _ = parser.parse_args_and_config(
        return_remaining_strings=True
    )

    main(script_args, training_args, model_args, dataset_args)
