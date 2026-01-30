#!/bin/bash

# gemma
python -m accelerate.commands.launch --num_processes=4 --config_file fsdp.yaml GANPO_trl.py \
    --dataset_name trl-lib/ultrafeedback_binarized \
    --model_name_or_path google/gemma-2-2b-it \
    --learning_rate 5.0e-7 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 16 \
    --gradient_checkpointing true \
    --eval_strategy steps \
    --eval_steps 100 \
    --save_steps 5000 \
    --output_dir ~/hdd/gemma-2-2b-it-GANPO \
    --no_remove_unused_columns \
    --report_to wandb \
    --save_only_model True


# llama
python -m accelerate.commands.launch --num_processes=4 --config_file fsdp.yaml GANPO_trl.py \
    --dataset_name princeton-nlp/llama3-ultrafeedback-armorm \
    --model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct \
    --logging_steps 5 \
    --learning_rate 1e-6 \
    --lr_scheduler_type cosine \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 16 \
    --gradient_checkpointing true \
    --warmup_ratio 0.1 \
    --optim adamw_torch \
    --eval_strategy steps \
    --eval_steps 100 \
    --save_steps 5000 \
    --output_dir ~/hdd/meta-llama3-8b-GANPO \
    --no_remove_unused_columns \
    --report_to wandb \
    --save_only_model True
