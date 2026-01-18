#!/bin/bash
#SBATCH --job-name              unilip
#SBATCH --partition             h800_batch
#SBATCH --nodes                 1
#SBATCH --tasks-per-node        1
#SBATCH --time                  72:00:00
#SBATCH --mem                   100G
#SBATCH --cpus-per-task         8
#SBATCH --gres                  gpu:2
#SBATCH --output                %j.out
#SBATCH --error                 %j.err


# You can modify the above job parameters
# gres: Number of GPU you want to occupy in this job
# mem: Memory you want to occupy in this job
# time: time limit of this job
# partition: node partition you want to use in this job
# cpus-per-task: cpu cores you want to request in this job

# Insert your commands here
cd ~/task/UniLIP

conda init
conda activate UniLIP

torchrun --nproc_per_node=2 train_csgo.py --csgo_config csgo_configs/exp5.yaml --deepspeed deepspeed_scripts/zero0.json --model_name_or_path UniLIP-1B --unilip_factor 10.6 --mllm_hf_path OpenGVLab/InternVL3-1B-hf --version internvl --data_type "mix" --csgo_image_folder data/preprocessed_data --mm_use_im_start_end False --mm_use_im_patch_token False --bf16 True --output_dir outputs/csgo_1b/exp5 --num_train_epochs 100 --per_device_train_batch_size 64 --per_device_eval_batch_size 64 --gradient_accumulation_steps 1 --eval_strategy "no" --save_strategy "steps" --save_steps 5000 --save_total_limit 1 --learning_rate 1e-4 --weight_decay 0. --warmup_ratio 0.003 --lr_scheduler_type "cosine_with_min_lr" --model_max_length 1024 --logging_steps 1 --tf32 True --gradient_checkpointing True --dataloader_num_workers 4 --lazy_preprocess True --n_query 256 --n_und_query 0 --report_to wandb --fix_dit False --fix_connect False --fix_llm True --action_dit_layer 3