# envs
1. benchmark_csgo.py

pip install torch torchvision torchmetrics[image] transformers pillow numpy tqdm scipy





# exp



## exp0
MASTER_ADDR=127.0.0.1 MASTER_PORT=29505 RANK=0 LOCAL_RANK=0 WORLD_SIZE=1

**train_csgo.py**
1. ```      CUDA_VISIBLE_DEVICES=1 MASTER_ADDR=127.0.0.1 MASTER_PORT=29505 RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 python train_csgo.py --csgo_config csgo_configs/exp0.yaml --deepspeed deepspeed_scripts/zero0.json --model_name_or_path UniLIP-1B --unilip_factor 10.6 --mllm_hf_path OpenGVLab/InternVL3-1B-hf --version internvl --data_type "mix" --csgo_image_folder data/preprocessed_data --mm_use_im_start_end False --mm_use_im_patch_token False --bf16 True --output_dir outputs/csgo_1b/exp0 --num_train_epochs 10 --per_device_train_batch_size 32 --per_device_eval_batch_size 32 --gradient_accumulation_steps 1 --eval_strategy "no" --save_strategy "steps" --save_steps 10 --save_total_limit 1 --learning_rate 1e-4 --weight_decay 0. --warmup_ratio 0.003 --lr_scheduler_type "cosine_with_min_lr" --model_max_length 1024 --logging_steps 1 --tf32 True --gradient_checkpointing True --dataloader_num_workers 16 --lazy_preprocess True --n_query 256 --n_und_query 0 --report_to wandb --fix_dit False --fix_connect False --fix_llm True     ```
2. debug ```      CUDA_VISIBLE_DEVICES=1 MASTER_ADDR=127.0.0.1 MASTER_PORT=29505 RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 python train_csgo.py --csgo_config csgo_configs/exp0.yaml --deepspeed deepspeed_scripts/zero0.json --model_name_or_path UniLIP-1B --unilip_factor 10.6 --mllm_hf_path OpenGVLab/InternVL3-1B-hf --version internvl --data_type "mix" --csgo_image_folder data/preprocessed_data --mm_use_im_start_end False --mm_use_im_patch_token False --bf16 True --output_dir outputs/csgo_1b/exp0_debug --num_train_epochs 1 --per_device_train_batch_size 2 --per_device_eval_batch_size 4 --gradient_accumulation_steps 1 --eval_strategy "no" --save_strategy "steps" --save_steps 20 --save_total_limit 1 --learning_rate 1e-4 --weight_decay 0. --warmup_ratio 0.003 --lr_scheduler_type "cosine_with_min_lr" --model_max_length 1024 --logging_steps 1 --tf32 True --gradient_checkpointing True --dataloader_num_workers 16 --lazy_preprocess True --n_query 256 --n_und_query 0 --report_to wandb --fix_dit False --fix_connect False --fix_llm True     ```

**eval_csgo.py**
``    CUDA_VISIBLE_DEVICES=1 python eval_csgo.py --csgo_config csgo_configs/test/exp0.yaml    ``



## exp0_1
MASTER_ADDR=127.0.0.1 MASTER_PORT=29505 RANK=0 LOCAL_RANK=0 WORLD_SIZE=1

**train_csgo.py**
```      CUDA_VISIBLE_DEVICES=1 MASTER_ADDR=127.0.0.1 MASTER_PORT=29505 RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 python train_csgo.py --csgo_config csgo_configs/exp0_1.yaml --deepspeed deepspeed_scripts/zero0.json --model_name_or_path UniLIP-1B --unilip_factor 10.6 --mllm_hf_path OpenGVLab/InternVL3-1B-hf --version internvl --data_type "mix" --csgo_image_folder data/preprocessed_data --mm_use_im_start_end False --mm_use_im_patch_token False --bf16 True --output_dir outputs/csgo_1b/exp0_1 --num_train_epochs 100 --per_device_train_batch_size 32 --per_device_eval_batch_size 32 --gradient_accumulation_steps 1 --eval_strategy "no" --save_strategy "steps" --save_steps 1000 --save_total_limit 1 --learning_rate 1e-4 --weight_decay 0. --warmup_ratio 0.003 --lr_scheduler_type "cosine_with_min_lr" --model_max_length 1024 --logging_steps 1 --tf32 True --gradient_checkpointing True --dataloader_num_workers 16 --lazy_preprocess True --n_query 256 --n_und_query 0 --report_to wandb --fix_dit False --fix_connect False --fix_llm True     ```

**eval_csgo.py**
``    CUDA_VISIBLE_DEVICES=1 python eval_csgo.py --csgo_config csgo_configs/test/exp0_1.yaml    ``



## exp1
MASTER_ADDR=127.0.0.1 MASTER_PORT=29505 RANK=0 LOCAL_RANK=0 WORLD_SIZE=1

- dust2

**train_csgo.py**
```     CUDA_VISIBLE_DEVICES=1 python train_csgo.py --csgo_config csgo_configs/exp1.yaml --deepspeed deepspeed_scripts/zero0.json --model_name_or_path UniLIP-1B --unilip_factor 10.6 --mllm_hf_path OpenGVLab/InternVL3-1B-hf --version internvl --data_type "mix" --csgo_image_folder data/preprocessed_data --mm_use_im_start_end False --mm_use_im_patch_token False --bf16 True --output_dir outputs/csgo_1b/exp1 --num_train_epochs 100 --per_device_train_batch_size 128 --per_device_eval_batch_size 128 --gradient_accumulation_steps 1 --eval_strategy "no" --save_strategy "steps" --save_steps 5000 --save_total_limit 1 --learning_rate 1e-4 --weight_decay 0. --warmup_ratio 0.003 --lr_scheduler_type "cosine_with_min_lr" --model_max_length 1024 --logging_steps 1 --tf32 True --gradient_checkpointing True --dataloader_num_workers 16 --lazy_preprocess True --n_query 256 --n_und_query 0 --report_to wandb --fix_dit False --fix_connect False --fix_llm True       ```

**eval_csgo.py**
``    CUDA_VISIBLE_DEVICES=1 python eval_csgo.py --csgo_config csgo_configs/test/exp1.yaml    ``



## exp2
MASTER_ADDR=127.0.0.1 MASTER_PORT=29505 RANK=0 LOCAL_RANK=0 WORLD_SIZE=1

- 3maps

**train_csgo.py**
```     CUDA_VISIBLE_DEVICES=1 python train_csgo.py --csgo_config csgo_configs/exp2.yaml --deepspeed deepspeed_scripts/zero0.json --model_name_or_path UniLIP-1B --unilip_factor 10.6 --mllm_hf_path OpenGVLab/InternVL3-1B-hf --version internvl --data_type "mix" --csgo_image_folder data/preprocessed_data --mm_use_im_start_end False --mm_use_im_patch_token False --bf16 True --output_dir outputs/csgo_1b/exp2 --num_train_epochs 100 --per_device_train_batch_size 128 --per_device_eval_batch_size 128 --gradient_accumulation_steps 1 --eval_strategy "no" --save_strategy "steps" --save_steps 5000 --save_total_limit 2 --learning_rate 1e-4 --weight_decay 0. --warmup_ratio 0.003 --lr_scheduler_type "cosine_with_min_lr" --model_max_length 1024 --logging_steps 1 --tf32 True --gradient_checkpointing True --dataloader_num_workers 16 --lazy_preprocess True --n_query 256 --n_und_query 0 --report_to wandb --fix_dit False --fix_connect False --fix_llm True       ```

**eval_csgo.py**
``    CUDA_VISIBLE_DEVICES=1 python eval_csgo.py --csgo_config csgo_configs/test/exp2.yaml      ``
**continuous gen**
``    CUDA_VISIBLE_DEVICES=1 python eval_csgo.py --csgo_config csgo_configs/test/exp2_1.yaml        ``
**benchmark_csgo.py todo**
``      python eval_metrics.py --gt ./data/gt --pred ./data/pred       ``
``      python eval_metrics.py --gt ./data/gt --pred ./data/pred --fid --lpips      ``
``      python eval_metrics.py --gt ./data/gt --pred ./data/pred --all --batch_size 8       ``



## exp3
MASTER_ADDR=127.0.0.1 MASTER_PORT=29505 RANK=0 LOCAL_RANK=0 WORLD_SIZE=1

- 3maps
- Unified Multi-Task UniLIP (Localization + Generation)

**train_csgo.py**
```     CUDA_VISIBLE_DEVICES=1 python train_csgo.py --csgo_config csgo_configs/exp3.yaml --deepspeed deepspeed_scripts/zero0.json --model_name_or_path UniLIP-1B --unilip_factor 10.6 --mllm_hf_path OpenGVLab/InternVL3-1B-hf --version internvl --data_type "mix" --csgo_image_folder data/preprocessed_data --mm_use_im_start_end False --mm_use_im_patch_token False --bf16 True --output_dir outputs/csgo_1b/exp3 --num_train_epochs 100 --per_device_train_batch_size 128 --per_device_eval_batch_size 128 --gradient_accumulation_steps 1 --eval_strategy "no" --save_strategy "steps" --save_steps 5000 --save_total_limit 1 --learning_rate 1e-4 --weight_decay 0. --warmup_ratio 0.003 --lr_scheduler_type "cosine_with_min_lr" --model_max_length 1024 --logging_steps 1 --tf32 True --gradient_checkpointing True --dataloader_num_workers 4 --lazy_preprocess True --n_query 256 --n_und_query 0 --report_to wandb --fix_dit False --fix_connect False --fix_llm True --action_dit_layer 3       ```



## exp3_1
MASTER_ADDR=127.0.0.1 MASTER_PORT=29505 RANK=0 LOCAL_RANK=0 WORLD_SIZE=1

- Unified Multi-Task UniLIP (Localization + Generation)

- Debugging: Unified_UniLIP without Localization
- dust2

**train_csgo.py**
```     CUDA_VISIBLE_DEVICES=1 MASTER_ADDR=127.0.0.1 MASTER_PORT=29505 RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 python train_csgo.py --csgo_config csgo_configs/exp3_1.yaml --deepspeed deepspeed_scripts/zero0.json --model_name_or_path UniLIP-1B --unilip_factor 10.6 --mllm_hf_path OpenGVLab/InternVL3-1B-hf --version internvl --data_type "mix" --csgo_image_folder data/preprocessed_data --mm_use_im_start_end False --mm_use_im_patch_token False --bf16 True --output_dir outputs/csgo_1b/exp3_1 --num_train_epochs 100 --per_device_train_batch_size 128 --per_device_eval_batch_size 128 --gradient_accumulation_steps 1 --eval_strategy "no" --save_strategy "steps" --save_steps 1000 --save_total_limit 1 --learning_rate 1e-4 --weight_decay 0. --warmup_ratio 0.003 --lr_scheduler_type "cosine_with_min_lr" --model_max_length 1024 --logging_steps 1 --tf32 True --gradient_checkpointing True --dataloader_num_workers 4 --lazy_preprocess True --n_query 256 --n_und_query 0 --report_to wandb --fix_dit False --fix_connect False --fix_llm True --action_dit_layer 3       ```



