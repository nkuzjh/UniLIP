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
**frames to video**
``    python frames_to_video.py --img_dir outputs_eval/exp2_1/test_20260103_220021/gen_imgs/de_dust2 --gt_dir data/preprocessed_data/de_dust2/imgs --output_dir outputs_eval/exp2_1/test_20260103_220021/gen_compared_videos/de_dust2 --max_duration 10        ``
``    python frames_to_video.py --img_dir outputs_eval/exp2_1/test_20260103_220021/gen_imgs/de_nuke --gt_dir data/preprocessed_data/de_nuke/imgs --output_dir outputs_eval/exp2_1/test_20260103_220021/gen_compared_videos/de_nuke --max_duration 10        ``
``    python frames_to_video.py --img_dir outputs_eval/exp2_1/test_20260103_220021/gen_imgs/de_ancient --gt_dir data/preprocessed_data/de_ancient/imgs --output_dir outputs_eval/exp2_1/test_20260103_220021/gen_compared_videos/de_ancient --max_duration 10        ``
**benchmark_csgo.py**
``      python benchmark_csgo.py --gt data/preprocessed_data/de_dust2/imgs --pred outputs_eval/exp2_1/test_20260103_220021/gen_imgs/de_dust2 --all --batch_size 8       ``
``      python benchmark_csgo.py --gt data/preprocessed_data/de_nuke/imgs --pred outputs_eval/exp2_1/test_20260103_220021/gen_imgs/de_nuke --all --batch_size 8      ``
``      python benchmark_csgo.py --gt data/preprocessed_data/de_ancient/imgs --pred outputs_eval/exp2_1/test_20260103_220021/gen_imgs/de_ancient --all --batch_size 8       ``



## exp3_1
MASTER_ADDR=127.0.0.1 MASTER_PORT=29505 RANK=0 LOCAL_RANK=0 WORLD_SIZE=1

- Unified Multi-Task UniLIP (Localization + Generation)

- ~~Debugging: Unified_UniLIP without Localization~~
- dust2
- debug_num_train_data=5000

**train_csgo.py**
```     CUDA_VISIBLE_DEVICES=1 MASTER_ADDR=127.0.0.1 MASTER_PORT=29505 RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 python train_csgo.py --csgo_config csgo_configs/exp3_1.yaml --deepspeed deepspeed_scripts/zero0.json --model_name_or_path UniLIP-1B --unilip_factor 10.6 --mllm_hf_path OpenGVLab/InternVL3-1B-hf --version internvl --data_type "mix" --csgo_image_folder data/preprocessed_data --mm_use_im_start_end False --mm_use_im_patch_token False --bf16 True --output_dir outputs/csgo_1b/exp3_1 --num_train_epochs 100 --per_device_train_batch_size 128 --per_device_eval_batch_size 128 --gradient_accumulation_steps 1 --eval_strategy "no" --save_strategy "steps" --save_steps 1000 --save_total_limit 1 --learning_rate 1e-4 --weight_decay 0. --warmup_ratio 0.003 --lr_scheduler_type "cosine_with_min_lr" --model_max_length 1024 --logging_steps 1 --tf32 True --gradient_checkpointing True --dataloader_num_workers 4 --lazy_preprocess True --n_query 256 --n_und_query 0 --report_to wandb --fix_dit False --fix_connect False --fix_llm True --action_dit_layer 3       ```
**eval_csgo.py**
``    CUDA_VISIBLE_DEVICES=0 python eval_csgo.py --csgo_config csgo_configs/test/exp3_1.yaml      ``
**eval_csgo_loc.py**
``    python eval_csgo_loc.py --csgo_config csgo_configs/test/exp3_1_loc.yaml      ``
**continuous gen todo**
**frames to video todo**
**benchmark_csgo.py todo**
``      python benchmark_csgo.py --gt data/preprocessed_data/de_dust2/imgs --pred outputs_eval/exp3_1/test_20260109_013431/gen_imgs/de_dust2 --all --batch_size 8       ``



## exp3
MASTER_ADDR=127.0.0.1 MASTER_PORT=29505 RANK=0 LOCAL_RANK=0 WORLD_SIZE=1

- 3maps
- Unified Multi-Task UniLIP (Localization + Generation)

**train_csgo.py**
```     CUDA_VISIBLE_DEVICES=1 python train_csgo.py --csgo_config csgo_configs/exp3.yaml --deepspeed deepspeed_scripts/zero0.json --model_name_or_path UniLIP-1B --unilip_factor 10.6 --mllm_hf_path OpenGVLab/InternVL3-1B-hf --version internvl --data_type "mix" --csgo_image_folder data/preprocessed_data --mm_use_im_start_end False --mm_use_im_patch_token False --bf16 True --output_dir outputs/csgo_1b/exp3 --num_train_epochs 100 --per_device_train_batch_size 128 --per_device_eval_batch_size 128 --gradient_accumulation_steps 1 --eval_strategy "no" --save_strategy "steps" --save_steps 5000 --save_total_limit 1 --learning_rate 1e-4 --weight_decay 0. --warmup_ratio 0.003 --lr_scheduler_type "cosine_with_min_lr" --model_max_length 1024 --logging_steps 1 --tf32 True --gradient_checkpointing True --dataloader_num_workers 4 --lazy_preprocess True --n_query 256 --n_und_query 0 --report_to wandb --fix_dit False --fix_connect False --fix_llm True --action_dit_layer 3       ```
**eval_csgo.py**
``    CUDA_VISIBLE_DEVICES=1 python eval_csgo.py --csgo_config csgo_configs/test/exp3.yaml      ``
**eval_csgo_loc.py**
``    CUDA_VISIBLE_DEVICES=1 python eval_csgo_loc.py --csgo_config csgo_configs/test/exp3_loc.yaml      ``
**benchmark_csgo.py**
``      python benchmark_csgo.py --gt data/preprocessed_data/de_dust2/imgs --pred outputs_eval/exp3/test_20260114_190325/gen_imgs/de_dust2 --all --batch_size 8       ``
``      python benchmark_csgo.py --gt data/preprocessed_data/de_nuke/imgs --pred outputs_eval/exp3/test_20260114_190325/gen_imgs/de_nuke --all --batch_size 8      ``
``      python benchmark_csgo.py --gt data/preprocessed_data/de_ancient/imgs --pred outputs_eval/exp3/test_20260114_190325/gen_imgs/de_ancient --all --batch_size 8       ``



## exp4_1
MASTER_ADDR=127.0.0.1 MASTER_PORT=29505 RANK=0 LOCAL_RANK=0 WORLD_SIZE=1

- Unified Multi-Task UniLIP (Localization + Generation)
- is_action_dit_dense_timestep: True

- dust2
- debug_num_train_data=5000

**train_csgo.py**
```     CUDA_VISIBLE_DEVICES=1 MASTER_ADDR=127.0.0.1 MASTER_PORT=29504 RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 python train_csgo.py --csgo_config csgo_configs/exp4_1.yaml --deepspeed deepspeed_scripts/zero0.json --model_name_or_path UniLIP-1B --unilip_factor 10.6 --mllm_hf_path OpenGVLab/InternVL3-1B-hf --version internvl --data_type "mix" --csgo_image_folder data/preprocessed_data --mm_use_im_start_end False --mm_use_im_patch_token False --bf16 True --output_dir outputs/csgo_1b/exp4_1 --num_train_epochs 100 --per_device_train_batch_size 64 --per_device_eval_batch_size 64 --gradient_accumulation_steps 1 --eval_strategy "no" --save_strategy "steps" --save_steps 2000 --save_total_limit 1 --learning_rate 1e-4 --weight_decay 0. --warmup_ratio 0.003 --lr_scheduler_type "cosine_with_min_lr" --model_max_length 1024 --logging_steps 1 --tf32 True --gradient_checkpointing True --dataloader_num_workers 4 --lazy_preprocess True --n_query 256 --n_und_query 0 --report_to wandb --fix_dit False --fix_connect False --fix_llm True --action_dit_layer 3       ```
**eval_csgo.py** 7800
``    CUDA_VISIBLE_DEVICES=1 python eval_csgo.py --csgo_config csgo_configs/test/exp4_1.yaml      ``
**eval_csgo_loc.py** 7800
``    CUDA_VISIBLE_DEVICES=1 python eval_csgo_loc.py --csgo_config csgo_configs/test/exp4_1_loc.yaml      ``
**benchmark_csgo.py todo**
``      python benchmark_csgo.py --gt data/preprocessed_data/de_dust2/imgs --pred outputs_eval/exp4_1/test_20260114_203910/gen_imgs/de_dust2 --all --batch_size 8       ``



## exp4
MASTER_ADDR=127.0.0.1 MASTER_PORT=29505 RANK=0 LOCAL_RANK=0 WORLD_SIZE=1

- Unified Multi-Task UniLIP (Localization + Generation)
- is_action_dit_dense_timestep: True

- 3maps
- 修复loc_head bug前step=5000
- 修复loc_head bug后重新跑 at fst

**train_csgo.py**
```     CUDA_VISIBLE_DEVICES=0 MASTER_ADDR=127.0.0.1 MASTER_PORT=29504 RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 python train_csgo.py --csgo_config csgo_configs/exp4.yaml --deepspeed deepspeed_scripts/zero0.json --model_name_or_path UniLIP-1B --unilip_factor 10.6 --mllm_hf_path OpenGVLab/InternVL3-1B-hf --version internvl --data_type "mix" --csgo_image_folder data/preprocessed_data --mm_use_im_start_end False --mm_use_im_patch_token False --bf16 True --output_dir outputs/csgo_1b/exp4 --num_train_epochs 100 --per_device_train_batch_size 128 --per_device_eval_batch_size 128 --gradient_accumulation_steps 1 --eval_strategy "no" --save_strategy "steps" --save_steps 5000 --save_total_limit 1 --learning_rate 1e-4 --weight_decay 0. --warmup_ratio 0.003 --lr_scheduler_type "cosine_with_min_lr" --model_max_length 1024 --logging_steps 1 --tf32 True --gradient_checkpointing True --dataloader_num_workers 4 --lazy_preprocess True --n_query 256 --n_und_query 0 --report_to wandb --fix_dit False --fix_connect False --fix_llm True --action_dit_layer 3       ```
```     CUDA_VISIBLE_DEVICES=0,1 MASTER_ADDR=127.0.0.1 MASTER_PORT=29504 RANK=0 LOCAL_RANK=0 WORLD_SIZE=2 python train_csgo.py --csgo_config csgo_configs/exp4.yaml --deepspeed deepspeed_scripts/zero0.json --model_name_or_path UniLIP-1B --unilip_factor 10.6 --mllm_hf_path OpenGVLab/InternVL3-1B-hf --version internvl --data_type "mix" --csgo_image_folder data/preprocessed_data --mm_use_im_start_end False --mm_use_im_patch_token False --bf16 True --output_dir outputs/csgo_1b/exp4 --num_train_epochs 100 --per_device_train_batch_size 64 --per_device_eval_batch_size 64 --gradient_accumulation_steps 1 --eval_strategy "no" --save_strategy "steps" --save_steps 5000 --save_total_limit 1 --learning_rate 1e-4 --weight_decay 0. --warmup_ratio 0.003 --lr_scheduler_type "cosine_with_min_lr" --model_max_length 1024 --logging_steps 1 --tf32 True --gradient_checkpointing True --dataloader_num_workers 4 --lazy_preprocess True --n_query 256 --n_und_query 0 --report_to wandb --fix_dit False --fix_connect False --fix_llm True --action_dit_layer 3       ```
```     CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train_csgo.py --csgo_config csgo_configs/exp4.yaml --deepspeed deepspeed_scripts/zero0.json --model_name_or_path UniLIP-1B --unilip_factor 10.6 --mllm_hf_path OpenGVLab/InternVL3-1B-hf --version internvl --data_type "mix" --csgo_image_folder data/preprocessed_data --mm_use_im_start_end False --mm_use_im_patch_token False --bf16 True --output_dir outputs/csgo_1b/exp4 --num_train_epochs 100 --per_device_train_batch_size 64 --per_device_eval_batch_size 64 --gradient_accumulation_steps 1 --eval_strategy "no" --save_strategy "steps" --save_steps 5000 --save_total_limit 1 --learning_rate 1e-4 --weight_decay 0. --warmup_ratio 0.003 --lr_scheduler_type "cosine_with_min_lr" --model_max_length 1024 --logging_steps 1 --tf32 True --gradient_checkpointing True --dataloader_num_workers 4 --lazy_preprocess True --n_query 256 --n_und_query 0 --report_to wandb --fix_dit False --fix_connect False --fix_llm True --action_dit_layer 3       ```
**eval_csgo.py** 5000
``    CUDA_VISIBLE_DEVICES=1 python eval_csgo.py --csgo_config csgo_configs/test/exp4.yaml      ``
**eval_csgo_loc.py** 5000
``    CUDA_VISIBLE_DEVICES=1 python eval_csgo_loc.py --csgo_config csgo_configs/test/exp4_loc.yaml      ``
**benchmark_csgo.py**
``      python benchmark_csgo.py --gt data/preprocessed_data/de_dust2/imgs --pred outputs_eval/exp4/test_20260115_233837/gen_imgs/de_dust2 --all --batch_size 8       ``
``      python benchmark_csgo.py --gt data/preprocessed_data/de_nuke/imgs --pred outputs_eval/exp4/test_20260115_233837/gen_imgs/de_nuke --all --batch_size 8      ``
``      python benchmark_csgo.py --gt data/preprocessed_data/de_ancient/imgs --pred outputs_eval/exp4/test_20260115_233837/gen_imgs/de_ancient --all --batch_size 8       ``



## exp4_2
MASTER_ADDR=127.0.0.1 MASTER_PORT=29505 RANK=0 LOCAL_RANK=0 WORLD_SIZE=1

- Unified Multi-Task UniLIP (Localization + Generation)
- is_action_dit_dense_timestep: True
- 修复loc_head bug前step=5000
- 修复loc_head bug后重新跑 at fst

- dust2
- 仅定位任务；task_mix_ratio = 1.0

**train_csgo.py**
```     CUDA_VISIBLE_DEVICES=0 MASTER_ADDR=127.0.0.1 MASTER_PORT=29504 RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 python train_csgo.py --csgo_config csgo_configs/exp4_2.yaml --deepspeed deepspeed_scripts/zero0.json --model_name_or_path UniLIP-1B --unilip_factor 10.6 --mllm_hf_path OpenGVLab/InternVL3-1B-hf --version internvl --data_type "mix" --csgo_image_folder data/preprocessed_data --mm_use_im_start_end False --mm_use_im_patch_token False --bf16 True --output_dir outputs/csgo_1b/exp4_2 --num_train_epochs 50 --per_device_train_batch_size 64 --per_device_eval_batch_size 64 --gradient_accumulation_steps 1 --eval_strategy "no" --save_strategy "steps" --save_steps 2000 --save_total_limit 1 --learning_rate 1e-4 --weight_decay 0. --warmup_ratio 0.003 --lr_scheduler_type "cosine_with_min_lr" --model_max_length 1024 --logging_steps 1 --tf32 True --gradient_checkpointing True --dataloader_num_workers 4 --lazy_preprocess True --n_query 256 --n_und_query 0 --report_to wandb --fix_dit False --fix_connect False --fix_llm True --action_dit_layer 3       ```
**eval_csgo_loc.py** 8000
``    CUDA_VISIBLE_DEVICES=0 python eval_csgo_loc.py --csgo_config csgo_configs/test/exp4_2_loc.yaml      ``


## exp5_1_debug
copy from exp5_1, debug loc_branch

**train_csgo.py**
```     CUDA_VISIBLE_DEVICES=1 MASTER_ADDR=127.0.0.1 MASTER_PORT=29500 RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 python train_csgo.py --csgo_config csgo_configs/exp5_1_debug.yaml --deepspeed deepspeed_scripts/zero0.json --model_name_or_path UniLIP-1B --unilip_factor 10.6 --mllm_hf_path OpenGVLab/InternVL3-1B-hf --version internvl --data_type "mix" --csgo_image_folder data/preprocessed_data --mm_use_im_start_end False --mm_use_im_patch_token False --bf16 True --output_dir outputs/csgo_1b/exp5_1_debug --num_train_epochs 100 --per_device_train_batch_size 32 --per_device_eval_batch_size 32 --gradient_accumulation_steps 1 --eval_strategy "no" --save_strategy "steps" --save_steps 1000 --save_total_limit 1 --learning_rate 1e-4 --weight_decay 0. --warmup_ratio 0.003 --lr_scheduler_type "cosine_with_min_lr" --model_max_length 1024 --logging_steps 1 --tf32 True --gradient_checkpointing True --dataloader_num_workers 4 --lazy_preprocess True --n_query 256 --n_und_query 0 --report_to wandb --fix_dit False --fix_connect False --fix_llm True --action_dit_layer 3       ```



## exp5_1
MASTER_ADDR=127.0.0.1 MASTER_PORT=29505 RANK=0 LOCAL_RANK=0 WORLD_SIZE=1

- Unified Multi-Task UniLIP (Localization + Generation)
- is_action_dit_dense_timestep: True

- dust2
- debug_num_train_data=10000
- is_loc_aux_loss
- 修复了loc_head中对action_dit_forward_with_adarmscond输出的hidden_state在整个序列上计算MSELoss的bug；正确的做法是，只计算seq+1位置的action state

**train_csgo.py**
```     CUDA_VISIBLE_DEVICES=1 MASTER_ADDR=127.0.0.1 MASTER_PORT=29500 RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 python train_csgo.py --csgo_config csgo_configs/exp5_1.yaml --deepspeed deepspeed_scripts/zero0.json --model_name_or_path UniLIP-1B --unilip_factor 10.6 --mllm_hf_path OpenGVLab/InternVL3-1B-hf --version internvl --data_type "mix" --csgo_image_folder data/preprocessed_data --mm_use_im_start_end False --mm_use_im_patch_token False --bf16 True --output_dir outputs/csgo_1b/exp5_1 --num_train_epochs 100 --per_device_train_batch_size 32 --per_device_eval_batch_size 32 --gradient_accumulation_steps 1 --eval_strategy "no" --save_strategy "steps" --save_steps 1000 --save_total_limit 1 --learning_rate 1e-4 --weight_decay 0. --warmup_ratio 0.003 --lr_scheduler_type "cosine_with_min_lr" --model_max_length 1024 --logging_steps 1 --tf32 True --gradient_checkpointing True --dataloader_num_workers 4 --lazy_preprocess True --n_query 256 --n_und_query 0 --report_to wandb --fix_dit False --fix_connect False --fix_llm True --action_dit_layer 3       ```
**eval_csgo.py** 4000 7800
``    CUDA_VISIBLE_DEVICES=1 python eval_csgo.py --csgo_config csgo_configs/test/exp5_1.yaml      ``
**eval_csgo_loc.py** 4000 7800
``    CUDA_VISIBLE_DEVICES=1 python eval_csgo_loc.py --csgo_config csgo_configs/test/exp5_1_loc.yaml      ``
**benchmark_csgo.py todo**
``      python benchmark_csgo.py --gt data/preprocessed_data/de_dust2/imgs --pred outputs_eval/exp5_1/test_20260116_132239/gen_imgs/de_dust2 --all --batch_size 8       ``



## exp5
MASTER_ADDR=127.0.0.1 MASTER_PORT=29505 RANK=0 LOCAL_RANK=0 WORLD_SIZE=1

- Unified Multi-Task UniLIP (Localization + Generation)
- is_action_dit_dense_timestep: True
- 修复loc_head bug，详细说明见exp5_1

- 3maps
- is_loc_aux_loss

**train_csgo.py**
```     CUDA_VISIBLE_DEVICES=1 MASTER_ADDR=127.0.0.1 MASTER_PORT=29504 RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 python train_csgo.py --csgo_config csgo_configs/exp5.yaml --deepspeed deepspeed_scripts/zero0.json --model_name_or_path UniLIP-1B --unilip_factor 10.6 --mllm_hf_path OpenGVLab/InternVL3-1B-hf --version internvl --data_type "mix" --csgo_image_folder data/preprocessed_data --mm_use_im_start_end False --mm_use_im_patch_token False --bf16 True --output_dir outputs/csgo_1b/exp5 --num_train_epochs 100 --per_device_train_batch_size 64 --per_device_eval_batch_size 64 --gradient_accumulation_steps 1 --eval_strategy "no" --save_strategy "steps" --save_steps 5000 --save_total_limit 1 --learning_rate 1e-4 --weight_decay 0. --warmup_ratio 0.003 --lr_scheduler_type "cosine_with_min_lr" --model_max_length 1024 --logging_steps 1 --tf32 True --gradient_checkpointing True --dataloader_num_workers 4 --lazy_preprocess True --n_query 256 --n_und_query 0 --report_to wandb --fix_dit False --fix_connect False --fix_llm True --action_dit_layer 3       ```
**eval_csgo.py** step=5000 60000
``    CUDA_VISIBLE_DEVICES=1 python eval_csgo.py --csgo_config csgo_configs/test/exp5.yaml      ``
    **benchmark_csgo.py todo**
    ``      CUDA_VISIBLE_DEVICES=1 python benchmark_csgo.py --gt data/preprocessed_data/de_dust2/imgs --pred outputs_eval/exp5/test_20260124_232313/gen_imgs/de_dust2 --all --batch_size 8       ``
    ``      CUDA_VISIBLE_DEVICES=1 python benchmark_csgo.py --gt data/preprocessed_data/de_nuke/imgs --pred outputs_eval/exp5/test_20260124_232313/gen_imgs/de_nuke --all --batch_size 8       ``
    ``      CUDA_VISIBLE_DEVICES=1 python benchmark_csgo.py --gt data/preprocessed_data/de_ancient/imgs --pred outputs_eval/exp5/test_20260124_232313/gen_imgs/de_ancient --all --batch_size 8       ``
**continuous gen todo** step=65000
``    CUDA_VISIBLE_DEVICES=1 python eval_csgo.py --csgo_config csgo_configs/test/exp5_conti.yaml      ``
    **benchmark_csgo.py todo**
    ``      CUDA_VISIBLE_DEVICES=1 python benchmark_csgo.py --gt data/preprocessed_data/de_dust2/imgs --pred outputs_eval/exp5_conti/test_20260125_172550/gen_imgs/de_dust2 --all --batch_size 8       ``
    ``      CUDA_VISIBLE_DEVICES=1 python benchmark_csgo.py --gt data/preprocessed_data/de_nuke/imgs --pred outputs_eval/exp5_conti/test_20260125_172550/gen_imgs/de_nuke --all --batch_size 8       ``
    ``      CUDA_VISIBLE_DEVICES=1 python benchmark_csgo.py --gt data/preprocessed_data/de_ancient/imgs --pred outputs_eval/exp5_conti/test_20260125_172550/gen_imgs/de_ancient --all --batch_size 8       ``
    **frames to video todo**
**eval_csgo_loc.py** step=5000 60000 80000
``    CUDA_VISIBLE_DEVICES=1 python eval_csgo_loc.py --csgo_config csgo_configs/test/exp5_loc.yaml      ``



## exp6
MASTER_ADDR=127.0.0.1 MASTER_PORT=29505 RANK=0 LOCAL_RANK=0 WORLD_SIZE=1

- Unified Multi-Task UniLIP (Localization + Generation)
- is_action_dit_dense_timestep: True
- 修复loc_head bug，详细说明见exp5_1
- 3maps
- is_loc_aux_loss

- is_lora_llm_backbone: True

**train_csgo.py**
```     CUDA_VISIBLE_DEVICES=1 MASTER_ADDR=127.0.0.1 MASTER_PORT=29504 RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 python train_csgo.py --csgo_config csgo_configs/exp6.yaml --deepspeed deepspeed_scripts/zero0.json --model_name_or_path UniLIP-1B --unilip_factor 10.6 --mllm_hf_path OpenGVLab/InternVL3-1B-hf --version internvl --data_type "mix" --csgo_image_folder data/preprocessed_data --mm_use_im_start_end False --mm_use_im_patch_token False --bf16 True --output_dir outputs/csgo_1b/exp6 --num_train_epochs 100 --per_device_train_batch_size 128 --per_device_eval_batch_size 128 --gradient_accumulation_steps 1 --eval_strategy "no" --save_strategy "steps" --save_steps 5000 --save_total_limit 1 --learning_rate 1e-4 --weight_decay 0. --warmup_ratio 0.003 --lr_scheduler_type "cosine_with_min_lr" --model_max_length 1024 --logging_steps 1 --tf32 True --gradient_checkpointing True --dataloader_num_workers 4 --lazy_preprocess True --n_query 256 --n_und_query 0 --report_to wandb --fix_dit False --fix_connect False --fix_llm True --action_dit_layer 3       ```
