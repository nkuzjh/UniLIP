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
**eval_csgo_loc.py** 8000 15600
``    CUDA_VISIBLE_DEVICES=0 python eval_csgo_loc.py --csgo_config csgo_configs/test/exp4_2_loc.yaml      ``



## exp4_3~6
MASTER_ADDR=127.0.0.1 MASTER_PORT=29505 RANK=0 LOCAL_RANK=0 WORLD_SIZE=1

- Unified Multi-Task UniLIP (Localization + Generation)
- is_action_dit_dense_timestep: True
- 修复loc_head bug前step=5000
- 修复loc_head bug后重新跑 at fst
- dust2
- 仅定位任务；task_mix_ratio = 1.0

- 增加action_dit_norm
- 修改hidden_states的zero_padding
- 修改pos_id拼接后的继承

**20260201实验小结**
- 目前exp4_3 exp4_4等仅微调lr、norm、hidden_states的方法与exp4_2无异，diffusion预测都集中在0.5左右(Z和pitch由于label大多集中在0.5，所以显得预测准确，实际results.json中也是都在0.5不能覆盖样本实际不在0.5的情况)。
- 等同于**没有获取到fps的condition**，仅在map上最小化normed(0~1)的均值。
- exp4_5和exp4_6等初始化了新的action_dit_projector，导致diffusion预测更加粗放，5D loc的MSE全都更大(xy上千error, pitch几百，z和yaw几十)；
- 这说明**fps的condition依旧没有生效**，**且初始化的action_dit_projector没有正确训练**导致预测随机性更大。

**20260202已修复小结**
- eval和generate_action2代码导致prompt中的<Image> token未正确的替换为UniLIP的256*<IMAGE_CONTEXT> token，使得fps和map的embeds没有拼接到hidden_states上。
- 虽然这与aciton_dit_norm、zero_padding、pos_id继承、action_dit_projector没有关系，但还是保持新增的这两个模块作为新的baseline进行exp8实验训练。
- 其次这个问题的修复也与balanceddataset没有关系，但为了优化显存提高训练速度，同样把balanceddataset作为baseline进行exp8实验训练。

**train_csgo.py**
- exp4_3_lr1e-4
```     CUDA_VISIBLE_DEVICES=1 MASTER_ADDR=127.0.0.1 MASTER_PORT=29504 RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 python train_csgo.py --csgo_config csgo_configs/exp4_3.yaml --deepspeed deepspeed_scripts/zero0.json --model_name_or_path UniLIP-1B --unilip_factor 10.6 --mllm_hf_path OpenGVLab/InternVL3-1B-hf --version internvl --data_type "mix" --csgo_image_folder data/preprocessed_data --mm_use_im_start_end False --mm_use_im_patch_token False --bf16 True --output_dir outputs/csgo_1b/exp4_3 --num_train_epochs 50 --per_device_train_batch_size 64 --per_device_eval_batch_size 64 --gradient_accumulation_steps 1 --eval_strategy "no" --save_strategy "steps" --save_steps 2000 --save_total_limit 1 --learning_rate 1e-4 --weight_decay 0. --warmup_ratio 0.003 --lr_scheduler_type "cosine_with_min_lr" --model_max_length 1024 --logging_steps 1 --tf32 True --gradient_checkpointing True --dataloader_num_workers 4 --lazy_preprocess True --n_query 256 --n_und_query 0 --report_to wandb --fix_dit False --fix_connect False --fix_llm True --action_dit_layer 3       ```
**eval_csgo_loc.py** 65:8000 1e-4
``    CUDA_VISIBLE_DEVICES=1 python eval_csgo_loc.py --csgo_config csgo_configs/test/exp4_3_loc_65_1e-4.yaml      ``
- exp4_3_lr5e-5
```     CUDA_VISIBLE_DEVICES=0 MASTER_ADDR=127.0.0.1 MASTER_PORT=29503 RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 python train_csgo.py --csgo_config csgo_configs/exp4_3.yaml --deepspeed deepspeed_scripts/zero0.json --model_name_or_path UniLIP-1B --unilip_factor 10.6 --mllm_hf_path OpenGVLab/InternVL3-1B-hf --version internvl --data_type "mix" --csgo_image_folder data/preprocessed_data --mm_use_im_start_end False --mm_use_im_patch_token False --bf16 True --output_dir outputs/csgo_1b/exp4_3 --num_train_epochs 50 --per_device_train_batch_size 64 --per_device_eval_batch_size 64 --gradient_accumulation_steps 1 --eval_strategy "no" --save_strategy "steps" --save_steps 2000 --save_total_limit 1 --learning_rate 5e-5 --weight_decay 0. --warmup_ratio 0.003 --lr_scheduler_type "cosine_with_min_lr" --model_max_length 1024 --logging_steps 1 --tf32 True --gradient_checkpointing True --dataloader_num_workers 4 --lazy_preprocess True --n_query 256 --n_und_query 0 --report_to wandb --fix_dit False --fix_connect False --fix_llm True --action_dit_layer 3       ```
**eval_csgo_loc.py** fst:4000 5e-5
``    CUDA_VISIBLE_DEVICES=1 python eval_csgo_loc.py --csgo_config csgo_configs/test/exp4_3_loc_fst_5e-5.yaml      ``
- exp4_4_lr5e-5
```     CUDA_VISIBLE_DEVICES=1 MASTER_ADDR=127.0.0.1 MASTER_PORT=29503 RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 python train_csgo.py --csgo_config csgo_configs/exp4_4.yaml --deepspeed deepspeed_scripts/zero0.json --model_name_or_path UniLIP-1B --unilip_factor 10.6 --mllm_hf_path OpenGVLab/InternVL3-1B-hf --version internvl --data_type "mix" --csgo_image_folder data/preprocessed_data --mm_use_im_start_end False --mm_use_im_patch_token False --bf16 True --output_dir outputs/csgo_1b/exp4_4 --num_train_epochs 50 --per_device_train_batch_size 64 --per_device_eval_batch_size 64 --gradient_accumulation_steps 1 --eval_strategy "no" --save_strategy "steps" --save_steps 2000 --save_total_limit 1 --learning_rate 5e-5 --weight_decay 0. --warmup_ratio 0.003 --lr_scheduler_type "cosine_with_min_lr" --model_max_length 1024 --logging_steps 1 --tf32 True --gradient_checkpointing True --dataloader_num_workers 4 --lazy_preprocess True --n_query 256 --n_und_query 0 --report_to wandb --fix_dit False --fix_connect False --fix_llm True --action_dit_layer 3       ```
**eval_csgo_loc.py** 21:24000
``    CUDA_VISIBLE_DEVICES=0 python eval_csgo_loc.py --csgo_config csgo_configs/test/exp4_4_loc.yaml      ``
**eval_csgo_loc.py** fst:8000 12000
``    CUDA_VISIBLE_DEVICES=1 python eval_csgo_loc.py --csgo_config csgo_configs/test/exp4_4_loc.yaml      ``
**eval_csgo_loc.py** 65:8000
``    CUDA_VISIBLE_DEVICES=1 python eval_csgo_loc.py --csgo_config csgo_configs/test/exp4_4_loc.yaml      ``
注意在下面加入exp7之后，如果配置balancedataset(is_multi_task_balanced=True)，则仅定位任务的配置(task_mix_ratio = 1.0)将失效，强制同一个batch内loc和gen样本数量均衡
- exp4_5_lr1e-4_locditproj + exp7
```     CUDA_VISIBLE_DEVICES=3 MASTER_ADDR=127.0.0.1 MASTER_PORT=29502 RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 python train_csgo.py --csgo_config csgo_configs/exp4_5.yaml --deepspeed deepspeed_scripts/zero0.json --model_name_or_path UniLIP-1B --unilip_factor 10.6 --mllm_hf_path OpenGVLab/InternVL3-1B-hf --version internvl --data_type "mix" --csgo_image_folder data/preprocessed_data --mm_use_im_start_end False --mm_use_im_patch_token False --bf16 True --output_dir outputs/csgo_1b/exp4_5 --num_train_epochs 50 --per_device_train_batch_size 32 --per_device_eval_batch_size 32 --gradient_accumulation_steps 1 --eval_strategy "no" --save_strategy "steps" --save_steps 2000 --save_total_limit 1 --learning_rate 1e-4 --weight_decay 0. --warmup_ratio 0.003 --lr_scheduler_type "cosine_with_min_lr" --model_max_length 1024 --logging_steps 1 --tf32 True --gradient_checkpointing True --dataloader_num_workers 4 --lazy_preprocess True --n_query 256 --n_und_query 0 --report_to wandb --fix_dit False --fix_connect False --fix_llm True --action_dit_layer 3       ``` 保存路径写错了，21上训练exp4_5但是ckpt在exp4_4
**eval_csgo_loc.py** fst:8000
``    CUDA_VISIBLE_DEVICES=0 python eval_csgo_loc.py --csgo_config csgo_configs/test/exp4_5_loc.yaml      ``
- exp4_5_1_lr1e-4_locditproj + exp7 \wo balancedataset
```     CUDA_VISIBLE_DEVICES=1 MASTER_ADDR=127.0.0.1 MASTER_PORT=29502 RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 python train_csgo.py --csgo_config csgo_configs/exp4_5_1.yaml --deepspeed deepspeed_scripts/zero0.json --model_name_or_path UniLIP-1B --unilip_factor 10.6 --mllm_hf_path OpenGVLab/InternVL3-1B-hf --version internvl --data_type "mix" --csgo_image_folder data/preprocessed_data --mm_use_im_start_end False --mm_use_im_patch_token False --bf16 True --output_dir outputs/csgo_1b/exp4_5_1 --num_train_epochs 50 --per_device_train_batch_size 64 --per_device_eval_batch_size 64 --gradient_accumulation_steps 1 --eval_strategy "no" --save_strategy "steps" --save_steps 2000 --save_total_limit 1 --learning_rate 1e-4 --weight_decay 0. --warmup_ratio 0.003 --lr_scheduler_type "cosine_with_min_lr" --model_max_length 1024 --logging_steps 1 --tf32 True --gradient_checkpointing True --dataloader_num_workers 4 --lazy_preprocess True --n_query 256 --n_und_query 0 --report_to wandb --fix_dit False --fix_connect False --fix_llm True --action_dit_layer 3       ```
**eval_csgo_loc.py** 78:15600
``    CUDA_VISIBLE_DEVICES=1 python eval_csgo_loc.py --csgo_config csgo_configs/test/exp4_5_1_loc.yaml      ``
- exp4_6_lr1e-4_locditproj_locditly6 + exp7
```     CUDA_VISIBLE_DEVICES=1 MASTER_ADDR=127.0.0.1 MASTER_PORT=29504 RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 python train_csgo.py --csgo_config csgo_configs/exp4_6.yaml --deepspeed deepspeed_scripts/zero0.json --model_name_or_path UniLIP-1B --unilip_factor 10.6 --mllm_hf_path OpenGVLab/InternVL3-1B-hf --version internvl --data_type "mix" --csgo_image_folder data/preprocessed_data --mm_use_im_start_end False --mm_use_im_patch_token False --bf16 True --output_dir outputs/csgo_1b/exp4_6 --num_train_epochs 50 --per_device_train_batch_size 32 --per_device_eval_batch_size 32 --gradient_accumulation_steps 1 --eval_strategy "no" --save_strategy "steps" --save_steps 2000 --save_total_limit 1 --learning_rate 1e-4 --weight_decay 0. --warmup_ratio 0.003 --lr_scheduler_type "cosine_with_min_lr" --model_max_length 1024 --logging_steps 1 --tf32 True --gradient_checkpointing True --dataloader_num_workers 4 --lazy_preprocess True --n_query 256 --n_und_query 0 --report_to wandb --fix_dit False --fix_connect False --fix_llm True --action_dit_layer 6       ```
**eval_csgo_loc.py** 65:16000
``    CUDA_VISIBLE_DEVICES=0 python eval_csgo_loc.py --csgo_config csgo_configs/test/exp4_6_loc_65.yaml      ``
**eval_csgo_loc.py** 67:31250
``    CUDA_VISIBLE_DEVICES=0 python eval_csgo_loc.py --csgo_config csgo_configs/test/exp4_6_loc_67.yaml      ``
- exp4_6_1_lr1e-4_locditproj_locditly6 + exp7  \wo balancedataset
```     CUDA_VISIBLE_DEVICES=1 MASTER_ADDR=127.0.0.1 MASTER_PORT=29504 RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 python train_csgo.py --csgo_config csgo_configs/exp4_6_1.yaml --deepspeed deepspeed_scripts/zero0.json --model_name_or_path UniLIP-1B --unilip_factor 10.6 --mllm_hf_path OpenGVLab/InternVL3-1B-hf --version internvl --data_type "mix" --csgo_image_folder data/preprocessed_data --mm_use_im_start_end False --mm_use_im_patch_token False --bf16 True --output_dir outputs/csgo_1b/exp4_6_1 --num_train_epochs 50 --per_device_train_batch_size 64 --per_device_eval_batch_size 64 --gradient_accumulation_steps 1 --eval_strategy "no" --save_strategy "steps" --save_steps 2000 --save_total_limit 1 --learning_rate 1e-4 --weight_decay 0. --warmup_ratio 0.003 --lr_scheduler_type "cosine_with_min_lr" --model_max_length 1024 --logging_steps 1 --tf32 True --gradient_checkpointing True --dataloader_num_workers 4 --lazy_preprocess True --n_query 256 --n_und_query 0 --report_to wandb --fix_dit False --fix_connect False --fix_llm True --action_dit_layer 6       ```
**eval_csgo_loc.py** 65:15600
``    CUDA_VISIBLE_DEVICES=1 python eval_csgo_loc.py --csgo_config csgo_configs/test/exp4_6_1_loc.yaml      ``
- exp4_7_lr1e-4_loclrnq **todo** + exp7
```     CUDA_VISIBLE_DEVICES=1 MASTER_ADDR=127.0.0.1 MASTER_PORT=29504 RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 python train_csgo.py --csgo_config csgo_configs/exp4_6.yaml --deepspeed deepspeed_scripts/zero0.json --model_name_or_path UniLIP-1B --unilip_factor 10.6 --mllm_hf_path OpenGVLab/InternVL3-1B-hf --version internvl --data_type "mix" --csgo_image_folder data/preprocessed_data --mm_use_im_start_end False --mm_use_im_patch_token False --bf16 True --output_dir outputs/csgo_1b/exp4_6 --num_train_epochs 50 --per_device_train_batch_size 64 --per_device_eval_batch_size 64 --gradient_accumulation_steps 1 --eval_strategy "no" --save_strategy "steps" --save_steps 2000 --save_total_limit 1 --learning_rate 1e-4 --weight_decay 0. --warmup_ratio 0.003 --lr_scheduler_type "cosine_with_min_lr" --model_max_length 1024 --logging_steps 1 --tf32 True --gradient_checkpointing True --dataloader_num_workers 4 --lazy_preprocess True --n_query 256 --n_und_query 0 --report_to wandb --fix_dit False --fix_connect False --fix_llm True --action_dit_layer 3       ```



### exp4_5_2
- exp4_5_1_lr1e-4_locditproj + exp7 \wo balancedataset

- 3maps
- lr 1e-4
- action_dit_projector_lr: 0.0005

```     CUDA_VISIBLE_DEVICES=0 MASTER_ADDR=127.0.0.1 MASTER_PORT=29504 RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 python train_csgo.py --csgo_config csgo_configs/exp4_5_2.yaml --deepspeed deepspeed_scripts/zero0.json --model_name_or_path UniLIP-1B --unilip_factor 10.6 --mllm_hf_path OpenGVLab/InternVL3-1B-hf --version internvl --data_type "mix" --csgo_image_folder data/preprocessed_data --mm_use_im_start_end False --mm_use_im_patch_token False --bf16 True --output_dir outputs/csgo_1b/exp4_5_2 --num_train_epochs 20 --per_device_train_batch_size 64 --per_device_eval_batch_size 64 --gradient_accumulation_steps 1 --eval_strategy "no" --save_strategy "steps" --save_steps 2000 --save_total_limit 1 --learning_rate 1e-4 --weight_decay 0. --warmup_ratio 0.003 --lr_scheduler_type "cosine_with_min_lr" --model_max_length 1024 --logging_steps 1 --tf32 True --gradient_checkpointing True --dataloader_num_workers 4 --lazy_preprocess True --n_query 256 --n_und_query 0 --report_to wandb --fix_dit False --fix_connect False --fix_llm True --action_dit_layer 3       ```
**debug**
```     CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=29504 train_csgo.py --csgo_config csgo_configs/exp4_5_2.yaml --deepspeed deepspeed_scripts/zero0.json --model_name_or_path UniLIP-1B --unilip_factor 10.6 --mllm_hf_path OpenGVLab/InternVL3-1B-hf --version internvl --data_type "mix" --csgo_image_folder data/preprocessed_data --mm_use_im_start_end False --mm_use_im_patch_token False --bf16 True --output_dir outputs/csgo_1b/exp4_5_2_2cuda --num_train_epochs 50 --per_device_train_batch_size 64 --per_device_eval_batch_size 64 --gradient_accumulation_steps 1 --eval_strategy "no" --save_strategy "steps" --save_steps 2000 --save_total_limit 1 --learning_rate 1e-4 --weight_decay 0. --warmup_ratio 0.003 --lr_scheduler_type "cosine_with_min_lr" --model_max_length 1024 --logging_steps 1 --tf32 True --gradient_checkpointing True --dataloader_num_workers 4 --lazy_preprocess True --n_query 256 --n_und_query 0 --report_to wandb --fix_dit False --fix_connect False --fix_llm True --action_dit_layer 3       ```
**eval_csgo_loc.py** 67:18740 lr 1e-4
```    CUDA_VISIBLE_DEVICES=0 python eval_csgo_loc.py --csgo_config csgo_configs/test/exp4_5_2_loc.yaml      ```



### exp4_5_3
- exp4_5_1_lr1e-4_locditproj + exp7 \wo balancedataset
- 3maps

- lr 1e-4 #5e-4 #1e-3
- action_dit_projector_lr: 0.001
- bs 64

```     CUDA_VISIBLE_DEVICES=0 MASTER_ADDR=127.0.0.1 MASTER_PORT=29503 RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 python train_csgo.py --csgo_config csgo_configs/exp4_5_3.yaml --deepspeed deepspeed_scripts/zero0.json --model_name_or_path UniLIP-1B --unilip_factor 10.6 --mllm_hf_path OpenGVLab/InternVL3-1B-hf --version internvl --data_type "mix" --csgo_image_folder data/preprocessed_data --mm_use_im_start_end False --mm_use_im_patch_token False --bf16 True --output_dir outputs/csgo_1b/exp4_5_3 --num_train_epochs 20 --per_device_train_batch_size 64 --per_device_eval_batch_size 64 --gradient_accumulation_steps 1 --eval_strategy "no" --save_strategy "steps" --save_steps 2000 --save_total_limit 1 --learning_rate 1e-4 --weight_decay 0. --warmup_ratio 0.003 --lr_scheduler_type "cosine_with_min_lr" --model_max_length 1024 --logging_steps 1 --tf32 True --gradient_checkpointing True --dataloader_num_workers 4 --lazy_preprocess True --n_query 256 --n_und_query 0 --report_to wandb --fix_dit False --fix_connect False --fix_llm True --action_dit_layer 3       ```
**eval_csgo_loc.py** 78:8000 lr 1e-3 masked_loc_loss: 0.853434 eval乱猜
```    CUDA_VISIBLE_DEVICES=0 python eval_csgo_loc.py --csgo_config csgo_configs/test/exp4_5_3_loc.yaml      ```
**eval_csgo_loc.py** 67:8000 lr 5e-4 masked_loc_loss: 1.118271 eval乱猜
```    CUDA_VISIBLE_DEVICES=0 python eval_csgo_loc.py --csgo_config csgo_configs/test/exp4_5_3_loc.yaml      ```
**eval_csgo_loc.py** 67:8000 lr 1e-4 masked_loc_loss:  eval training
```    CUDA_VISIBLE_DEVICES=0 python eval_csgo_loc.py --csgo_config csgo_configs/test/exp4_5_3_loc.yaml      ```

### exp4_6_2
- exp4_5_1_lr1e-4_locditproj + exp7 \wo balancedataset
- 3maps
- lr 1e-4 #1e-3
- action_dit_projector_lr: 0.001
- bs 64 #128

- action_dit_layer 6

```     CUDA_VISIBLE_DEVICES=0 MASTER_ADDR=127.0.0.1 MASTER_PORT=29504 RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 python train_csgo.py --csgo_config csgo_configs/exp4_6_2.yaml --deepspeed deepspeed_scripts/zero0.json --model_name_or_path UniLIP-1B --unilip_factor 10.6 --mllm_hf_path OpenGVLab/InternVL3-1B-hf --version internvl --data_type "mix" --csgo_image_folder data/preprocessed_data --mm_use_im_start_end False --mm_use_im_patch_token False --bf16 True --output_dir outputs/csgo_1b/exp4_6_2 --num_train_epochs 50 --per_device_train_batch_size 64 --per_device_eval_batch_size 64 --gradient_accumulation_steps 1 --eval_strategy "no" --save_strategy "steps" --save_steps 2000 --save_total_limit 1 --learning_rate 1e-4 --weight_decay 0. --warmup_ratio 0.003 --lr_scheduler_type "cosine_with_min_lr" --model_max_length 1024 --logging_steps 1 --tf32 True --gradient_checkpointing True --dataloader_num_workers 4 --lazy_preprocess True --n_query 256 --n_und_query 0 --report_to wandb --fix_dit False --fix_connect False --fix_llm True --action_dit_layer 6       ```
**eval_csgo_loc.py** 67:4000 bs 128 lr 1e-3 masked_loc_loss: 0.933967 eval
```    CUDA_VISIBLE_DEVICES=0 python eval_csgo_loc.py --csgo_config csgo_configs/test/exp4_6_2_loc.yaml      ```
**eval_csgo_loc.py** 67: bs 64 lr 1e-4 masked_loc_loss:  eval training
```    CUDA_VISIBLE_DEVICES=0 python eval_csgo_loc.py --csgo_config csgo_configs/test/exp4_6_2_loc.yaml      ```

### exp4_10
- exp4_5_1_lr1e-4_locditproj + exp7 \wo balancedataset
- 3maps
- lr 1e-4
- action_dit_projector_lr: 0.001
- bs 64

- is_aciton_dit_vae_small_init: True

```     CUDA_VISIBLE_DEVICES=0 MASTER_ADDR=127.0.0.1 MASTER_PORT=29503 RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 python train_csgo.py --csgo_config csgo_configs/exp4_10.yaml --deepspeed deepspeed_scripts/zero0.json --model_name_or_path UniLIP-1B --unilip_factor 10.6 --mllm_hf_path OpenGVLab/InternVL3-1B-hf --version internvl --data_type "mix" --csgo_image_folder data/preprocessed_data --mm_use_im_start_end False --mm_use_im_patch_token False --bf16 True --output_dir outputs/csgo_1b/exp4_10 --num_train_epochs 20 --per_device_train_batch_size 64 --per_device_eval_batch_size 64 --gradient_accumulation_steps 1 --eval_strategy "no" --save_strategy "steps" --save_steps 2000 --save_total_limit 1 --learning_rate 1e-4 --weight_decay 0. --warmup_ratio 0.003 --lr_scheduler_type "cosine_with_min_lr" --model_max_length 1024 --logging_steps 1 --tf32 True --gradient_checkpointing True --dataloader_num_workers 4 --lazy_preprocess True --n_query 256 --n_und_query 0 --report_to wandb --fix_dit False --fix_connect False --fix_llm True --action_dit_layer 3       ```



## exp5_1_debug
copy from exp5_1, debug loc_branch

**train_csgo.py**
```     CUDA_VISIBLE_DEVICES=1 MASTER_ADDR=127.0.0.1 MASTER_PORT=29500 RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 python train_csgo.py --csgo_config csgo_configs/exp5_1_debug.yaml --deepspeed deepspeed_scripts/zero0.json --model_name_or_path UniLIP-1B --unilip_factor 10.6 --mllm_hf_path OpenGVLab/InternVL3-1B-hf --version internvl --data_type "mix" --csgo_image_folder data/preprocessed_data --mm_use_im_start_end False --mm_use_im_patch_token False --bf16 True --output_dir outputs/csgo_1b/exp5_1_debug --num_train_epochs 100 --per_device_train_batch_size 8 --per_device_eval_batch_size 8 --gradient_accumulation_steps 1 --eval_strategy "no" --save_strategy "steps" --save_steps 1000 --save_total_limit 1 --learning_rate 1e-4 --weight_decay 0. --warmup_ratio 0.003 --lr_scheduler_type "cosine_with_min_lr" --model_max_length 1024 --logging_steps 1 --tf32 True --gradient_checkpointing True --dataloader_num_workers 4 --lazy_preprocess True --n_query 256 --n_und_query 0 --report_to wandb --fix_dit False --fix_connect False --fix_llm True --action_dit_layer 3       ```



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
**continuous gen** step=65000
``    CUDA_VISIBLE_DEVICES=1 python eval_csgo.py --csgo_config csgo_configs/test/exp5_conti.yaml      ``
    **benchmark_csgo.py**
    ``      CUDA_VISIBLE_DEVICES=1 python benchmark_csgo.py --gt data/preprocessed_data/de_dust2/imgs --pred outputs_eval/exp5_conti/test_20260125_172550/gen_imgs/de_dust2 --all --batch_size 8       ``
    ``      CUDA_VISIBLE_DEVICES=1 python benchmark_csgo.py --gt data/preprocessed_data/de_nuke/imgs --pred outputs_eval/exp5_conti/test_20260125_172550/gen_imgs/de_nuke --all --batch_size 8       ``
    ``      CUDA_VISIBLE_DEVICES=1 python benchmark_csgo.py --gt data/preprocessed_data/de_ancient/imgs --pred outputs_eval/exp5_conti/test_20260125_172550/gen_imgs/de_ancient --all --batch_size 8       ``
    **frames to video**
    ``    python frames_to_video.py --img_dir outputs_eval/exp5_conti/test_20260125_172550/gen_imgs/de_dust2 --gt_dir data/preprocessed_data/de_dust2/imgs --output_dir outputs_eval/exp5_conti/test_20260125_172550/gen_compared_videos/de_dust2 --max_duration 10        ``
    ``    python frames_to_video.py --img_dir outputs_eval/exp5_conti/test_20260125_172550/gen_imgs/de_nuke --gt_dir data/preprocessed_data/de_nuke/imgs --output_dir outputs_eval/exp5_conti/test_20260125_172550/gen_compared_videos/de_nuke --max_duration 10        ``
    ``    python frames_to_video.py --img_dir outputs_eval/exp5_conti/test_20260125_172550/gen_imgs/de_ancient --gt_dir data/preprocessed_data/de_ancient/imgs --output_dir outputs_eval/exp5_conti/test_20260125_172550/gen_compared_videos/de_ancient --max_duration 10        ``
**eval_csgo_loc.py** step=5000 60000 80000 90000
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


## exp7
**改造完成后整合进入其他exp运行**
- Action_Dit：增加gradient_checkpointing
- MultiTaskUniLIP代码：增加按照loc/gen切分数据进入branch的功能，节省显存
- MultiTaskDataset代码：增加loc/gen均衡采样的代码 **需单独配置yaml参数 is_multi_task_balanced: True**

    Batch Size 设置： 在 TrainingArguments 中，per_device_train_batch_size 代表的是 Sampler 采样的次数。如果设置 BS=64，实际上 Collator 会收到 64 个列表，展平后变成 128 个样本。
    Epoch 长度： 由于每个索引现在产生 2 个样本，实际上每个 Epoch 模型见到的数据量翻倍了。这是符合预期的多任务训练行为。
    建议：
        将原本的 BS 减半 (例如设为 64)，这样最终进入模型的 Batch Size 依然是 128 (64 Loc + 64 Gen)。
        如果 BS 减半， Epoch 相比单任务训练翻倍，那模型见到的数据量和之前一致，但是迭代次数会翻倍
        如果将 BS 不变，Epoch 不变， 那模型见到的数据量和迭代次数都和之前一致，但是需要显存翻倍（假设gen和loc所需显存一样）



## exp8
**exp5 + exp7 + exp4_5_1/exp4_6_1**
- 已在exp4_5_1和exp4_6_1修复了eval和generate_action2代码导致的定位推理结果都等于0.5的问题
**exp5**
- Unified Multi-Task UniLIP (Localization + Generation)
- is_action_dit_dense_timestep: True
- 修复loc_head bug，详细说明见exp5_1
- 3maps
- is_loc_aux_loss
**exp7**
- Action_Dit：增加gradient_checkpointing
- MultiTaskUniLIP代码：增加按照loc/gen切分数据进入branch的功能，节省显存
- MultiTaskDataset代码：增加loc/gen均衡采样的代码。需单独配置yaml参数 is_multi_task_balanced: True，且会令参数task_mix_ratio用于控制定位/生成数据比例的功能失效
**exp4_5_1**
- 增加action_dit_norm
- 修改hidden_states的zero_padding
- 修改pos_id拼接后的继承
- is_action_dit_projector: True
**exp4_6_1暂不使用**
- action_dit_layer: 6 #仅exp4_6_1，在启动命令中实现

**train_csgo.py**
```     CUDA_VISIBLE_DEVICES=1 MASTER_ADDR=127.0.0.1 MASTER_PORT=29504 RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 python train_csgo.py --csgo_config csgo_configs/exp8.yaml --deepspeed deepspeed_scripts/zero0.json --model_name_or_path UniLIP-1B --unilip_factor 10.6 --mllm_hf_path OpenGVLab/InternVL3-1B-hf --version internvl --data_type "mix" --csgo_image_folder data/preprocessed_data --mm_use_im_start_end False --mm_use_im_patch_token False --bf16 True --output_dir outputs/csgo_1b/exp8 --num_train_epochs 100 --per_device_train_batch_size 64 --per_device_eval_batch_size 64 --gradient_accumulation_steps 1 --eval_strategy "no" --save_strategy "steps" --save_steps 2000 --save_total_limit 1 --learning_rate 1e-4 --weight_decay 0. --warmup_ratio 0.003 --lr_scheduler_type "cosine_with_min_lr" --model_max_length 1024 --logging_steps 1 --tf32 True --gradient_checkpointing True --dataloader_num_workers 4 --lazy_preprocess True --n_query 256 --n_und_query 0 --report_to wandb --fix_dit False --fix_connect False --fix_llm True --action_dit_layer 3       ```
**eval_csgo_loc.py** step=10000
```    CUDA_VISIBLE_DEVICES=1 python eval_csgo_loc.py --csgo_config csgo_configs/test/exp8_loc.yaml      ```
**eval_csgo.py** step=12000
``    CUDA_VISIBLE_DEVICES=1 python eval_csgo.py --csgo_config csgo_configs/test/exp8_gen.yaml      ``
    **benchmark_csgo.py**
    ``      CUDA_VISIBLE_DEVICES=1 python benchmark_csgo.py --gt data/preprocessed_data/de_dust2/imgs --pred outputs_eval/exp8_gen/test_20260204_195820/gen_imgs/de_dust2 --all --batch_size 8       ``
    ``      CUDA_VISIBLE_DEVICES=1 python benchmark_csgo.py --gt data/preprocessed_data/de_nuke/imgs --pred outputs_eval/exp8_gen/test_20260204_195820/gen_imgs/de_nuke --all --batch_size 8       ``
    ``      CUDA_VISIBLE_DEVICES=0 python benchmark_csgo.py --gt data/preprocessed_data/de_ancient/imgs --pred outputs_eval/exp8_gen/test_20260204_195820/gen_imgs/de_ancient --all --batch_size 8       ``
**continuous gen**  step=16000
``    CUDA_VISIBLE_DEVICES=1 python eval_csgo.py --csgo_config csgo_configs/test/exp8_gen.yaml      ``
    **benchmark_csgo.py**
    ``      CUDA_VISIBLE_DEVICES=1 python benchmark_csgo.py --gt data/preprocessed_data/de_dust2/imgs --pred outputs_eval/exp8_gen/test_20260205_134439/gen_imgs/de_dust2 --all --batch_size 8       ``
    ``      CUDA_VISIBLE_DEVICES=1 python benchmark_csgo.py --gt data/preprocessed_data/de_nuke/imgs --pred outputs_eval/exp8_gen/test_20260205_134439/gen_imgs/de_nuke --all --batch_size 8       ``
    ``      CUDA_VISIBLE_DEVICES=1 python benchmark_csgo.py --gt data/preprocessed_data/de_ancient/imgs --pred outputs_eval/exp8_gen/test_20260205_134439/gen_imgs/de_ancient --all --batch_size 8       ``
    **frames to video**
    ``    python frames_to_video.py --img_dir outputs_eval/exp8_gen/test_20260205_134439/gen_imgs/de_dust2 --gt_dir data/preprocessed_data/de_dust2/imgs --output_dir outputs_eval/exp8_gen/test_20260205_134439/gen_compared_videos/de_dust2 --max_duration 10        ``
    ``    python frames_to_video.py --img_dir outputs_eval/exp8_gen/test_20260205_134439/gen_imgs/de_nuke --gt_dir data/preprocessed_data/de_nuke/imgs --output_dir outputs_eval/exp8_gen/test_20260205_134439/gen_compared_videos/de_nuke --max_duration 10        ``
    ``    python frames_to_video.py --img_dir outputs_eval/exp8_gen/test_20260205_134439/gen_imgs/de_ancient --gt_dir data/preprocessed_data/de_ancient/imgs --output_dir outputs_eval/exp8_gen/test_20260205_134439/gen_compared_videos/de_ancient --max_duration 10        ``



## exp8_1
- epoch 50
- lr 1e-4 #5e-4
- alpha_loc_aux_loss: 0.1
- alpha_loc_loss: 2

**train_csgo.py**
```     CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=29504 train_csgo.py --csgo_config csgo_configs/exp8_1.yaml --deepspeed deepspeed_scripts/zero0.json --model_name_or_path UniLIP-1B --unilip_factor 10.6 --mllm_hf_path OpenGVLab/InternVL3-1B-hf --version internvl --data_type "mix" --csgo_image_folder data/preprocessed_data --mm_use_im_start_end False --mm_use_im_patch_token False --bf16 True --output_dir outputs/csgo_1b/exp8_1 --num_train_epochs 50 --per_device_train_batch_size 32 --per_device_eval_batch_size 32 --gradient_accumulation_steps 1 --eval_strategy "no" --save_strategy "steps" --save_steps 2000 --save_total_limit 1 --learning_rate 1e-4 --weight_decay 0. --warmup_ratio 0.003 --lr_scheduler_type "cosine_with_min_lr" --model_max_length 1024 --logging_steps 1 --tf32 True --gradient_checkpointing True --dataloader_num_workers 4 --lazy_preprocess True --n_query 256 --n_und_query 0 --report_to wandb --fix_dit False --fix_connect False --fix_llm True --action_dit_layer 3       ```
**eval_csgo_loc.py**
step=6000(5e-4 alpha_loc_loss: 10, masked_loc_loss: 1.229501, eval结果不收敛)
    ``    CUDA_VISIBLE_DEVICES=1 python eval_csgo_loc.py --csgo_config csgo_configs/test/exp8_1_loc.yaml      ``
step=6000(5e-4 alpha_loc_loss: 5, masked_loc_loss: 1.663208, eval结果稍好但仍不收敛)
    ``    CUDA_VISIBLE_DEVICES=1 python eval_csgo_loc.py --csgo_config csgo_configs/test/exp8_1_loc.yaml      ``
step=(1e-4 alpha_loc_loss: 2, masked_loc_loss:, eval结果) running~
    ``    CUDA_VISIBLE_DEVICES=1 python eval_csgo_loc.py --csgo_config csgo_configs/test/exp8_1_loc.yaml      ``



## exp9
**exp8 + autodl**
- 在autodl上多卡训练exp8
**train_csgo.py**
```     CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=29504 train_csgo.py --csgo_config csgo_configs/exp9.yaml --deepspeed deepspeed_scripts/zero0.json --model_name_or_path UniLIP-1B --unilip_factor 10.6 --mllm_hf_path OpenGVLab/InternVL3-1B-hf --version internvl --data_type "mix" --csgo_image_folder data/preprocessed_data --mm_use_im_start_end False --mm_use_im_patch_token False --bf16 True --output_dir outputs/csgo_1b/exp9 --num_train_epochs 100 --per_device_train_batch_size 112 --per_device_eval_batch_size 112 --gradient_accumulation_steps 1 --eval_strategy "no" --save_strategy "steps" --save_steps 2000 --save_total_limit 1 --learning_rate 1e-4 --weight_decay 0. --warmup_ratio 0.003 --lr_scheduler_type "cosine_with_min_lr" --model_max_length 1024 --logging_steps 1 --tf32 True --gradient_checkpointing True --dataloader_num_workers 8 --lazy_preprocess True --n_query 256 --n_und_query 0 --report_to wandb --fix_dit False --fix_connect False --fix_llm True --action_dit_layer 3       ```
**eval_csgo_loc.py** step=2000
```    CUDA_VISIBLE_DEVICES=1 python eval_csgo_loc.py --csgo_config csgo_configs/test/exp9_loc.yaml      ```
**eval_csgo.py** step=2000
``    CUDA_VISIBLE_DEVICES=1 python eval_csgo.py --csgo_config csgo_configs/test/exp9_gen.yaml      ``
    **benchmark_csgo.py**
    ``      CUDA_VISIBLE_DEVICES=1 python benchmark_csgo.py --gt data/preprocessed_data/de_dust2/imgs --pred outputs_eval/exp9_gen/test_20260205_234138/gen_imgs/de_dust2 --all --batch_size 8       ``
    ``      CUDA_VISIBLE_DEVICES=1 python benchmark_csgo.py --gt data/preprocessed_data/de_nuke/imgs --pred outputs_eval/exp9_gen/test_20260205_234138/gen_imgs/de_nuke --all --batch_size 8       ``
    ``      CUDA_VISIBLE_DEVICES=1 python benchmark_csgo.py --gt data/preprocessed_data/de_ancient/imgs --pred outputs_eval/exp9_gen/test_20260205_234138/gen_imgs/de_ancient --all --batch_size 8       ``
**continuous gen**  step=2000
``    CUDA_VISIBLE_DEVICES=1 python eval_csgo.py --csgo_config csgo_configs/test/exp9_gen_conti.yaml      ``
    **benchmark_csgo.py**
    ``      CUDA_VISIBLE_DEVICES=1 python benchmark_csgo.py --gt data/preprocessed_data/de_dust2/imgs --pred outputs_eval/exp9_gen_conti/test_20260205_234140/gen_imgs/de_dust2 --all --batch_size 8       ``
    ``      CUDA_VISIBLE_DEVICES=1 python benchmark_csgo.py --gt data/preprocessed_data/de_nuke/imgs --pred outputs_eval/exp9_gen_conti/test_20260205_234140/gen_imgs/de_nuke --all --batch_size 8       ``
    ``      CUDA_VISIBLE_DEVICES=1 python benchmark_csgo.py --gt data/preprocessed_data/de_ancient/imgs --pred outputs_eval/exp9_gen_conti/test_20260205_234140/gen_imgs/de_ancient --all --batch_size 8       ``
    **frames to video**
    ``    python frames_to_video.py --img_dir outputs_eval/exp9_gen_conti/test_20260205_234140/gen_imgs/de_dust2 --gt_dir data/preprocessed_data/de_dust2/imgs --output_dir outputs_eval/exp9_gen_conti/test_20260205_234140/gen_compared_videos/de_dust2 --max_duration 10        ``
    ``    python frames_to_video.py --img_dir outputs_eval/exp9_gen_conti/test_20260205_234140/gen_imgs/de_nuke --gt_dir data/preprocessed_data/de_nuke/imgs --output_dir outputs_eval/exp9_gen_conti/test_20260205_234140/gen_compared_videos/de_nuke --max_duration 10        ``
    ``    python frames_to_video.py --img_dir outputs_eval/exp9_gen_conti/test_20260205_234140/gen_imgs/de_ancient --gt_dir data/preprocessed_data/de_ancient/imgs --output_dir outputs_eval/exp9_gen_conti/test_20260205_234140/gen_compared_videos/de_ancient --max_duration 10        ``