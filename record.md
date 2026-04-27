# envs
1. benchmark_csgo.py

pip install torch torchvision torchmetrics[image] transformers pillow numpy tqdm scipy

git clone https://github.com/ragor114/PyTorch-Frechet-Video-Distance.git third_party/PyTorch-Frechet-Video-Distance
cp -r third_party/PyTorch-Frechet-Video-Distance/fvd_metric ./fvd_metric



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

### exp2_2
- exp2

- bs 128
- is_aciton_dit_vae_small_init: True
- logger + wandb => Trainer
- is_lora
- img_size=224
- use_pi05_action_dit=True
- is_loc_aux_loss: False
- is_multi_task_balanced: False； task_mix_ratio: 0.0； 这两个参数达到只训练gen的目的

```     CUDA_VISIBLE_DEVICES=1 MASTER_ADDR=127.0.0.1 MASTER_PORT=29503 RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 python train_csgo.py --csgo_config csgo_configs/exp2_2.yaml --deepspeed deepspeed_scripts/zero0.json --model_name_or_path UniLIP-1B --unilip_factor 10.6 --mllm_hf_path OpenGVLab/InternVL3-1B-hf --version internvl --data_type "mix" --csgo_image_folder data/preprocessed_data --mm_use_im_start_end False --mm_use_im_patch_token False --bf16 True --output_dir outputs/csgo_1b/exp2_2 --num_train_epochs 50 --per_device_train_batch_size 128 --per_device_eval_batch_size 128 --gradient_accumulation_steps 1 --eval_strategy "no" --save_strategy "steps" --save_steps 1000 --save_total_limit 2 --learning_rate 1e-4 --weight_decay 0. --warmup_ratio 0.003 --lr_scheduler_type "cosine_with_min_lr" --model_max_length 1024 --logging_steps 1 --tf32 True --gradient_checkpointing True --dataloader_num_workers 4 --lazy_preprocess True --n_query 256 --n_und_query 0 --report_to wandb --fix_dit False --fix_connect False --fix_llm True --lora_r 16      ```
**eval_csgo.py** 67: 3000 23400
``    CUDA_VISIBLE_DEVICES=0 python eval_csgo.py --csgo_config csgo_configs/test/exp2_2_gen.yaml      ``
**eval_csgo.py** 67: 3000 23400
``    CUDA_VISIBLE_DEVICES=0 python eval_csgo.py --csgo_config csgo_configs/test/exp2_2_gen.yaml      ``
    **benchmark_csgo.py**
    ``      CUDA_VISIBLE_DEVICES=0 python benchmark_csgo.py --gt data/preprocessed_data/de_dust2/imgs --pred outputs_eval/exp2_2_gen/test_20260322_230512/gen_imgs/de_dust2 --all --batch_size 8       ``
    ``      CUDA_VISIBLE_DEVICES=0 python benchmark_csgo.py --gt data/preprocessed_data/de_nuke/imgs --pred outputs_eval/exp2_2_gen/test_20260322_230512/gen_imgs/de_nuke --all --batch_size 8      ``
    ``      CUDA_VISIBLE_DEVICES=0 python benchmark_csgo.py --gt data/preprocessed_data/de_ancient/imgs --pred outputs_eval/exp2_2_gen/test_20260322_230512/gen_imgs/de_ancient --all --batch_size 8       ``
**continuous gen** 67: 3000 23400
``    CUDA_VISIBLE_DEVICES=0 python eval_csgo.py --csgo_config csgo_configs/test/exp2_2_gen_conti.yaml        ``
    **benchmark_csgo.py**
    ``      CUDA_VISIBLE_DEVICES=0 python benchmark_csgo.py --gt data/preprocessed_data/de_dust2/imgs --pred outputs_eval/exp2_2_gen_conti/test_20260322_230513/gen_imgs/de_dust2 --all --batch_size 8       ``
    ``      CUDA_VISIBLE_DEVICES=0 python benchmark_csgo.py --gt data/preprocessed_data/de_nuke/imgs --pred outputs_eval/exp2_2_gen_conti/test_20260322_230513/gen_imgs/de_nuke --all --batch_size 8      ``
    ``      CUDA_VISIBLE_DEVICES=0 python benchmark_csgo.py --gt data/preprocessed_data/de_ancient/imgs --pred outputs_eval/exp2_2_gen_conti/test_20260322_230513/gen_imgs/de_ancient --all --batch_size 8       ``

### exp2_3
- exp2
- bs 128
- is_aciton_dit_vae_small_init: True
- logger + wandb => Trainer
- is_lora
- use_pi05_action_dit=True
- is_loc_aux_loss: False
- is_multi_task_balanced: False； task_mix_ratio: 0.0； 这两个参数达到只训练gen的目的

- img_size=448
- epoch=100

```     CUDA_VISIBLE_DEVICES=0 MASTER_ADDR=127.0.0.1 MASTER_PORT=29503 RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 python train_csgo.py --csgo_config csgo_configs/exp2_3.yaml --deepspeed deepspeed_scripts/zero0.json --model_name_or_path UniLIP-1B --unilip_factor 10.6 --mllm_hf_path OpenGVLab/InternVL3-1B-hf --version internvl --data_type "mix" --csgo_image_folder data/preprocessed_data --mm_use_im_start_end False --mm_use_im_patch_token False --bf16 True --output_dir outputs/csgo_1b/exp2_3 --num_train_epochs 100 --per_device_train_batch_size 128 --per_device_eval_batch_size 128 --gradient_accumulation_steps 1 --eval_strategy "no" --save_strategy "steps" --save_steps 8000 --save_total_limit 3 --learning_rate 1e-4 --weight_decay 0. --warmup_ratio 0.003 --lr_scheduler_type "cosine_with_min_lr" --model_max_length 1024 --logging_steps 1 --tf32 True --gradient_checkpointing True --dataloader_num_workers 4 --lazy_preprocess True --n_query 256 --n_und_query 0 --report_to wandb --fix_dit False --fix_connect False --fix_llm True --lora_r 16      ```
**eval_csgo.py** 67: 46800
    ```        CUDA_VISIBLE_DEVICES=1 python eval_csgo.py --csgo_config csgo_configs/test/exp2_3_gen.yaml      ```
    **benchmark_csgo.py**
    ``      CUDA_VISIBLE_DEVICES=1 python benchmark_csgo.py --gt data/preprocessed_data/de_dust2/imgs --pred outputs_eval/exp2_3_gen/test_20260329_173633/gen_imgs/de_dust2 --all --batch_size 8       ``
    ``      CUDA_VISIBLE_DEVICES=1 python benchmark_csgo.py --gt data/preprocessed_data/de_nuke/imgs --pred outputs_eval/exp2_3_gen/test_20260329_173633/gen_imgs/de_nuke --all --batch_size 8       ``
    ``      CUDA_VISIBLE_DEVICES=1 python benchmark_csgo.py --gt data/preprocessed_data/de_ancient/imgs --pred outputs_eval/exp2_3_gen/test_20260329_173633/gen_imgs/de_ancient --all --batch_size 8       ``
**continuous gen** 67: 46800
    ```        CUDA_VISIBLE_DEVICES=1 python eval_csgo.py --csgo_config csgo_configs/test/exp2_3_gen_conti.yaml      ```
    **benchmark_csgo.py**
    ``      CUDA_VISIBLE_DEVICES=1 python benchmark_csgo.py --gt data/preprocessed_data/de_dust2/imgs --pred outputs_eval/exp2_3_gen_conti/test_20260329_173638/gen_imgs/de_dust2 --all --batch_size 8       ``
    ``      CUDA_VISIBLE_DEVICES=1 python benchmark_csgo.py --gt data/preprocessed_data/de_nuke/imgs --pred outputs_eval/exp2_3_gen_conti/test_20260329_173638/gen_imgs/de_nuke --all --batch_size 8       ``
    ``      CUDA_VISIBLE_DEVICES=1 python benchmark_csgo.py --gt data/preprocessed_data/de_ancient/imgs --pred outputs_eval/exp2_3_gen_conti/test_20260329_173638/gen_imgs/de_ancient --all --batch_size 8       ``
    **frames to video**
    ``    python frames_to_video.py --img_dir outputs_eval/exp2_3_gen_conti/test_20260329_173638/gen_imgs/de_dust2 --gt_dir data/preprocessed_data/de_dust2/imgs --output_dir outputs_eval/exp2_3_gen_conti/test_20260329_173638/gen_compared_videos/de_dust2 --max_duration 10        ``
    ``    python frames_to_video.py --img_dir outputs_eval/exp2_3_gen_conti/test_20260329_173638/gen_imgs/de_nuke --gt_dir data/preprocessed_data/de_nuke/imgs --output_dir outputs_eval/exp2_3_gen_conti/test_20260329_173638/gen_compared_videos/de_nuke --max_duration 10        ``
    ``    python frames_to_video.py --img_dir outputs_eval/exp2_3_gen_conti/test_20260329_173638/gen_imgs/de_ancient --gt_dir data/preprocessed_data/de_ancient/imgs --output_dir outputs_eval/exp2_3_gen_conti/test_20260329_173638/gen_compared_videos/de_ancient --max_duration 10        ``




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
**eval_csgo_loc.py** 67:8000 lr 1e-4masked_loc_loss: 0.013299 eval
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
**eval_csgo_loc.py** 67:46850 bs 64 lr 1e-4 masked_loc_loss:  eval
```    CUDA_VISIBLE_DEVICES=0 python eval_csgo_loc.py --csgo_config csgo_configs/test/exp4_6_2_loc.yaml      ```

### exp4_6_3
- exp4_5_1_lr1e-4_locditproj + exp7 \wo balancedataset
- 3maps
- lr 1e-4 #1e-3
- action_dit_projector_lr: 0.0005
- bs 32

- action_dit_layer 24

```     CUDA_VISIBLE_DEVICES=0 MASTER_ADDR=127.0.0.1 MASTER_PORT=29504 RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 python train_csgo.py --csgo_config csgo_configs/exp4_6_3.yaml --deepspeed deepspeed_scripts/zero0.json --model_name_or_path UniLIP-1B --unilip_factor 10.6 --mllm_hf_path OpenGVLab/InternVL3-1B-hf --version internvl --data_type "mix" --csgo_image_folder data/preprocessed_data --mm_use_im_start_end False --mm_use_im_patch_token False --bf16 True --output_dir outputs/csgo_1b/exp4_6_3 --num_train_epochs 50 --per_device_train_batch_size 32 --per_device_eval_batch_size 32 --gradient_accumulation_steps 1 --eval_strategy "no" --save_strategy "steps" --save_steps 2000 --save_total_limit 1 --learning_rate 1e-4 --weight_decay 0. --warmup_ratio 0.003 --lr_scheduler_type "cosine_with_min_lr" --model_max_length 1024 --logging_steps 1 --tf32 True --gradient_checkpointing True --dataloader_num_workers 4 --lazy_preprocess True --n_query 256 --n_und_query 0 --report_to wandb --fix_dit False --fix_connect False --fix_llm True --action_dit_layer 24       ```
**eval_csgo_loc.py** 67:22000
```    CUDA_VISIBLE_DEVICES=0 python eval_csgo_loc.py --csgo_config csgo_configs/test/exp4_6_3_loc.yaml      ```

## exp4_10
- exp4_5_2

- is_aciton_dit_vae_small_init: True

```     CUDA_VISIBLE_DEVICES=0 MASTER_ADDR=127.0.0.1 MASTER_PORT=29503 RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 python train_csgo.py --csgo_config csgo_configs/exp4_10.yaml --deepspeed deepspeed_scripts/zero0.json --model_name_or_path UniLIP-1B --unilip_factor 10.6 --mllm_hf_path OpenGVLab/InternVL3-1B-hf --version internvl --data_type "mix" --csgo_image_folder data/preprocessed_data --mm_use_im_start_end False --mm_use_im_patch_token False --bf16 True --output_dir outputs/csgo_1b/exp4_10 --num_train_epochs 20 --per_device_train_batch_size 64 --per_device_eval_batch_size 64 --gradient_accumulation_steps 1 --eval_strategy "no" --save_strategy "steps" --save_steps 2000 --save_total_limit 1 --learning_rate 1e-4 --weight_decay 0. --warmup_ratio 0.003 --lr_scheduler_type "cosine_with_min_lr" --model_max_length 1024 --logging_steps 1 --tf32 True --gradient_checkpointing True --dataloader_num_workers 4 --lazy_preprocess True --n_query 256 --n_und_query 0 --report_to wandb --fix_dit False --fix_connect False --fix_llm True --action_dit_layer 3       ```
**eval_csgo_loc.py** 67:18740
```    CUDA_VISIBLE_DEVICES=0 python eval_csgo_loc.py --csgo_config csgo_configs/test/exp4_10_loc.yaml      ```

## exp4_11
- exp4_6_3
- action_dit_layer 24

- bs 128
- is_aciton_dit_vae_small_init: True
- logger + wandb => Trainer
- is_lora
- img_size=224

```     CUDA_VISIBLE_DEVICES=0 MASTER_ADDR=127.0.0.1 MASTER_PORT=29504 RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 python train_csgo.py --csgo_config csgo_configs/exp4_11.yaml --deepspeed deepspeed_scripts/zero0.json --model_name_or_path UniLIP-1B --unilip_factor 10.6 --mllm_hf_path OpenGVLab/InternVL3-1B-hf --version internvl --data_type "mix" --csgo_image_folder data/preprocessed_data --mm_use_im_start_end False --mm_use_im_patch_token False --bf16 True --output_dir outputs/csgo_1b/exp4_11 --num_train_epochs 50 --per_device_train_batch_size 128 --per_device_eval_batch_size 128 --gradient_accumulation_steps 1 --eval_strategy "no" --save_strategy "steps" --save_steps 2000 --save_total_limit 1 --learning_rate 1e-4 --weight_decay 0. --warmup_ratio 0.003 --lr_scheduler_type "cosine_with_min_lr" --model_max_length 1024 --logging_steps 1 --tf32 True --gradient_checkpointing True --dataloader_num_workers 4 --lazy_preprocess True --n_query 256 --n_und_query 0 --report_to wandb --fix_dit True --fix_connect True --fix_llm True --action_dit_layer 24 --lora_r 16      ```
**eval_csgo_loc.py** 67:23400
```    CUDA_VISIBLE_DEVICES=0 python eval_csgo_loc.py --csgo_config csgo_configs/test/exp4_11_loc.yaml      ```

## exp4_12*

### exp4_12
- exp4_6_3
<!-- - action_dit_layer 24 -->
- bs 128
- is_aciton_dit_vae_small_init: True
- logger + wandb => Trainer
- is_lora
- img_size=224

- use_pi05_action_dit=True

```     CUDA_VISIBLE_DEVICES=0 MASTER_ADDR=127.0.0.1 MASTER_PORT=29502 RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 python train_csgo.py --csgo_config csgo_configs/exp4_12.yaml --deepspeed deepspeed_scripts/zero0.json --model_name_or_path UniLIP-1B --unilip_factor 10.6 --mllm_hf_path OpenGVLab/InternVL3-1B-hf --version internvl --data_type "mix" --csgo_image_folder data/preprocessed_data --mm_use_im_start_end False --mm_use_im_patch_token False --bf16 True --output_dir outputs/csgo_1b/exp4_12 --num_train_epochs 50 --per_device_train_batch_size 128 --per_device_eval_batch_size 128 --gradient_accumulation_steps 1 --eval_strategy "no" --save_strategy "steps" --save_steps 1000 --save_total_limit 3 --learning_rate 1e-4 --weight_decay 0. --warmup_ratio 0.003 --lr_scheduler_type "cosine_with_min_lr" --model_max_length 1024 --logging_steps 1 --tf32 True --gradient_checkpointing True --dataloader_num_workers 4 --lazy_preprocess True --n_query 256 --n_und_query 0 --report_to wandb --fix_dit False --fix_connect False --fix_llm True --lora_r 16      ```
**eval_csgo_loc.py** 65: step=6000 67: 8000 10000 12000 23450
```    CUDA_VISIBLE_DEVICES=0 python eval_csgo_loc.py --csgo_config csgo_configs/test/exp4_12_loc.yaml      ```

### exp4_12_1
- exp4_6_3
<!-- - action_dit_layer 24 -->
- bs 128
- is_aciton_dit_vae_small_init: True
- logger + wandb => Trainer
- is_lora
- img_size=224

- use_pi05_action_dit=True
- fix_vit False
- fix_llm False

```     CUDA_VISIBLE_DEVICES=0 MASTER_ADDR=127.0.0.1 MASTER_PORT=29504 RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 python train_csgo.py --csgo_config csgo_configs/exp4_12_1.yaml --deepspeed deepspeed_scripts/zero0.json --model_name_or_path UniLIP-1B --unilip_factor 10.6 --mllm_hf_path OpenGVLab/InternVL3-1B-hf --version internvl --data_type "mix" --csgo_image_folder data/preprocessed_data --mm_use_im_start_end False --mm_use_im_patch_token False --bf16 True --output_dir outputs/csgo_1b/exp4_12_1 --num_train_epochs 50 --per_device_train_batch_size 64 --per_device_eval_batch_size 64 --gradient_accumulation_steps 2 --eval_strategy "no" --save_strategy "steps" --save_steps 1000 --save_total_limit 3 --learning_rate 1e-4 --weight_decay 0. --warmup_ratio 0.003 --lr_scheduler_type "cosine_with_min_lr" --model_max_length 1024 --logging_steps 1 --tf32 True --gradient_checkpointing True --dataloader_num_workers 4 --lazy_preprocess True --n_query 256 --n_und_query 0 --report_to wandb --fix_dit True --fix_connect True --fix_llm False --fix_vit False --lora_r 16      ```
**eval_csgo_loc.py** 67: step=2000 3000 4000 21000 23450 修复eval_loc加载权重bug:23450
```    CUDA_VISIBLE_DEVICES=0 python eval_csgo_loc.py --csgo_config csgo_configs/test/exp4_12_1_loc.yaml      ```

### exp4_12_6
- exp4_6_3
<!-- - action_dit_layer 24 -->
- bs 128
- is_aciton_dit_vae_small_init: True
- logger + wandb => Trainer
- use_pi05_action_dit=True

- img_size=224
- fix_vit True
- fix_llm True
- is_lora False

```     CUDA_VISIBLE_DEVICES=0 MASTER_ADDR=127.0.0.1 MASTER_PORT=29503 RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 python train_csgo.py --csgo_config csgo_configs/exp4_12_6.yaml --deepspeed deepspeed_scripts/zero0.json --model_name_or_path UniLIP-1B --unilip_factor 10.6 --mllm_hf_path OpenGVLab/InternVL3-1B-hf --version internvl --data_type "mix" --csgo_image_folder data/preprocessed_data --mm_use_im_start_end False --mm_use_im_patch_token False --bf16 True --output_dir outputs/csgo_1b/exp4_12_6 --num_train_epochs 50 --per_device_train_batch_size 64 --per_device_eval_batch_size 64 --gradient_accumulation_steps 2 --eval_strategy "no" --save_strategy "steps" --save_steps 1000 --save_total_limit 3 --learning_rate 1e-4 --weight_decay 0. --warmup_ratio 0.003 --lr_scheduler_type "cosine_with_min_lr" --model_max_length 1024 --logging_steps 1 --tf32 True --gradient_checkpointing True --dataloader_num_workers 4 --lazy_preprocess True --n_query 256 --n_und_query 0 --report_to wandb --fix_dit True --fix_connect True --fix_llm True --fix_vit True      ```
**eval_csgo_loc.py** 67: 23450
```    CUDA_VISIBLE_DEVICES=0 python eval_csgo_loc.py --csgo_config csgo_configs/test/exp4_12_6_loc.yaml      ```

### exp4_12_7
- exp4_6_3
<!-- - action_dit_layer 24 -->
- bs 128
- is_aciton_dit_vae_small_init: True
- logger + wandb => Trainer
- use_pi05_action_dit=True

- img_size=224
- fix_vit False
- fix_llm False
- is_lora False

```     CUDA_VISIBLE_DEVICES=2 MASTER_ADDR=127.0.0.1 MASTER_PORT=29502 RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 python train_csgo.py --csgo_config csgo_configs/exp4_12_7.yaml --deepspeed deepspeed_scripts/zero0.json --model_name_or_path UniLIP-1B --unilip_factor 10.6 --mllm_hf_path OpenGVLab/InternVL3-1B-hf --version internvl --data_type "mix" --csgo_image_folder data/preprocessed_data --mm_use_im_start_end False --mm_use_im_patch_token False --bf16 True --output_dir outputs/csgo_1b/exp4_12_7 --num_train_epochs 50 --per_device_train_batch_size 128 --per_device_eval_batch_size 128 --gradient_accumulation_steps 1 --eval_strategy "no" --save_strategy "steps" --save_steps 8000 --save_total_limit 3 --learning_rate 1e-4 --weight_decay 0. --warmup_ratio 0.003 --lr_scheduler_type "cosine_with_min_lr" --model_max_length 1024 --logging_steps 1 --tf32 True --gradient_checkpointing True --dataloader_num_workers 4 --lazy_preprocess True --n_query 256 --n_und_query 0 --report_to wandb --fix_dit True --fix_connect True --fix_llm False --fix_vit False      ```
**eval_csgo_loc.py** 67:23450 修复eval_loc加载权重bug:23450 修复eval_loc加载权重bug，fst未cp RMSAdaNorm，需要在本地重新训练：8000 23400
```    CUDA_VISIBLE_DEVICES=0 python eval_csgo_loc.py --csgo_config csgo_configs/test/exp4_12_7_loc.yaml      ```

### exp4_12_2
- exp4_6_3
<!-- - action_dit_layer 24 -->
- bs 128
- is_aciton_dit_vae_small_init: True
- logger + wandb => Trainer
- use_pi05_action_dit=True

- img_size=448
- is_lora: False

```     CUDA_VISIBLE_DEVICES=0 MASTER_ADDR=127.0.0.1 MASTER_PORT=29506 RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 python train_csgo.py --csgo_config csgo_configs/exp4_12_2.yaml --deepspeed deepspeed_scripts/zero0.json --model_name_or_path UniLIP-1B --unilip_factor 10.6 --mllm_hf_path OpenGVLab/InternVL3-1B-hf --version internvl --data_type "mix" --csgo_image_folder data/preprocessed_data --mm_use_im_start_end False --mm_use_im_patch_token False --bf16 True --output_dir outputs/csgo_1b/exp4_12_2 --num_train_epochs 50 --per_device_train_batch_size 128 --per_device_eval_batch_size 128 --gradient_accumulation_steps 1 --eval_strategy "no" --save_strategy "steps" --save_steps 8000 --save_total_limit 3 --learning_rate 1e-4 --weight_decay 0. --warmup_ratio 0.003 --lr_scheduler_type "cosine_with_min_lr" --model_max_length 1024 --logging_steps 1 --tf32 True --gradient_checkpointing True --dataloader_num_workers 4 --lazy_preprocess True --n_query 256 --n_und_query 0 --report_to wandb --fix_dit True --fix_connect True --fix_llm True --fix_vit True     ```
**eval_csgo_loc.py** 67: 23450 再推理一次23450
```    CUDA_VISIBLE_DEVICES=1 python eval_csgo_loc.py --csgo_config csgo_configs/test/exp4_12_2_loc.yaml      ```

### exp4_12_4
- exp4_6_3
<!-- - action_dit_layer 24 -->
- bs 128
- is_aciton_dit_vae_small_init: True
- logger + wandb => Trainer
- use_pi05_action_dit=True

- img_size=448
- fix_llm: False
- fix_vit: False
- is_lora: False

```     CUDA_VISIBLE_DEVICES=1 MASTER_ADDR=127.0.0.1 MASTER_PORT=29507 RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 python train_csgo.py --csgo_config csgo_configs/exp4_12_4.yaml --deepspeed deepspeed_scripts/zero0.json --model_name_or_path UniLIP-1B --unilip_factor 10.6 --mllm_hf_path OpenGVLab/InternVL3-1B-hf --version internvl --data_type "mix" --csgo_image_folder data/preprocessed_data --mm_use_im_start_end False --mm_use_im_patch_token False --bf16 True --output_dir outputs/csgo_1b/exp4_12_4 --num_train_epochs 50 --per_device_train_batch_size 128 --per_device_eval_batch_size 128 --gradient_accumulation_steps 1 --eval_strategy "no" --save_strategy "steps" --save_steps 8000 --save_total_limit 3 --learning_rate 1e-4 --weight_decay 0. --warmup_ratio 0.003 --lr_scheduler_type "cosine_with_min_lr" --model_max_length 1024 --logging_steps 1 --tf32 True --gradient_checkpointing True --dataloader_num_workers 4 --lazy_preprocess True --n_query 256 --n_und_query 0 --report_to wandb --fix_dit True --fix_connect True --fix_llm False --fix_vit False     ```
**eval_csgo_loc.py** fst h800: 23400 修复eval_loc加载权重bug，fst未cp RMSAdaNorm，需要在本地重新训练：8000 23400
```    CUDA_VISIBLE_DEVICES=0 python eval_csgo_loc.py --csgo_config csgo_configs/test/exp4_12_4_loc.yaml      ```

### exp4_12_5
- exp4_6_3
<!-- - action_dit_layer 24 -->
- bs 128
- is_aciton_dit_vae_small_init: True
- logger + wandb => Trainer
- use_pi05_action_dit=True

- img_size=448
- fix_llm: True
- fix_vit: True
- is_lora: True

```     CUDA_VISIBLE_DEVICES=0 MASTER_ADDR=127.0.0.1 MASTER_PORT=29508 RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 python train_csgo.py --csgo_config csgo_configs/exp4_12_5.yaml --deepspeed deepspeed_scripts/zero0.json --model_name_or_path UniLIP-1B --unilip_factor 10.6 --mllm_hf_path OpenGVLab/InternVL3-1B-hf --version internvl --data_type "mix" --csgo_image_folder data/preprocessed_data --mm_use_im_start_end False --mm_use_im_patch_token False --bf16 True --output_dir outputs/csgo_1b/exp4_12_5 --num_train_epochs 50 --per_device_train_batch_size 128 --per_device_eval_batch_size 128 --gradient_accumulation_steps 1 --eval_strategy "no" --save_strategy "steps" --save_steps 8000 --save_total_limit 3 --learning_rate 1e-4 --weight_decay 0. --warmup_ratio 0.003 --lr_scheduler_type "cosine_with_min_lr" --model_max_length 1024 --logging_steps 1 --tf32 True --gradient_checkpointing True --dataloader_num_workers 4 --lazy_preprocess True --n_query 256 --n_und_query 0 --report_to wandb --fix_dit True --fix_connect True --fix_llm True --fix_vit True --lora_r 16     ```
**eval_csgo_loc.py** fst a100: 16000 23400 修复eval_loc加载权重bug，fst未cp RMSAdaNorm，需要在本地重新训练：16000 23400
```    CUDA_VISIBLE_DEVICES=0 python eval_csgo_loc.py --csgo_config csgo_configs/test/exp4_12_5_loc.yaml      ```

### exp4_12_3
- exp4_6_3
<!-- - action_dit_layer 24 -->
- bs 128
- is_aciton_dit_vae_small_init: True
- logger + wandb => Trainer
- use_pi05_action_dit=True

- img_size=448
- fix_llm: False
- fix_vit: False
- is_lora: True

```     CUDA_VISIBLE_DEVICES=0 MASTER_ADDR=127.0.0.1 MASTER_PORT=29507 RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 python train_csgo.py --csgo_config csgo_configs/exp4_12_3.yaml --deepspeed deepspeed_scripts/zero0.json --model_name_or_path UniLIP-1B --unilip_factor 10.6 --mllm_hf_path OpenGVLab/InternVL3-1B-hf --version internvl --data_type "mix" --csgo_image_folder data/preprocessed_data --mm_use_im_start_end False --mm_use_im_patch_token False --bf16 True --output_dir outputs/csgo_1b/exp4_12_3 --num_train_epochs 50 --per_device_train_batch_size 64 --per_device_eval_batch_size 64 --gradient_accumulation_steps 2 --eval_strategy "no" --save_strategy "steps" --save_steps 6000 --save_total_limit 4 --learning_rate 1e-4 --weight_decay 0. --warmup_ratio 0.003 --lr_scheduler_type "cosine_with_min_lr" --model_max_length 1024 --logging_steps 1 --tf32 True --gradient_checkpointing True --dataloader_num_workers 4 --lazy_preprocess True --n_query 256 --n_und_query 0 --report_to wandb --fix_dit True --fix_connect True --fix_llm False --fix_vit False --lora_r 16     ```
**eval_csgo_loc.py** 67: 23450 修复eval_loc加载权重bug： 23450
```    CUDA_VISIBLE_DEVICES=0 python eval_csgo_loc.py --csgo_config csgo_configs/test/exp4_12_3_loc.yaml      ```



## exp4_13*

### exp4_13
- exp4_6_3
<!-- - action_dit_layer 24 -->
- bs 128
- is_aciton_dit_vae_small_init: True
- logger + wandb => Trainer
- img_size=224

- use_pi05_action_dit=False
- action_dit_projector_lr: 1e-3 #vit_regression_head和action_dit_projector共用同一个lr
- use_vit_regression_head:  True
- is_lora: False # 只训练head，所以无lora


```     CUDA_VISIBLE_DEVICES=0 MASTER_ADDR=127.0.0.1 MASTER_PORT=29504 RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 python train_csgo.py --csgo_config csgo_configs/exp4_13.yaml --deepspeed deepspeed_scripts/zero0.json --model_name_or_path UniLIP-1B --unilip_factor 10.6 --mllm_hf_path OpenGVLab/InternVL3-1B-hf --version internvl --data_type "mix" --csgo_image_folder data/preprocessed_data --mm_use_im_start_end False --mm_use_im_patch_token False --bf16 True --output_dir outputs/csgo_1b/exp4_13 --num_train_epochs 50 --per_device_train_batch_size 64 --per_device_eval_batch_size 64 --gradient_accumulation_steps 2 --eval_strategy "no" --save_strategy "steps" --save_steps 6000 --save_total_limit 4 --learning_rate 1e-4 --weight_decay 0. --warmup_ratio 0.003 --lr_scheduler_type "cosine_with_min_lr" --model_max_length 1024 --logging_steps 1 --tf32 True --gradient_checkpointing True --dataloader_num_workers 4 --lazy_preprocess True --n_query 256 --n_und_query 0 --report_to wandb --fix_dit True --fix_connect True --fix_llm True --fix_vit True      ```
**eval_csgo_loc.py**67: step=2000 3000 4000 6000 23450 lr=1e-3: 6000 12000 18000 23450
```    CUDA_VISIBLE_DEVICES=0 python eval_csgo_loc.py --csgo_config csgo_configs/test/exp4_13_loc.yaml      ```

### exp4_13_1
- exp4_6_3
<!-- - action_dit_layer 24 -->
- bs 128
- is_aciton_dit_vae_small_init: True
- logger + wandb => Trainer
- img_size=224

- use_pi05_action_dit=False
- action_dit_projector_lr: 1e-3
- use_vit_regression_head:  True
- fix_vit: False
- is_lora: True # vit使用lora，head直接训

```     CUDA_VISIBLE_DEVICES=0 MASTER_ADDR=127.0.0.1 MASTER_PORT=29505 RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 python train_csgo.py --csgo_config csgo_configs/exp4_13_1.yaml --deepspeed deepspeed_scripts/zero0.json --model_name_or_path UniLIP-1B --unilip_factor 10.6 --mllm_hf_path OpenGVLab/InternVL3-1B-hf --version internvl --data_type "mix" --csgo_image_folder data/preprocessed_data --mm_use_im_start_end False --mm_use_im_patch_token False --bf16 True --output_dir outputs/csgo_1b/exp4_13_1 --num_train_epochs 50 --per_device_train_batch_size 64 --per_device_eval_batch_size 64 --gradient_accumulation_steps 2 --eval_strategy "no" --save_strategy "steps" --save_steps 6000 --save_total_limit 4 --learning_rate 1e-4 --weight_decay 0. --warmup_ratio 0.003 --lr_scheduler_type "cosine_with_min_lr" --model_max_length 1024 --logging_steps 1 --tf32 True --gradient_checkpointing True --dataloader_num_workers 4 --lazy_preprocess True --n_query 256 --n_und_query 0 --report_to wandb --fix_dit True --fix_connect True --fix_llm True --fix_vit False --lora_r 16      ```
**eval_csgo_loc.py**67: step=2000 3000 4000 6000 23450 lr=1e-3: 23450
```    CUDA_VISIBLE_DEVICES=0 python eval_csgo_loc.py --csgo_config csgo_configs/test/exp4_13_1_loc.yaml      ```

### exp4_13_1_1
- exp4_6_3
<!-- - action_dit_layer 24 -->
- bs 128
- is_aciton_dit_vae_small_init: True
- logger + wandb => Trainer
- img_size=224
- use_pi05_action_dit=False
- action_dit_projector_lr: 1e-3
- use_vit_regression_head:  True
- fix_vit: False

- is_lora: False

```     CUDA_VISIBLE_DEVICES=0 MASTER_ADDR=127.0.0.1 MASTER_PORT=29508 RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 python train_csgo.py --csgo_config csgo_configs/exp4_13_1_1.yaml --deepspeed deepspeed_scripts/zero0.json --model_name_or_path UniLIP-1B --unilip_factor 10.6 --mllm_hf_path OpenGVLab/InternVL3-1B-hf --version internvl --data_type "mix" --csgo_image_folder data/preprocessed_data --mm_use_im_start_end False --mm_use_im_patch_token False --bf16 True --output_dir outputs/csgo_1b/exp4_13_1_1 --num_train_epochs 50 --per_device_train_batch_size 64 --per_device_eval_batch_size 64 --gradient_accumulation_steps 2 --eval_strategy "no" --save_strategy "steps" --save_steps 6000 --save_total_limit 4 --learning_rate 1e-4 --weight_decay 0. --warmup_ratio 0.003 --lr_scheduler_type "cosine_with_min_lr" --model_max_length 1024 --logging_steps 1 --tf32 True --gradient_checkpointing True --dataloader_num_workers 4 --lazy_preprocess True --n_query 256 --n_und_query 0 --report_to wandb --fix_dit True --fix_connect True --fix_llm True --fix_vit False      ```
**eval_csgo_loc.py**67: step=23450
```    CUDA_VISIBLE_DEVICES=0 python eval_csgo_loc.py --csgo_config csgo_configs/test/exp4_13_1_1_loc.yaml      ```

### exp4_13_2
- exp4_6_3
<!-- - action_dit_layer 24 -->
- bs 128
- is_aciton_dit_vae_small_init: True
- logger + wandb => Trainer
- img_size=224
- use_pi05_action_dit=False
- action_dit_projector_lr: 5e-4 #vit_regression_head和action_dit_projector共用同一个lr
- is_lora: False # 只训练head，所以无lora

- use_vit_regression_head: False #两种head只能二选一
- use_vit_cls_regression_head: True

```     CUDA_VISIBLE_DEVICES=0 MASTER_ADDR=127.0.0.1 MASTER_PORT=29502 RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 python train_csgo.py --csgo_config csgo_configs/exp4_13_2.yaml --deepspeed deepspeed_scripts/zero0.json --model_name_or_path UniLIP-1B --unilip_factor 10.6 --mllm_hf_path OpenGVLab/InternVL3-1B-hf --version internvl --data_type "mix" --csgo_image_folder data/preprocessed_data --mm_use_im_start_end False --mm_use_im_patch_token False --bf16 True --output_dir outputs/csgo_1b/exp4_13_2 --num_train_epochs 50 --per_device_train_batch_size 64 --per_device_eval_batch_size 64 --gradient_accumulation_steps 2 --eval_strategy "no" --save_strategy "steps" --save_steps 1000 --save_total_limit 3 --learning_rate 1e-4 --weight_decay 0. --warmup_ratio 0.003 --lr_scheduler_type "cosine_with_min_lr" --model_max_length 1024 --logging_steps 1 --tf32 True --gradient_checkpointing True --dataloader_num_workers 4 --lazy_preprocess True --n_query 256 --n_und_query 0 --report_to wandb --fix_dit True --fix_connect True --fix_llm True --fix_vit True      ```
**eval_csgo_loc.py**67: step=23450
```    CUDA_VISIBLE_DEVICES=0 python eval_csgo_loc.py --csgo_config csgo_configs/test/exp4_13_2_loc.yaml      ```

### exp4_13_3
- exp4_6_3
<!-- - action_dit_layer 24 -->
- bs 128
- is_aciton_dit_vae_small_init: True
- logger + wandb => Trainer
- img_size=224
- use_pi05_action_dit=False
- action_dit_projector_lr: 5e-4
- fix_vit: False
- is_lora: True # vit使用lora，head直接训

- use_vit_regression_head: False #两种head只能二选一
- use_vit_cls_regression_head: True

```     CUDA_VISIBLE_DEVICES=0 MASTER_ADDR=127.0.0.1 MASTER_PORT=29503 RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 python train_csgo.py --csgo_config csgo_configs/exp4_13_3.yaml --deepspeed deepspeed_scripts/zero0.json --model_name_or_path UniLIP-1B --unilip_factor 10.6 --mllm_hf_path OpenGVLab/InternVL3-1B-hf --version internvl --data_type "mix" --csgo_image_folder data/preprocessed_data --mm_use_im_start_end False --mm_use_im_patch_token False --bf16 True --output_dir outputs/csgo_1b/exp4_13_3 --num_train_epochs 50 --per_device_train_batch_size 64 --per_device_eval_batch_size 64 --gradient_accumulation_steps 2 --eval_strategy "no" --save_strategy "steps" --save_steps 1000 --save_total_limit 3 --learning_rate 1e-4 --weight_decay 0. --warmup_ratio 0.003 --lr_scheduler_type "cosine_with_min_lr" --model_max_length 1024 --logging_steps 1 --tf32 True --gradient_checkpointing True --dataloader_num_workers 4 --lazy_preprocess True --n_query 256 --n_und_query 0 --report_to wandb --fix_dit True --fix_connect True --fix_llm True --fix_vit False --lora_r 16      ```
**eval_csgo_loc.py**67: 23450
```    CUDA_VISIBLE_DEVICES=0 python eval_csgo_loc.py --csgo_config csgo_configs/test/exp4_13_3_loc.yaml      ```

### exp4_13_3_1
- exp4_6_3
<!-- - action_dit_layer 24 -->
- bs 128
- is_aciton_dit_vae_small_init: True
- logger + wandb => Trainer
- img_size=224
- use_pi05_action_dit=False
- action_dit_projector_lr: 5e-4
- fix_vit: False
- use_vit_regression_head: False #两种head只能二选一
- use_vit_cls_regression_head: True

- is_lora: False

```     CUDA_VISIBLE_DEVICES=0 MASTER_ADDR=127.0.0.1 MASTER_PORT=29509 RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 python train_csgo.py --csgo_config csgo_configs/exp4_13_3_1.yaml --deepspeed deepspeed_scripts/zero0.json --model_name_or_path UniLIP-1B --unilip_factor 10.6 --mllm_hf_path OpenGVLab/InternVL3-1B-hf --version internvl --data_type "mix" --csgo_image_folder data/preprocessed_data --mm_use_im_start_end False --mm_use_im_patch_token False --bf16 True --output_dir outputs/csgo_1b/exp4_13_3_1 --num_train_epochs 50 --per_device_train_batch_size 64 --per_device_eval_batch_size 64 --gradient_accumulation_steps 2 --eval_strategy "no" --save_strategy "steps" --save_steps 6000 --save_total_limit 4 --learning_rate 1e-4 --weight_decay 0. --warmup_ratio 0.003 --lr_scheduler_type "cosine_with_min_lr" --model_max_length 1024 --logging_steps 1 --tf32 True --gradient_checkpointing True --dataloader_num_workers 4 --lazy_preprocess True --n_query 256 --n_und_query 0 --report_to wandb --fix_dit True --fix_connect True --fix_llm True --fix_vit False      ```
**eval_csgo_loc.py**67:
```    CUDA_VISIBLE_DEVICES=0 python eval_csgo_loc.py --csgo_config csgo_configs/test/exp4_13_3_1_loc.yaml      ```


### exp4_13_4
- exp4_6_3
<!-- - action_dit_layer 24 -->
- bs 128
- is_aciton_dit_vae_small_init: True
- logger + wandb => Trainer
- action_dit_projector_lr: 5e-4

- use_pi05_action_dit=False
- use_vit_regression_head: False
- use_vit_cls_regression_head: False
- use_codex_vit_regression_head: True
- loc_use_circular_loss: True (loc_xy_loss_weight: 1.0 loc_z_loss_weight: 1.0 loc_angle_loss_weight: 2.0)
- img_size=224
- fix_vit: False
- is_lora: True

```     CUDA_VISIBLE_DEVICES=0 MASTER_ADDR=127.0.0.1 MASTER_PORT=29506 RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 python train_csgo.py --csgo_config csgo_configs/exp4_13_4.yaml --deepspeed deepspeed_scripts/zero0.json --model_name_or_path UniLIP-1B --unilip_factor 10.6 --mllm_hf_path OpenGVLab/InternVL3-1B-hf --version internvl --data_type "mix" --csgo_image_folder data/preprocessed_data --mm_use_im_start_end False --mm_use_im_patch_token False --bf16 True --output_dir outputs/csgo_1b/exp4_13_4 --num_train_epochs 50 --per_device_train_batch_size 64 --per_device_eval_batch_size 64 --gradient_accumulation_steps 2 --eval_strategy "no" --save_strategy "steps" --save_steps 8000 --save_total_limit 3 --learning_rate 1e-4 --weight_decay 0. --warmup_ratio 0.003 --lr_scheduler_type "cosine_with_min_lr" --model_max_length 1024 --logging_steps 1 --tf32 True --gradient_checkpointing True --dataloader_num_workers 4 --lazy_preprocess True --n_query 256 --n_und_query 0 --report_to wandb --fix_dit True --fix_connect True --fix_llm True --fix_vit False --lora_r 16      ```
**eval_csgo_loc.py** 65: 8000 67: 23450 修复eval_loc加载权重bug： 23450
```    CUDA_VISIBLE_DEVICES=0 python eval_csgo_loc.py --csgo_config csgo_configs/test/exp4_13_4_loc.yaml      ```

#### exp4_13_4_1
- exp4_6_3
<!-- - action_dit_layer 24 -->
- bs 128
- is_aciton_dit_vae_small_init: True
- logger + wandb => Trainer


- use_pi05_action_dit=False
- use_vit_regression_head: False
- use_vit_cls_regression_head: False
- use_codex_vit_regression_head: True
- loc_use_circular_loss: True (loc_xy_loss_weight: 1.0 loc_z_loss_weight: 1.0 loc_angle_loss_weight: 2.0)
- img_size=224
- fix_vit: False
- is_lora: True

- action_dit_projector_lr: 1e-4
- lr=5e-5
- alpha_loc=1
- action_dit_projector_lr: 1e-4
- lr=5e-5
- alpha_loc=1

```     CUDA_VISIBLE_DEVICES=1 MASTER_ADDR=127.0.0.1 MASTER_PORT=29506 RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 python train_csgo.py --csgo_config csgo_configs/exp4_13_4_1.yaml --deepspeed deepspeed_scripts/zero0.json --model_name_or_path UniLIP-1B --unilip_factor 10.6 --mllm_hf_path OpenGVLab/InternVL3-1B-hf --version internvl --data_type "mix" --csgo_image_folder data/preprocessed_data --mm_use_im_start_end False --mm_use_im_patch_token False --bf16 True --output_dir outputs/csgo_1b/exp4_13_4_1 --num_train_epochs 50 --per_device_train_batch_size 128 --per_device_eval_batch_size 128 --gradient_accumulation_steps 1 --eval_strategy "no" --save_strategy "steps" --save_steps 8000 --save_total_limit 3 --learning_rate 5e-5 --weight_decay 0. --warmup_ratio 0.003 --lr_scheduler_type "cosine_with_min_lr" --model_max_length 1024 --logging_steps 1 --tf32 True --gradient_checkpointing True --dataloader_num_workers 4 --lazy_preprocess True --n_query 256 --n_und_query 0 --report_to wandb --fix_dit True --fix_connect True --fix_llm True --fix_vit False --lora_r 16      ```
**eval_csgo_loc.py**65: 23450 调整超参: 23400
```    CUDA_VISIBLE_DEVICES=1 python eval_csgo_loc.py --csgo_config csgo_configs/test/exp4_13_4_1_loc.yaml      ```

### exp4_13_5
- exp4_6_3
<!-- - action_dit_layer 24 -->
- bs 128
- is_aciton_dit_vae_small_init: True
- logger + wandb => Trainer
- action_dit_projector_lr: 5e-4

- use_pi05_action_dit=False
- use_vit_regression_head: False
- use_vit_cls_regression_head: False
- use_codex_vit_regression_head: True
- loc_use_circular_loss: True (loc_xy_loss_weight: 1.0 loc_z_loss_weight: 1.0 loc_angle_loss_weight: 2.0)
- img_size=224
- fix_vit: False
- is_lora: False

```     CUDA_VISIBLE_DEVICES=5 MASTER_ADDR=127.0.0.1 MASTER_PORT=29508 RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 python train_csgo.py --csgo_config csgo_configs/exp4_13_5.yaml --deepspeed deepspeed_scripts/zero0.json --model_name_or_path UniLIP-1B --unilip_factor 10.6 --mllm_hf_path OpenGVLab/InternVL3-1B-hf --version internvl --data_type "mix" --csgo_image_folder data/preprocessed_data --mm_use_im_start_end False --mm_use_im_patch_token False --bf16 True --output_dir outputs/csgo_1b/exp4_13_5 --num_train_epochs 50 --per_device_train_batch_size 128 --per_device_eval_batch_size 128 --gradient_accumulation_steps 1 --eval_strategy "no" --save_strategy "steps" --save_steps 8000 --save_total_limit 3 --learning_rate 1e-4 --weight_decay 0. --warmup_ratio 0.003 --lr_scheduler_type "cosine_with_min_lr" --model_max_length 1024 --logging_steps 1 --tf32 True --gradient_checkpointing True --dataloader_num_workers 4 --lazy_preprocess True --n_query 256 --n_und_query 0 --report_to wandb --fix_dit True --fix_connect True --fix_llm True --fix_vit False      ```
**eval_csgo_loc.py** fst h800 23400 修复eval_loc加载权重bug: 23400
```    CUDA_VISIBLE_DEVICES=0 python eval_csgo_loc.py --csgo_config csgo_configs/test/exp4_13_5_loc.yaml      ```

#### exp4_13_5_1
- exp4_6_3
<!-- - action_dit_layer 24 -->
- bs 128
- is_aciton_dit_vae_small_init: True
- logger + wandb => Trainer

- use_pi05_action_dit=False
- use_vit_regression_head: False
- use_vit_cls_regression_head: False
- use_codex_vit_regression_head: True
- loc_use_circular_loss: True (loc_xy_loss_weight: 1.0 loc_z_loss_weight: 1.0 loc_angle_loss_weight: 2.0)
- img_size=224
- fix_vit: False
- is_lora: False

- action_dit_projector_lr: 1e-4
- lr=5e-5
- alpha_loc=1

```     CUDA_VISIBLE_DEVICES=1 MASTER_ADDR=127.0.0.1 MASTER_PORT=29508 RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 python train_csgo.py --csgo_config csgo_configs/exp4_13_5_1.yaml --deepspeed deepspeed_scripts/zero0.json --model_name_or_path UniLIP-1B --unilip_factor 10.6 --mllm_hf_path OpenGVLab/InternVL3-1B-hf --version internvl --data_type "mix" --csgo_image_folder data/preprocessed_data --mm_use_im_start_end False --mm_use_im_patch_token False --bf16 True --output_dir outputs/csgo_1b/exp4_13_5_1 --num_train_epochs 50 --per_device_train_batch_size 128 --per_device_eval_batch_size 128 --gradient_accumulation_steps 1 --eval_strategy "no" --save_strategy "steps" --save_steps 8000 --save_total_limit 3 --learning_rate 5e-5 --weight_decay 0. --warmup_ratio 0.003 --lr_scheduler_type "cosine_with_min_lr" --model_max_length 1024 --logging_steps 1 --tf32 True --gradient_checkpointing True --dataloader_num_workers 4 --lazy_preprocess True --n_query 256 --n_und_query 0 --report_to wandb --fix_dit True --fix_connect True --fix_llm True --fix_vit False      ```
**eval_csgo_loc.py**65: 23450 调整超参： 23400
```    CUDA_VISIBLE_DEVICES=1 python eval_csgo_loc.py --csgo_config csgo_configs/test/exp4_13_5_1_loc.yaml      ```

### exp4_13_6
- exp4_6_3
<!-- - action_dit_layer 24 -->
- bs 128
- is_aciton_dit_vae_small_init: True
- logger + wandb => Trainer
- action_dit_projector_lr: 5e-4

- use_pi05_action_dit=False
- use_vit_regression_head: False
- use_vit_cls_regression_head: False
- use_codex_vit_regression_head: True
- loc_use_circular_loss: True (loc_xy_loss_weight: 1.0 loc_z_loss_weight: 1.0 loc_angle_loss_weight: 2.0)
- img_size=448
- fix_vit: False
- is_lora: True

```     CUDA_VISIBLE_DEVICES=0 MASTER_ADDR=127.0.0.1 MASTER_PORT=29509 RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 python train_csgo.py --csgo_config csgo_configs/exp4_13_6.yaml --deepspeed deepspeed_scripts/zero0.json --model_name_or_path UniLIP-1B --unilip_factor 10.6 --mllm_hf_path OpenGVLab/InternVL3-1B-hf --version internvl --data_type "mix" --csgo_image_folder data/preprocessed_data --mm_use_im_start_end False --mm_use_im_patch_token False --bf16 True --output_dir outputs/csgo_1b/exp4_13_6 --num_train_epochs 50 --per_device_train_batch_size 64 --per_device_eval_batch_size 64 --gradient_accumulation_steps 2 --eval_strategy "no" --save_strategy "steps" --save_steps 8000 --save_total_limit 3 --learning_rate 1e-4 --weight_decay 0. --warmup_ratio 0.003 --lr_scheduler_type "cosine_with_min_lr" --model_max_length 1024 --logging_steps 1 --tf32 True --gradient_checkpointing True --dataloader_num_workers 4 --lazy_preprocess True --n_query 256 --n_und_query 0 --report_to wandb --fix_dit True --fix_connect True --fix_llm True --fix_vit False --lora_r 16      ```
**eval_csgo_loc.py**  65: 8000 修复eval_loc加载权重bug： 23450
```    CUDA_VISIBLE_DEVICES=0 python eval_csgo_loc.py --csgo_config csgo_configs/test/exp4_13_6_loc.yaml      ```

#### exp4_13_6_1
- exp4_6_3
<!-- - action_dit_layer 24 -->
- bs 128
- is_aciton_dit_vae_small_init: True
- logger + wandb => Trainer

- use_pi05_action_dit=False
- use_vit_regression_head: False
- use_vit_cls_regression_head: False
- use_codex_vit_regression_head: True
- loc_use_circular_loss: True (loc_xy_loss_weight: 1.0 loc_z_loss_weight: 1.0 loc_angle_loss_weight: 2.0)
- img_size=448
- fix_vit: False
- is_lora: True

- action_dit_projector_lr: 1e-4
- lr=5e-5
- alpha_loc=1

```     CUDA_VISIBLE_DEVICES=0 MASTER_ADDR=127.0.0.1 MASTER_PORT=29509 RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 python train_csgo.py --csgo_config csgo_configs/exp4_13_6_1.yaml --deepspeed deepspeed_scripts/zero0.json --model_name_or_path UniLIP-1B --unilip_factor 10.6 --mllm_hf_path OpenGVLab/InternVL3-1B-hf --version internvl --data_type "mix" --csgo_image_folder data/preprocessed_data --mm_use_im_start_end False --mm_use_im_patch_token False --bf16 True --output_dir outputs/csgo_1b/exp4_13_6_1 --num_train_epochs 50 --per_device_train_batch_size 128 --per_device_eval_batch_size 128 --gradient_accumulation_steps 1 --eval_strategy "no" --save_strategy "steps" --save_steps 8000 --save_total_limit 3 --learning_rate 5e-5 --weight_decay 0. --warmup_ratio 0.003 --lr_scheduler_type "cosine_with_min_lr" --model_max_length 1024 --logging_steps 1 --tf32 True --gradient_checkpointing True --dataloader_num_workers 4 --lazy_preprocess True --n_query 256 --n_und_query 0 --report_to wandb --fix_dit True --fix_connect True --fix_llm True --fix_vit False --lora_r 16      ```
**eval_csgo_loc.py**  调整超参： 8000 23400
```    CUDA_VISIBLE_DEVICES=1 python eval_csgo_loc.py --csgo_config csgo_configs/test/exp4_13_6_1_loc.yaml      ```

### exp4_13_7
- exp4_6_3
<!-- - action_dit_layer 24 -->
- bs 128
- is_aciton_dit_vae_small_init: True
- logger + wandb => Trainer
- action_dit_projector_lr: 5e-4

- use_pi05_action_dit=False
- use_vit_regression_head: False
- use_vit_cls_regression_head: False
- use_codex_vit_regression_head: True
- loc_use_circular_loss: True (loc_xy_loss_weight: 1.0 loc_z_loss_weight: 1.0 loc_angle_loss_weight: 2.0)
- img_size=448
- fix_vit: False
- is_lora: False

```     CUDA_VISIBLE_DEVICES=2 MASTER_ADDR=127.0.0.1 MASTER_PORT=29509 RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 python train_csgo.py --csgo_config csgo_configs/exp4_13_7.yaml --deepspeed deepspeed_scripts/zero0.json --model_name_or_path UniLIP-1B --unilip_factor 10.6 --mllm_hf_path OpenGVLab/InternVL3-1B-hf --version internvl --data_type "mix" --csgo_image_folder data/preprocessed_data --mm_use_im_start_end False --mm_use_im_patch_token False --bf16 True --output_dir outputs/csgo_1b/exp4_13_7 --num_train_epochs 50 --per_device_train_batch_size 128 --per_device_eval_batch_size 128 --gradient_accumulation_steps 1 --eval_strategy "no" --save_strategy "steps" --save_steps 8000 --save_total_limit 3 --learning_rate 1e-4 --weight_decay 0. --warmup_ratio 0.003 --lr_scheduler_type "cosine_with_min_lr" --model_max_length 1024 --logging_steps 1 --tf32 True --gradient_checkpointing True --dataloader_num_workers 4 --lazy_preprocess True --n_query 256 --n_und_query 0 --report_to wandb --fix_dit True --fix_connect True --fix_llm True --fix_vit False      ```
**eval_csgo_loc.py** fst a100: 16000 23400 修复eval_loc加载权重bug：23400
```    CUDA_VISIBLE_DEVICES=0 python eval_csgo_loc.py --csgo_config csgo_configs/test/exp4_13_7_loc.yaml      ```

#### exp4_13_7_1
- exp4_6_3
<!-- - action_dit_layer 24 -->
- bs 128
- is_aciton_dit_vae_small_init: True
- logger + wandb => Trainer
- action_dit_projector_lr: 5e-4

- use_pi05_action_dit=False
- use_vit_regression_head: False
- use_vit_cls_regression_head: False
- use_codex_vit_regression_head: True
- loc_use_circular_loss: True (loc_xy_loss_weight: 1.0 loc_z_loss_weight: 1.0 loc_angle_loss_weight: 2.0)
- img_size=448
- fix_vit: False
- is_lora: False

- action_dit_projector_lr: 1e-4
- lr=5e-5
- alpha_loc=1

```     CUDA_VISIBLE_DEVICES=1 MASTER_ADDR=127.0.0.1 MASTER_PORT=29505 RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 python train_csgo.py --csgo_config csgo_configs/exp4_13_7_1.yaml --deepspeed deepspeed_scripts/zero0.json --model_name_or_path UniLIP-1B --unilip_factor 10.6 --mllm_hf_path OpenGVLab/InternVL3-1B-hf --version internvl --data_type "mix" --csgo_image_folder data/preprocessed_data --mm_use_im_start_end False --mm_use_im_patch_token False --bf16 True --output_dir outputs/csgo_1b/exp4_13_7_1 --num_train_epochs 50 --per_device_train_batch_size 128 --per_device_eval_batch_size 128 --gradient_accumulation_steps 1 --eval_strategy "no" --save_strategy "steps" --save_steps 8000 --save_total_limit 3 --learning_rate 5e-5 --weight_decay 0. --warmup_ratio 0.003 --lr_scheduler_type "cosine_with_min_lr" --model_max_length 1024 --logging_steps 1 --tf32 True --gradient_checkpointing True --dataloader_num_workers 4 --lazy_preprocess True --n_query 256 --n_und_query 0 --report_to wandb --fix_dit True --fix_connect True --fix_llm True --fix_vit False      ```
**eval_csgo_loc.py** 调整超参： 8000 23400
```    CUDA_VISIBLE_DEVICES=1 python eval_csgo_loc.py --csgo_config csgo_configs/test/exp4_13_7_1_loc.yaml      ```

### exp4_13_8
- exp4_6_3
<!-- - action_dit_layer 24 -->
- bs 128
- is_aciton_dit_vae_small_init: True
- logger + wandb => Trainer

- use_pi05_action_dit=False
- use_vit_regression_head: False
- use_vit_cls_regression_head: False
- use_codex_vit_regression_head: True
- loc_use_circular_loss: True (loc_xy_loss_weight: 1.0 loc_z_loss_weight: 1.0 loc_angle_loss_weight: 2.0)


- is_lora: False
- img_size=224
- fix_vit: True
- alpha_loc=1
- action_dit_projector_lr: 5e-4
- is_aciton_dit_vae_small_init: False

```     CUDA_VISIBLE_DEVICES=0 MASTER_ADDR=127.0.0.1 MASTER_PORT=29505 RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 python train_csgo.py --csgo_config csgo_configs/exp4_13_8.yaml --deepspeed deepspeed_scripts/zero0.json --model_name_or_path UniLIP-1B --unilip_factor 10.6 --mllm_hf_path OpenGVLab/InternVL3-1B-hf --version internvl --data_type "mix" --csgo_image_folder data/preprocessed_data --mm_use_im_start_end False --mm_use_im_patch_token False --bf16 True --output_dir outputs/csgo_1b/exp4_13_8 --num_train_epochs 50 --per_device_train_batch_size 64 --per_device_eval_batch_size 64 --gradient_accumulation_steps 2 --eval_strategy "no" --save_strategy "steps" --save_steps 8000 --save_total_limit 3 --learning_rate 1e-4 --weight_decay 0. --warmup_ratio 0.003 --lr_scheduler_type "cosine_with_min_lr" --model_max_length 1024 --logging_steps 1 --tf32 True --gradient_checkpointing True --dataloader_num_workers 4 --lazy_preprocess True --n_query 256 --n_und_query 0 --report_to wandb --fix_dit True --fix_connect True --fix_llm True --fix_vit True      ```
**eval_csgo_loc.py**67: 23450
```    CUDA_VISIBLE_DEVICES=0 python eval_csgo_loc.py --csgo_config csgo_configs/test/exp4_13_8_loc.yaml      ```

### exp4_13_9
- exp4_6_3
<!-- - action_dit_layer 24 -->
- bs 128
- is_aciton_dit_vae_small_init: True
- logger + wandb => Trainer

- use_pi05_action_dit=False
- use_vit_regression_head: False
- use_vit_cls_regression_head: False
- use_codex_vit_regression_head: True
- loc_use_circular_loss: True (loc_xy_loss_weight: 1.0 loc_z_loss_weight: 1.0 loc_angle_loss_weight: 2.0)

- is_lora: False
- img_size=448
- fix_vit: True
- alpha_loc=1
- action_dit_projector_lr: 5e-4
- is_aciton_dit_vae_small_init: False

```     CUDA_VISIBLE_DEVICES=0 MASTER_ADDR=127.0.0.1 MASTER_PORT=29507 RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 python train_csgo.py --csgo_config csgo_configs/exp4_13_9.yaml --deepspeed deepspeed_scripts/zero0.json --model_name_or_path UniLIP-1B --unilip_factor 10.6 --mllm_hf_path OpenGVLab/InternVL3-1B-hf --version internvl --data_type "mix" --csgo_image_folder data/preprocessed_data --mm_use_im_start_end False --mm_use_im_patch_token False --bf16 True --output_dir outputs/csgo_1b/exp4_13_9 --num_train_epochs 50 --per_device_train_batch_size 64 --per_device_eval_batch_size 64 --gradient_accumulation_steps 2 --eval_strategy "no" --save_strategy "steps" --save_steps 8000 --save_total_limit 3 --learning_rate 1e-4 --weight_decay 0. --warmup_ratio 0.003 --lr_scheduler_type "cosine_with_min_lr" --model_max_length 1024 --logging_steps 1 --tf32 True --gradient_checkpointing True --dataloader_num_workers 4 --lazy_preprocess True --n_query 256 --n_und_query 0 --report_to wandb --fix_dit True --fix_connect True --fix_llm True --fix_vit True      ```
**eval_csgo_loc.py**67: 8000 23450
```    CUDA_VISIBLE_DEVICES=0 python eval_csgo_loc.py --csgo_config csgo_configs/test/exp4_13_9_loc.yaml      ```




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
**eval_csgo_loc.py** step=10000 40000
```    CUDA_VISIBLE_DEVICES=0 python eval_csgo_loc.py --csgo_config csgo_configs/test/exp8_loc.yaml      ```
**eval_csgo.py** step=12000 40000
``    CUDA_VISIBLE_DEVICES=0 python eval_csgo.py --csgo_config csgo_configs/test/exp8_gen.yaml      ``
    **benchmark_csgo.py**
    ``      CUDA_VISIBLE_DEVICES=0 python benchmark_csgo.py --gt data/preprocessed_data/de_dust2/imgs --pred outputs_eval/exp8_gen/test_20260209_161159/gen_imgs/de_dust2 --all --batch_size 8       ``
    ``      CUDA_VISIBLE_DEVICES=0 python benchmark_csgo.py --gt data/preprocessed_data/de_nuke/imgs --pred outputs_eval/exp8_gen/test_20260209_161159/gen_imgs/de_nuke --all --batch_size 8       ``
    ``      CUDA_VISIBLE_DEVICES=0 python benchmark_csgo.py --gt data/preprocessed_data/de_ancient/imgs --pred outputs_eval/exp8_gen/test_20260209_161159/gen_imgs/de_ancient --all --batch_size 8       ``
**continuous gen**  step=16000 40000
``    CUDA_VISIBLE_DEVICES=0 python eval_csgo.py --csgo_config csgo_configs/test/exp8_gen_conti.yaml      ``
    **benchmark_csgo.py**
    ``      CUDA_VISIBLE_DEVICES=0 python benchmark_csgo.py --gt data/preprocessed_data/de_dust2/imgs --pred outputs_eval/exp8_gen_conti/test_20260209_161230/gen_imgs/de_dust2 --all --batch_size 8       ``
    ``      CUDA_VISIBLE_DEVICES=0 python benchmark_csgo.py --gt data/preprocessed_data/de_nuke/imgs --pred outputs_eval/exp8_gen_conti/test_20260209_161230/gen_imgs/de_nuke --all --batch_size 8       ``
    ``      CUDA_VISIBLE_DEVICES=0 python benchmark_csgo.py --gt data/preprocessed_data/de_ancient/imgs --pred outputs_eval/exp8_gen_conti/test_20260209_161230/gen_imgs/de_ancient --all --batch_size 8       ``
    **frames to video**
    ``    python frames_to_video.py --img_dir outputs_eval/exp8_gen_conti/test_20260209_161230/gen_imgs/de_dust2 --gt_dir data/preprocessed_data/de_dust2/imgs --output_dir outputs_eval/exp8_gen_conti/test_20260209_161230/gen_compared_videos/de_dust2 --max_duration 10        ``
    ``    python frames_to_video.py --img_dir outputs_eval/exp8_gen_conti/test_20260209_161230/gen_imgs/de_nuke --gt_dir data/preprocessed_data/de_nuke/imgs --output_dir outputs_eval/exp8_gen_conti/test_20260209_161230/gen_compared_videos/de_nuke --max_duration 10        ``
    ``    python frames_to_video.py --img_dir outputs_eval/exp8_gen_conti/test_20260209_161230/gen_imgs/de_ancient --gt_dir data/preprocessed_data/de_ancient/imgs --output_dir outputs_eval/exp8_gen_conti/test_20260209_161230/gen_compared_videos/de_ancient --max_duration 10        ``



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



## exp10
- cp from exp8
- action_dit_projector_lr: 0.0005
- action_dit_layer 6
- is_aciton_dit_vae_small_init: True
- alpha_loc_aux_loss: 0.1
- alpha_loc_loss: 2

- logger + wandb => Trainer
- lora=16
**train_csgo.py**
```     CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node=1 --master_port=29505 train_csgo.py --csgo_config csgo_configs/exp10.yaml --deepspeed deepspeed_scripts/zero0.json --model_name_or_path UniLIP-1B --unilip_factor 10.6 --mllm_hf_path OpenGVLab/InternVL3-1B-hf --version internvl --data_type "mix" --csgo_image_folder data/preprocessed_data --mm_use_im_start_end False --mm_use_im_patch_token False --bf16 True --output_dir outputs/csgo_1b/exp10 --num_train_epochs 100 --per_device_train_batch_size 2 --per_device_eval_batch_size 2 --gradient_accumulation_steps 1 --eval_strategy "no" --save_strategy "steps" --save_steps 1000 --save_total_limit 1 --learning_rate 1e-4 --weight_decay 0. --warmup_ratio 0.003 --lr_scheduler_type "cosine_with_min_lr" --model_max_length 1024 --logging_steps 1 --tf32 True --gradient_checkpointing True --dataloader_num_workers 4 --lazy_preprocess True --n_query 256 --n_und_query 0 --report_to wandb --fix_dit False --fix_connect False --fix_llm True --action_dit_layer 6 --lora_r 16    ```



## exp10_1
- cp from exp8
- action_dit_projector_lr: 0.0005
- is_aciton_dit_vae_small_init: True
- alpha_loc_aux_loss: 0.1
- alpha_loc_loss: 2
- logger + wandb => Trainer
- lora=16

- action_dit_layer 3
- img_size=224

**train_csgo.py**
单卡
```     CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node=1 --master_port=29504 train_csgo.py --csgo_config csgo_configs/exp10_1.yaml --deepspeed deepspeed_scripts/zero0.json --model_name_or_path UniLIP-1B --unilip_factor 10.6 --mllm_hf_path OpenGVLab/InternVL3-1B-hf --version internvl --data_type "mix" --csgo_image_folder data/preprocessed_data --mm_use_im_start_end False --mm_use_im_patch_token False --bf16 True --output_dir outputs/csgo_1b/exp10_1 --num_train_epochs 100 --per_device_train_batch_size 128 --per_device_eval_batch_size 128 --gradient_accumulation_steps 1 --eval_strategy "no" --save_strategy "steps" --save_steps 1000 --save_total_limit 1 --learning_rate 1e-4 --weight_decay 0. --warmup_ratio 0.003 --lr_scheduler_type "cosine_with_min_lr" --model_max_length 1024 --logging_steps 1 --tf32 True --gradient_checkpointing True --dataloader_num_workers 4 --lazy_preprocess True --n_query 256 --n_und_query 0 --report_to wandb --fix_dit False --fix_connect False --fix_llm True --action_dit_layer 3 --lora_r 16    ```
多卡，resume时需配合修改yaml实现单卡resume至多卡
```     CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=29504 train_csgo.py --csgo_config csgo_configs/exp10_1.yaml --deepspeed deepspeed_scripts/zero0.json --model_name_or_path UniLIP-1B --unilip_factor 10.6 --mllm_hf_path OpenGVLab/InternVL3-1B-hf --version internvl --data_type "mix" --csgo_image_folder data/preprocessed_data --mm_use_im_start_end False --mm_use_im_patch_token False --bf16 True --output_dir outputs/csgo_1b/exp10_1 --num_train_epochs 100 --per_device_train_batch_size 256 --per_device_eval_batch_size 256 --gradient_accumulation_steps 1 --eval_strategy "no" --save_strategy "steps" --save_steps 1000 --save_total_limit 1 --learning_rate 1e-4 --weight_decay 0. --warmup_ratio 0.003 --lr_scheduler_type "cosine_with_min_lr" --model_max_length 1024 --logging_steps 1 --tf32 True --gradient_checkpointing True --dataloader_num_workers 1 --lazy_preprocess True --n_query 256 --n_und_query 0 --report_to wandb --fix_dit False --fix_connect False --fix_llm True --action_dit_layer 3 --lora_r 16    ```
**eval_csgo_loc.py** step=14000
```    CUDA_VISIBLE_DEVICES=1 python eval_csgo_loc.py --csgo_config csgo_configs/test/exp10_1_loc.yaml      ```
**eval_csgo.py** step=14000
```    CUDA_VISIBLE_DEVICES=1 python eval_csgo.py --csgo_config csgo_configs/test/exp10_1_gen.yaml      ```
    **benchmark_csgo.py**
**continuous gen**  step=14000
```    CUDA_VISIBLE_DEVICES=1 python eval_csgo.py --csgo_config csgo_configs/test/exp10_1_gen_conti.yaml      ```
    **benchmark_csgo.py**
    **frames to video**



## exp10_2
- cp from exp10_1
- action_dit_projector_lr: 0.0005
- is_aciton_dit_vae_small_init: True
<!-- - alpha_loc_aux_loss: 0.1 -->
- alpha_loc_loss: 2
- logger + wandb => Trainer
- lora=16
- action_dit_layer 3
- img_size=224

- is_loc_aux_loss: False

**train_csgo.py**
```     CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node=1 --master_port=29504 train_csgo.py --csgo_config csgo_configs/exp10_2.yaml --deepspeed deepspeed_scripts/zero0.json --model_name_or_path UniLIP-1B --unilip_factor 10.6 --mllm_hf_path OpenGVLab/InternVL3-1B-hf --version internvl --data_type "mix" --csgo_image_folder data/preprocessed_data --mm_use_im_start_end False --mm_use_im_patch_token False --bf16 True --output_dir outputs/csgo_1b/exp10_2 --num_train_epochs 100 --per_device_train_batch_size 128 --per_device_eval_batch_size 128 --gradient_accumulation_steps 1 --eval_strategy "no" --save_strategy "steps" --save_steps 1000 --save_total_limit 1 --learning_rate 1e-4 --weight_decay 0. --warmup_ratio 0.003 --lr_scheduler_type "cosine_with_min_lr" --model_max_length 1024 --logging_steps 1 --tf32 True --gradient_checkpointing True --dataloader_num_workers 4 --lazy_preprocess True --n_query 256 --n_und_query 0 --report_to wandb --fix_dit False --fix_connect False --fix_llm True --action_dit_layer 3 --lora_r 16    ```



## exp11
- cp from exp8
- action_dit_projector_lr: 0.0005
- is_aciton_dit_vae_small_init: True
- alpha_loc_aux_loss: 0.1
- alpha_loc_loss: 2
- logger + wandb => Trainer
- lora=16
<!-- - action_dit_layer 3 -->
- img_size=224

- use_pi05_action_dit=True
- bs=128; grad_accum=2

像openpi一样，需要把自己实现的带AdaRMS的NORM代码复制到transformers中进行替换
**~~cp -r unilip/openpi_src/models_pytorch/transformers_replace/* /home/jiahao/miniconda3/envs/UniLIP/lib/python3.11/site-packages/transformers/~~**
**cp -r unilip/openpi_src/models_pytorch/transformers_4573_replace/gemma/* /home/jiahao/miniconda3/envs/UniLIP/lib/python3.11/site-packages/transformers/models/gemma/**

~~也可以使用下述脚本执行：~~
~~**python setup_transformers.py**~~
~~**grep "cond_dim" $(python -c "import transformers, os; print(os.path.join(os.path.dirname(transformers.__file__), 'models/gemma/modeling_gemma.py'))")**~~
~~并且需要降级transfomers版本到openpi使用的4.53.2。~~
~~降级后--deepspeed deepspeed_scripts/zero0.json会报错，改为直接在代码中显式写死这个参数。~~
恢复4.57.3版本，直接在该版本上覆盖replace的代码。

**train_csgo.py**
```     CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=29505 train_csgo.py --csgo_config csgo_configs/exp11.yaml --deepspeed deepspeed_scripts/zero0.json --model_name_or_path UniLIP-1B --unilip_factor 10.6 --mllm_hf_path OpenGVLab/InternVL3-1B-hf --version internvl --data_type "mix" --csgo_image_folder data/preprocessed_data --mm_use_im_start_end False --mm_use_im_patch_token False --bf16 True --output_dir outputs/csgo_1b/exp11 --num_train_epochs 100 --per_device_train_batch_size 128 --per_device_eval_batch_size 128 --gradient_accumulation_steps 4 --eval_strategy "no" --save_strategy "steps" --save_steps 1000 --save_total_limit 2 --learning_rate 1e-4 --weight_decay 0. --warmup_ratio 0.003 --lr_scheduler_type "cosine_with_min_lr" --model_max_length 1024 --logging_steps 1 --tf32 True --gradient_checkpointing True --dataloader_num_workers 2 --lazy_preprocess True --n_query 256 --n_und_query 0 --report_to wandb --fix_dit False --fix_connect False --fix_llm True --lora_r 16    ```
**eval_csgo_loc.py** 67: step=3000 4000 5000 误删ckpt重新训练65: 5000 11700
```    CUDA_VISIBLE_DEVICES=1 python eval_csgo_loc.py --csgo_config csgo_configs/test/exp11_loc.yaml      ```
**eval_csgo.py** 67: step=3000 5000 误删ckpt重新训练65: 11700
```    CUDA_VISIBLE_DEVICES=1 python eval_csgo.py --csgo_config csgo_configs/test/exp11_gen.yaml      ```
    **benchmark_csgo.py**
    ``      CUDA_VISIBLE_DEVICES=1 python benchmark_csgo.py --gt data/preprocessed_data/de_dust2/imgs --pred outputs_eval/exp11_gen/test_20260331_163844/gen_imgs/de_dust2 --all --batch_size 8       ``
    ``      CUDA_VISIBLE_DEVICES=1 python benchmark_csgo.py --gt data/preprocessed_data/de_nuke/imgs --pred outputs_eval/exp11_gen/test_20260331_163844/gen_imgs/de_nuke --all --batch_size 8       ``
    ``      CUDA_VISIBLE_DEVICES=1 python benchmark_csgo.py --gt data/preprocessed_data/de_ancient/imgs --pred outputs_eval/exp11_gen/test_20260331_163844/gen_imgs/de_ancient --all --batch_size 8       ``
**continuous gen**  67: step=3000 5000 误删ckpt重新训练65: 11700
```    CUDA_VISIBLE_DEVICES=1 python eval_csgo.py --csgo_config csgo_configs/test/exp11_gen_conti.yaml      ```
    **benchmark_csgo.py**
    ``      CUDA_VISIBLE_DEVICES=1 python benchmark_csgo.py --gt data/preprocessed_data/de_dust2/imgs --pred outputs_eval/exp11_gen_conti/test_20260331_163847/gen_imgs/de_dust2 --all --batch_size 8       ``
    ``      CUDA_VISIBLE_DEVICES=1 python benchmark_csgo.py --gt data/preprocessed_data/de_nuke/imgs --pred outputs_eval/exp11_gen_conti/test_20260331_163847/gen_imgs/de_nuke --all --batch_size 8       ``
    ``      CUDA_VISIBLE_DEVICES=1 python benchmark_csgo.py --gt data/preprocessed_data/de_ancient/imgs --pred outputs_eval/exp11_gen_conti/test_20260331_163847/gen_imgs/de_ancient --all --batch_size 8       ``
    **frames to video**



## exp13
- cp from exp11
- action_dit_projector_lr: 0.0005
- is_aciton_dit_vae_small_init: True
- alpha_loc_aux_loss: 0.1
- alpha_loc_loss: 2
- logger + wandb => Trainer
- lora=16
<!-- - action_dit_layer 3 -->
- img_size=224
- use_pi05_action_dit=True
- bs=128; grad_accum=2

- is_loc_aux_loss: False

**train_csgo.py**
```     CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=29505 train_csgo.py --csgo_config csgo_configs/exp13.yaml --deepspeed deepspeed_scripts/zero0.json --model_name_or_path UniLIP-1B --unilip_factor 10.6 --mllm_hf_path OpenGVLab/InternVL3-1B-hf --version internvl --data_type "mix" --csgo_image_folder data/preprocessed_data --mm_use_im_start_end False --mm_use_im_patch_token False --bf16 True --output_dir outputs/csgo_1b/exp13 --num_train_epochs 100 --per_device_train_batch_size 512 --per_device_eval_batch_size 512 --gradient_accumulation_steps 2 --eval_strategy "no" --save_strategy "steps" --save_steps 1000 --save_total_limit 1 --learning_rate 1e-4 --weight_decay 0. --warmup_ratio 0.003 --lr_scheduler_type "cosine_with_min_lr" --model_max_length 1024 --logging_steps 1 --tf32 True --gradient_checkpointing True --dataloader_num_workers 4 --lazy_preprocess True --n_query 256 --n_und_query 0 --report_to wandb --fix_dit False --fix_connect False --fix_llm True --lora_r 16    ```
**eval_csgo_loc.py** 65: step=4000  67：9000 11700
```    CUDA_VISIBLE_DEVICES=0 python eval_csgo_loc.py --csgo_config csgo_configs/test/exp13_loc.yaml      ```
**eval_csgo.py** 67:
```    CUDA_VISIBLE_DEVICES=1 python eval_csgo.py --csgo_config csgo_configs/test/exp13_gen.yaml      ```
    **benchmark_csgo.py**
**continuous gen**  67:
```    CUDA_VISIBLE_DEVICES=1 python eval_csgo.py --csgo_config csgo_configs/test/exp13_gen_conti.yaml      ```
    **benchmark_csgo.py**
    **frames to video**





## exp16
- fork from exp2
- resume_ckpt_path: "outputs/csgo_1b/exp2/checkpoint-46800/model.safetensors"

- use_external_loc_model: True
- external_loc_use_circular_loss: True
- external_loc_repo_root: "csgosquare"
- external_loc_config_path: "configs_reg_newdata/exp5_2.yaml"
- external_loc_checkpoint_path: "checkpoints_reg_newdata/exp5_2/20251227_091745/current_model.pth"
- img_size: 224
- is_lora: False
- alpha_loc_aux_loss: 1000
- 仅gen数据

**train_csgo.py**
```     CUDA_VISIBLE_DEVICES=3 torchrun --nproc_per_node=1 --master_port=29510 train_csgo.py --csgo_config csgo_configs/exp16.yaml --deepspeed deepspeed_scripts/zero0.json --model_name_or_path UniLIP-1B --unilip_factor 10.6 --mllm_hf_path OpenGVLab/InternVL3-1B-hf --version internvl --data_type "mix" --csgo_image_folder data/preprocessed_data --mm_use_im_start_end False --mm_use_im_patch_token False --bf16 True --output_dir outputs/csgo_1b/exp16 --num_train_epochs 10 --per_device_train_batch_size 8 --per_device_eval_batch_size 8 --gradient_accumulation_steps 16 --eval_strategy "no" --save_strategy "steps" --save_steps 1000 --save_total_limit 2 --learning_rate 1e-4 --weight_decay 0. --warmup_ratio 0.003 --lr_scheduler_type "cosine_with_min_lr" --model_max_length 1024 --logging_steps 1 --tf32 True --gradient_checkpointing True --dataloader_num_workers 16 --lazy_preprocess True --n_query 256 --n_und_query 0 --report_to wandb --fix_dit False --fix_connect False --fix_llm True       ```
**eval_csgo_loc.py**
**eval_csgo.py** fst step=1000 4690
``    CUDA_VISIBLE_DEVICES=0 python eval_csgo.py --csgo_config csgo_configs/test/exp16_gen.yaml      ``
    **benchmark_csgo.py**
    ``      CUDA_VISIBLE_DEVICES=0 python benchmark_csgo.py --gt data/preprocessed_data/de_dust2/imgs --pred outputs_eval/exp16_gen/test_20260329_174845/gen_imgs/de_dust2 --all --batch_size 8       ``
    ``      CUDA_VISIBLE_DEVICES=0 python benchmark_csgo.py --gt data/preprocessed_data/de_nuke/imgs --pred outputs_eval/exp16_gen/test_20260329_174845/gen_imgs/de_nuke --all --batch_size 8       ``
    ``      CUDA_VISIBLE_DEVICES=0 python benchmark_csgo.py --gt data/preprocessed_data/de_ancient/imgs --pred outputs_eval/exp16_gen/test_20260329_174845/gen_imgs/de_ancient --all --batch_size 8       ``
**continuous gen** fst step=4690
```        CUDA_VISIBLE_DEVICES=0 python eval_csgo.py --csgo_config csgo_configs/test/exp16_gen_conti.yaml      ```
    **benchmark_csgo.py**
    ``      CUDA_VISIBLE_DEVICES=0 python benchmark_csgo.py --gt data/preprocessed_data/de_dust2/imgs --pred outputs_eval/exp16_gen_conti/test_20260330_152302/gen_imgs/de_dust2 --all --batch_size 8       ``
    ``      CUDA_VISIBLE_DEVICES=0 python benchmark_csgo.py --gt data/preprocessed_data/de_nuke/imgs --pred outputs_eval/exp16_gen_conti/test_20260330_152302/gen_imgs/de_nuke --all --batch_size 8       ``
    ``      CUDA_VISIBLE_DEVICES=0 python benchmark_csgo.py --gt data/preprocessed_data/de_ancient/imgs --pred outputs_eval/exp16_gen_conti/test_20260330_152302/gen_imgs/de_ancient --all --batch_size 8       ``
    **frames to video**

### exp16_1
- fork from exp2
- resume_ckpt_path: "outputs/csgo_1b/exp2/checkpoint-46800/model.safetensors"
- use_external_loc_model: True
- external_loc_use_circular_loss: True
- external_loc_repo_root: "csgosquare"
- external_loc_config_path: "configs_reg_newdata/exp5_2.yaml"
- external_loc_checkpoint_path: "checkpoints_reg_newdata/exp5_2/20251227_091745/current_model.pth"
- img_size: 224
- is_lora: False
- alpha_loc_aux_loss: 1000
- 仅gen数据

- compare to exp16
- is_loc_aux_loss: False

**train_csgo.py**
```     CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=29511 train_csgo.py --csgo_config csgo_configs/exp16_1.yaml --deepspeed deepspeed_scripts/zero0.json --model_name_or_path UniLIP-1B --unilip_factor 10.6 --mllm_hf_path OpenGVLab/InternVL3-1B-hf --version internvl --data_type "mix" --csgo_image_folder data/preprocessed_data --mm_use_im_start_end False --mm_use_im_patch_token False --bf16 True --output_dir outputs/csgo_1b/exp16_1 --num_train_epochs 10 --per_device_train_batch_size 8 --per_device_eval_batch_size 8 --gradient_accumulation_steps 16 --eval_strategy "no" --save_strategy "steps" --save_steps 1000 --save_total_limit 2 --learning_rate 1e-4 --weight_decay 0. --warmup_ratio 0.003 --lr_scheduler_type "cosine_with_min_lr" --model_max_length 1024 --logging_steps 1 --tf32 True --gradient_checkpointing True --dataloader_num_workers 16 --lazy_preprocess True --n_query 256 --n_und_query 0 --report_to wandb --fix_dit False --fix_connect False --fix_llm True       ```
**eval_csgo_loc.py**
**eval_csgo.py** fst step=1000 4690
``    CUDA_VISIBLE_DEVICES=0 python eval_csgo.py --csgo_config csgo_configs/test/exp16_1_gen.yaml      ``
    **benchmark_csgo.py**
    ``      CUDA_VISIBLE_DEVICES=0 python benchmark_csgo.py --gt data/preprocessed_data/de_dust2/imgs --pred outputs_eval/exp16_1_gen/test_20260329_174903/gen_imgs/de_dust2 --all --batch_size 8       ``
    ``      CUDA_VISIBLE_DEVICES=0 python benchmark_csgo.py --gt data/preprocessed_data/de_nuke/imgs --pred outputs_eval/exp16_1_gen/test_20260329_174903/gen_imgs/de_nuke --all --batch_size 8       ``
    ``      CUDA_VISIBLE_DEVICES=0 python benchmark_csgo.py --gt data/preprocessed_data/de_ancient/imgs --pred outputs_eval/exp16_1_gen/test_20260329_174903/gen_imgs/de_ancient --all --batch_size 8       ``
**continuous gen** fst step=4690
```        CUDA_VISIBLE_DEVICES=1 python eval_csgo.py --csgo_config csgo_configs/test/exp16_1_gen_conti.yaml      ```
    **benchmark_csgo.py**
    ``      CUDA_VISIBLE_DEVICES=1 python benchmark_csgo.py --gt data/preprocessed_data/de_dust2/imgs --pred outputs_eval/exp16_1_gen_conti/test_20260330_152325/gen_imgs/de_dust2 --all --batch_size 8       ``
    ``      CUDA_VISIBLE_DEVICES=1 python benchmark_csgo.py --gt data/preprocessed_data/de_nuke/imgs --pred outputs_eval/exp16_1_gen_conti/test_20260330_152325/gen_imgs/de_nuke --all --batch_size 8       ``
    ``      CUDA_VISIBLE_DEVICES=1 python benchmark_csgo.py --gt data/preprocessed_data/de_ancient/imgs --pred outputs_eval/exp16_1_gen_conti/test_20260330_152325/gen_imgs/de_ancient --all --batch_size 8       ``
    **frames to video**
    ``    python frames_to_video.py --img_dir outputs_eval/exp16_1_gen_conti/test_20260330_152325/gen_imgs/de_dust2 --gt_dir data/preprocessed_data/de_dust2/imgs --output_dir outputs_eval/exp16_1_gen_conti/test_20260330_152325/gen_compared_videos/de_dust2 --max_duration 10        ``
    ``    python frames_to_video.py --img_dir outputs_eval/exp16_1_gen_conti/test_20260330_152325/gen_imgs/de_nuke --gt_dir data/preprocessed_data/de_nuke/imgs --output_dir outputs_eval/exp16_1_gen_conti/test_20260330_152325/gen_compared_videos/de_nuke --max_duration 10        ``
    ``    python frames_to_video.py --img_dir outputs_eval/exp16_1_gen_conti/test_20260330_152325/gen_imgs/de_ancient --gt_dir data/preprocessed_data/de_ancient/imgs --output_dir outputs_eval/exp16_1_gen_conti/test_20260330_152325/gen_compared_videos/de_ancient --max_duration 10        ``

### exp16_2
- fork from exp2
- resume_ckpt_path: "outputs/csgo_1b/exp2/checkpoint-46800/model.safetensors"
- use_external_loc_model: True
- external_loc_use_circular_loss: True
- external_loc_repo_root: "csgosquare"
- external_loc_config_path: "configs_reg_newdata/exp5_2.yaml"
- external_loc_checkpoint_path: "checkpoints_reg_newdata/exp5_2/20251227_091745/current_model.pth"
- img_size: 224
- is_lora: False
- 仅gen数据

- exp16调参
- is_loc_aux_loss: True
- alpha_loc_aux_loss: 100

**train_csgo.py**
```     CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node=1 --master_port=29511 train_csgo.py --csgo_config csgo_configs/exp16_2.yaml --deepspeed deepspeed_scripts/zero0.json --model_name_or_path UniLIP-1B --unilip_factor 10.6 --mllm_hf_path OpenGVLab/InternVL3-1B-hf --version internvl --data_type "mix" --csgo_image_folder data/preprocessed_data --mm_use_im_start_end False --mm_use_im_patch_token False --bf16 True --output_dir outputs/csgo_1b/exp16_2 --num_train_epochs 10 --per_device_train_batch_size 8 --per_device_eval_batch_size 8 --gradient_accumulation_steps 16 --eval_strategy "no" --save_strategy "steps" --save_steps 1000 --save_total_limit 2 --learning_rate 1e-4 --weight_decay 0. --warmup_ratio 0.003 --lr_scheduler_type "cosine_with_min_lr" --model_max_length 1024 --logging_steps 1 --tf32 True --gradient_checkpointing True --dataloader_num_workers 16 --lazy_preprocess True --n_query 256 --n_und_query 0 --report_to wandb --fix_dit False --fix_connect False --fix_llm True       ```
**eval_csgo.py** fst step=4690
``    CUDA_VISIBLE_DEVICES=0 python eval_csgo.py --csgo_config csgo_configs/test/exp16_2_gen.yaml      ``
    **benchmark_csgo.py**
    ``      CUDA_VISIBLE_DEVICES=0 python benchmark_csgo.py --gt data/preprocessed_data/de_dust2/imgs --pred outputs_eval/exp16_2_gen/test_20260401_214653/gen_imgs/de_dust2 --all --batch_size 8       ``
    ``      CUDA_VISIBLE_DEVICES=0 python benchmark_csgo.py --gt data/preprocessed_data/de_nuke/imgs --pred outputs_eval/exp16_2_gen/test_20260401_214653/gen_imgs/de_nuke --all --batch_size 8       ``
    ``      CUDA_VISIBLE_DEVICES=0 python benchmark_csgo.py --gt data/preprocessed_data/de_ancient/imgs --pred outputs_eval/exp16_2_gen/test_20260401_214653/gen_imgs/de_ancient --all --batch_size 8       ``
**continuous gen** fst step=4690
```        CUDA_VISIBLE_DEVICES=1 python eval_csgo.py --csgo_config csgo_configs/test/exp16_2_gen_conti.yaml      ```
    **benchmark_csgo.py**
    ``      CUDA_VISIBLE_DEVICES=1 python benchmark_csgo.py --gt data/preprocessed_data/de_dust2/imgs --pred outputs_eval/exp16_2_gen_conti/test_20260401_214707/gen_imgs/de_dust2 --all --batch_size 8       ``
    ``      CUDA_VISIBLE_DEVICES=1 python benchmark_csgo.py --gt data/preprocessed_data/de_nuke/imgs --pred outputs_eval/exp16_2_gen_conti/test_20260401_214707/gen_imgs/de_nuke --all --batch_size 8       ``
    ``      CUDA_VISIBLE_DEVICES=1 python benchmark_csgo.py --gt data/preprocessed_data/de_ancient/imgs --pred outputs_eval/exp16_2_gen_conti/test_20260401_214707/gen_imgs/de_ancient --all --batch_size 8       ``
    **frames to video**
    ``    python frames_to_video.py --img_dir outputs_eval/exp16_2_gen_conti/test_20260401_214707/gen_imgs/de_dust2 --gt_dir data/preprocessed_data/de_dust2/imgs --output_dir outputs_eval/exp16_2_gen_conti/test_20260401_214707/gen_compared_videos/de_dust2 --max_duration 10        ``
    ``    python frames_to_video.py --img_dir outputs_eval/exp16_2_gen_conti/test_20260401_214707/gen_imgs/de_nuke --gt_dir data/preprocessed_data/de_nuke/imgs --output_dir outputs_eval/exp16_2_gen_conti/test_20260401_214707/gen_compared_videos/de_nuke --max_duration 10        ``
    ``    python frames_to_video.py --img_dir outputs_eval/exp16_2_gen_conti/test_20260401_214707/gen_imgs/de_ancient --gt_dir data/preprocessed_data/de_ancient/imgs --output_dir outputs_eval/exp16_2_gen_conti/test_20260401_214707/gen_compared_videos/de_ancient --max_duration 10        ``

### exp16_3
- fork from exp2
- resume_ckpt_path: "outputs/csgo_1b/exp2/checkpoint-46800/model.safetensors"
- use_external_loc_model: True
- external_loc_use_circular_loss: True
- external_loc_repo_root: "csgosquare"
- external_loc_config_path: "configs_reg_newdata/exp5_2.yaml"
- external_loc_checkpoint_path: "checkpoints_reg_newdata/exp5_2/20251227_091745/current_model.pth"
- img_size: 224
- is_lora: False
- 仅gen数据

- exp16调参
- is_loc_aux_loss: True
- alpha_loc_aux_loss: 10

**train_csgo.py**
```     CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node=1 --master_port=29511 train_csgo.py --csgo_config csgo_configs/exp16_3.yaml --deepspeed deepspeed_scripts/zero0.json --model_name_or_path UniLIP-1B --unilip_factor 10.6 --mllm_hf_path OpenGVLab/InternVL3-1B-hf --version internvl --data_type "mix" --csgo_image_folder data/preprocessed_data --mm_use_im_start_end False --mm_use_im_patch_token False --bf16 True --output_dir outputs/csgo_1b/exp16_3 --num_train_epochs 10 --per_device_train_batch_size 8 --per_device_eval_batch_size 8 --gradient_accumulation_steps 16 --eval_strategy "no" --save_strategy "steps" --save_steps 1000 --save_total_limit 2 --learning_rate 1e-4 --weight_decay 0. --warmup_ratio 0.003 --lr_scheduler_type "cosine_with_min_lr" --model_max_length 1024 --logging_steps 1 --tf32 True --gradient_checkpointing True --dataloader_num_workers 16 --lazy_preprocess True --n_query 256 --n_und_query 0 --report_to wandb --fix_dit False --fix_connect False --fix_llm True       ```
**eval_csgo.py** fst step=4690
``    CUDA_VISIBLE_DEVICES=0 python eval_csgo.py --csgo_config csgo_configs/test/exp16_3_gen.yaml      ``
    **benchmark_csgo.py**
    ``      CUDA_VISIBLE_DEVICES=0 python benchmark_csgo.py --gt data/preprocessed_data/de_dust2/imgs --pred outputs_eval/exp16_3_gen/test_20260401_214730/gen_imgs/de_dust2 --all --batch_size 8       ``
    ``      CUDA_VISIBLE_DEVICES=0 python benchmark_csgo.py --gt data/preprocessed_data/de_nuke/imgs --pred outputs_eval/exp16_3_gen/test_20260401_214730/gen_imgs/de_nuke --all --batch_size 8       ``
    ``      CUDA_VISIBLE_DEVICES=0 python benchmark_csgo.py --gt data/preprocessed_data/de_ancient/imgs --pred outputs_eval/exp16_3_gen/test_20260401_214730/gen_imgs/de_ancient --all --batch_size 8       ``
**continuous gen** fst step=4690
```        CUDA_VISIBLE_DEVICES=1 python eval_csgo.py --csgo_config csgo_configs/test/exp16_3_gen_conti.yaml      ```
    **benchmark_csgo.py**
    ``      CUDA_VISIBLE_DEVICES=1 python benchmark_csgo.py --gt data/preprocessed_data/de_dust2/imgs --pred outputs_eval/exp16_3_gen_conti/test_20260401_214905/gen_imgs/de_dust2 --all --batch_size 8       ``
    ``      CUDA_VISIBLE_DEVICES=1 python benchmark_csgo.py --gt data/preprocessed_data/de_nuke/imgs --pred outputs_eval/exp16_3_gen_conti/test_20260401_214905/gen_imgs/de_nuke --all --batch_size 8       ``
    ``      CUDA_VISIBLE_DEVICES=1 python benchmark_csgo.py --gt data/preprocessed_data/de_ancient/imgs --pred outputs_eval/exp16_3_gen_conti/test_20260401_214905/gen_imgs/de_ancient --all --batch_size 8       ``
    **frames to video**
    ``    python frames_to_video.py --img_dir outputs_eval/exp16_3_gen_conti/test_20260401_214905/gen_imgs/de_dust2 --gt_dir data/preprocessed_data/de_dust2/imgs --output_dir outputs_eval/exp16_3_gen_conti/test_20260401_214905/gen_compared_videos/de_dust2 --max_duration 10        ``
    ``    python frames_to_video.py --img_dir outputs_eval/exp16_3_gen_conti/test_20260401_214905/gen_imgs/de_nuke --gt_dir data/preprocessed_data/de_nuke/imgs --output_dir outputs_eval/exp16_3_gen_conti/test_20260401_214905/gen_compared_videos/de_nuke --max_duration 10        ``
    ``    python frames_to_video.py --img_dir outputs_eval/exp16_3_gen_conti/test_20260401_214905/gen_imgs/de_ancient --gt_dir data/preprocessed_data/de_ancient/imgs --output_dir outputs_eval/exp16_3_gen_conti/test_20260401_214905/gen_compared_videos/de_ancient --max_duration 10        ``

### exp16_4
- fork from exp2
- resume_ckpt_path: "outputs/csgo_1b/exp2/checkpoint-46800/model.safetensors"
- use_external_loc_model: True
- external_loc_use_circular_loss: True
- external_loc_repo_root: "csgosquare"
- external_loc_config_path: "configs_reg_newdata/exp5_2.yaml"
- external_loc_checkpoint_path: "checkpoints_reg_newdata/exp5_2/20251227_091745/current_model.pth"
- img_size: 224
- is_lora: False
- 仅gen数据

- exp16调参
- is_loc_aux_loss: True
- alpha_loc_aux_loss: 1

**train_csgo.py**
```     CUDA_VISIBLE_DEVICES=2 torchrun --nproc_per_node=1 --master_port=29512 train_csgo.py --csgo_config csgo_configs/exp16_4.yaml --deepspeed deepspeed_scripts/zero0.json --model_name_or_path UniLIP-1B --unilip_factor 10.6 --mllm_hf_path OpenGVLab/InternVL3-1B-hf --version internvl --data_type "mix" --csgo_image_folder data/preprocessed_data --mm_use_im_start_end False --mm_use_im_patch_token False --bf16 True --output_dir outputs/csgo_1b/exp16_4 --num_train_epochs 10 --per_device_train_batch_size 8 --per_device_eval_batch_size 8 --gradient_accumulation_steps 16 --eval_strategy "no" --save_strategy "steps" --save_steps 1000 --save_total_limit 2 --learning_rate 1e-4 --weight_decay 0. --warmup_ratio 0.003 --lr_scheduler_type "cosine_with_min_lr" --model_max_length 1024 --logging_steps 1 --tf32 True --gradient_checkpointing True --dataloader_num_workers 16 --lazy_preprocess True --n_query 256 --n_und_query 0 --report_to wandb --fix_dit False --fix_connect False --fix_llm True       ```
**eval_csgo.py** fst step=4690
``    CUDA_VISIBLE_DEVICES=0 python eval_csgo.py --csgo_config csgo_configs/test/exp16_4_gen.yaml      ``
    **benchmark_csgo.py**
    ``      CUDA_VISIBLE_DEVICES=0 python benchmark_csgo.py --gt data/preprocessed_data/de_dust2/imgs --pred outputs_eval/exp16_4_gen/test_20260401_214830/gen_imgs/de_dust2 --all --batch_size 8       ``
    ``      CUDA_VISIBLE_DEVICES=0 python benchmark_csgo.py --gt data/preprocessed_data/de_nuke/imgs --pred outputs_eval/exp16_4_gen/test_20260401_214830/gen_imgs/de_nuke --all --batch_size 8       ``
    ``      CUDA_VISIBLE_DEVICES=0 python benchmark_csgo.py --gt data/preprocessed_data/de_ancient/imgs --pred outputs_eval/exp16_4_gen/test_20260401_214830/gen_imgs/de_ancient --all --batch_size 8       ``
**continuous gen** fst step=4690
```        CUDA_VISIBLE_DEVICES=1 python eval_csgo.py --csgo_config csgo_configs/test/exp16_4_gen_conti.yaml      ```
    **benchmark_csgo.py**
    ``      CUDA_VISIBLE_DEVICES=1 python benchmark_csgo.py --gt data/preprocessed_data/de_dust2/imgs --pred outputs_eval/exp16_4_gen_conti/test_20260401_214827/gen_imgs/de_dust2 --all --batch_size 8       ``
    ``      CUDA_VISIBLE_DEVICES=1 python benchmark_csgo.py --gt data/preprocessed_data/de_nuke/imgs --pred outputs_eval/exp16_4_gen_conti/test_20260401_214827/gen_imgs/de_nuke --all --batch_size 8       ``
    ``      CUDA_VISIBLE_DEVICES=1 python benchmark_csgo.py --gt data/preprocessed_data/de_ancient/imgs --pred outputs_eval/exp16_4_gen_conti/test_20260401_214827/gen_imgs/de_ancient --all --batch_size 8       ``
    **frames to video**
    ``    python frames_to_video.py --img_dir outputs_eval/exp16_4_gen_conti/test_20260401_214827/gen_imgs/de_dust2 --gt_dir data/preprocessed_data/de_dust2/imgs --output_dir outputs_eval/exp16_4_gen_conti/test_20260401_214827/gen_compared_videos/de_dust2 --max_duration 10        ``
    ``    python frames_to_video.py --img_dir outputs_eval/exp16_4_gen_conti/test_20260401_214827/gen_imgs/de_nuke --gt_dir data/preprocessed_data/de_nuke/imgs --output_dir outputs_eval/exp16_4_gen_conti/test_20260401_214827/gen_compared_videos/de_nuke --max_duration 10        ``
    ``    python frames_to_video.py --img_dir outputs_eval/exp16_4_gen_conti/test_20260401_214827/gen_imgs/de_ancient --gt_dir data/preprocessed_data/de_ancient/imgs --output_dir outputs_eval/exp16_4_gen_conti/test_20260401_214827/gen_compared_videos/de_ancient --max_duration 10        ``



## exp14

### exp14_gen
- 约等于 exp2_3 + 无LoRA + 30 epoch
- 仅 gen 数据
- img_size: 448
- is_lora: False
- use_pi05_action_dit: True
- is_action_dit_dense_timestep: True
- is_action_dit_projector: True
- action_dit_projector_lr: 0.0005
- action_dit_lr: 0.0001
- is_aciton_dit_vae_small_init: False

**train_csgo.py**
```     CUDA_VISIBLE_DEVICES=1 MASTER_ADDR=127.0.0.1 MASTER_PORT=29518 RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 python train_csgo.py --csgo_config csgo_configs/exp14_gen.yaml --deepspeed deepspeed_scripts/zero0.json --model_name_or_path UniLIP-1B --unilip_factor 10.6 --mllm_hf_path OpenGVLab/InternVL3-1B-hf --version internvl --data_type "mix" --csgo_image_folder data/preprocessed_data --mm_use_im_start_end False --mm_use_im_patch_token False --bf16 True --output_dir outputs/csgo_1b/exp14_gen --num_train_epochs 30 --per_device_train_batch_size 128 --per_device_eval_batch_size 128 --gradient_accumulation_steps 1 --eval_strategy "no" --save_strategy "steps" --save_steps 4000 --save_total_limit 3 --learning_rate 1e-4 --weight_decay 0. --warmup_ratio 0.003 --lr_scheduler_type "cosine_with_min_lr" --model_max_length 1024 --logging_steps 1 --tf32 True --gradient_checkpointing True --dataloader_num_workers 4 --lazy_preprocess True --n_query 256 --n_und_query 0 --report_to wandb --fix_dit False --fix_connect False --fix_llm True      ```
**eval_csgo.py** fst step=4000 8000对比exp17_2 14040
``    CUDA_VISIBLE_DEVICES=0 python eval_csgo.py --csgo_config csgo_configs/test/exp14_gen.yaml      ``
    **benchmark_csgo.py**
    ``      CUDA_VISIBLE_DEVICES=0 python benchmark_csgo.py --gt data/preprocessed_data/de_dust2/imgs --pred outputs_eval/exp14_gen/test_20260403_230025/gen_imgs/de_dust2 --all --batch_size 8       ``
    ``      CUDA_VISIBLE_DEVICES=0 python benchmark_csgo.py --gt data/preprocessed_data/de_nuke/imgs --pred outputs_eval/exp14_gen/test_20260403_230025/gen_imgs/de_nuke --all --batch_size 8       ``
    ``      CUDA_VISIBLE_DEVICES=0 python benchmark_csgo.py --gt data/preprocessed_data/de_ancient/imgs --pred outputs_eval/exp14_gen/test_20260403_230025/gen_imgs/de_ancient --all --batch_size 8       ``
**continuous gen** fst step=4000 8000对比exp17_2 14040
```   CUDA_VISIBLE_DEVICES=1 python eval_csgo.py --csgo_config csgo_configs/test/exp14_gen_conti.yaml      ```
    **benchmark_csgo.py**
    ``      CUDA_VISIBLE_DEVICES=1 python benchmark_csgo.py --gt data/preprocessed_data/de_dust2/imgs --pred outputs_eval/exp14_gen_conti/test_20260403_230033/gen_imgs/de_dust2 --all --batch_size 8       ``
    ``      CUDA_VISIBLE_DEVICES=1 python benchmark_csgo.py --gt data/preprocessed_data/de_nuke/imgs --pred outputs_eval/exp14_gen_conti/test_20260403_230033/gen_imgs/de_nuke --all --batch_size 8       ``
    ``      CUDA_VISIBLE_DEVICES=1 python benchmark_csgo.py --gt data/preprocessed_data/de_ancient/imgs --pred outputs_eval/exp14_gen_conti/test_20260403_230033/gen_imgs/de_ancient --all --batch_size 8       ``
    **frames to video** 14040：test_20260403_230033    8000：test_20260404_141243
    ``    python frames_to_video.py --img_dir outputs_eval/exp14_gen_conti/test_20260404_141243/gen_imgs/de_dust2 --gt_dir data/preprocessed_data/de_dust2/imgs --output_dir outputs_eval/exp14_gen_conti/test_20260404_141243/gen_compared_videos/de_dust2 --max_duration 10        ``
    ``    python frames_to_video.py --img_dir outputs_eval/exp14_gen_conti/test_20260404_141243/gen_imgs/de_nuke --gt_dir data/preprocessed_data/de_nuke/imgs --output_dir outputs_eval/exp14_gen_conti/test_20260404_141243/gen_compared_videos/de_nuke --max_duration 10        ``
    ``    python frames_to_video.py --img_dir outputs_eval/exp14_gen_conti/test_20260404_141243/gen_imgs/de_ancient --gt_dir data/preprocessed_data/de_ancient/imgs --output_dir outputs_eval/exp14_gen_conti/test_20260404_141243/gen_compared_videos/de_ancient --max_duration 10        ``

### exp14_dust2_gen
- exp14_gen

- dust2
- epoch=50

**train_csgo.py**
```     CUDA_VISIBLE_DEVICES=4 MASTER_ADDR=127.0.0.1 MASTER_PORT=29518 RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 python train_csgo.py --csgo_config csgo_configs/exp14_dust2_gen.yaml --deepspeed deepspeed_scripts/zero0.json --model_name_or_path UniLIP-1B --unilip_factor 10.6 --mllm_hf_path OpenGVLab/InternVL3-1B-hf --version internvl --data_type "mix" --csgo_image_folder data/preprocessed_data --mm_use_im_start_end False --mm_use_im_patch_token False --bf16 True --output_dir outputs/csgo_1b/exp14_dust2_gen --num_train_epochs 50 --per_device_train_batch_size 128 --per_device_eval_batch_size 128 --gradient_accumulation_steps 1 --eval_strategy "no" --save_strategy "steps" --save_steps 4000 --save_total_limit 4 --learning_rate 1e-4 --weight_decay 0. --warmup_ratio 0.003 --lr_scheduler_type "cosine_with_min_lr" --model_max_length 1024 --logging_steps 1 --tf32 True --gradient_checkpointing True --dataloader_num_workers 4 --lazy_preprocess True --n_query 256 --n_und_query 0 --report_to wandb --fix_dit False --fix_connect False --fix_llm True      ```
**eval_csgo.py** step=7800 4000
``    CUDA_VISIBLE_DEVICES=1 python eval_csgo.py --csgo_config csgo_configs/test/exp14_dust2_gen.yaml      ``
    **benchmark_csgo.py**
    ``      CUDA_VISIBLE_DEVICES=1 python benchmark_csgo.py --gt data/preprocessed_data/de_dust2/imgs --pred outputs_eval/exp14_dust2_gen/test_20260417_012300/gen_imgs/de_dust2 --all --batch_size 8       ``
    **benchmark_csgo_v1.py after eval_csgo.py** step=7800 test_20260417_012300 4000 test_20260414_001035
    ``      CUDA_VISIBLE_DEVICES=0 python benchmark_csgo_v1.py --gt data/preprocessed_data/de_dust2/imgs --pred outputs_eval/exp14_dust2_gen/test_20260414_001035/gen_imgs/de_dust2 --batch_size 8 --device cuda --paired_size 448 --data_dir data/preprocessed_data --map_name de_dust2       ``
**continuous gen** step=7800 4000
```   CUDA_VISIBLE_DEVICES=1 python eval_csgo.py --csgo_config csgo_configs/test/exp14_dust2_gen_conti.yaml      ```
    **benchmark_csgo.py**
    ``      CUDA_VISIBLE_DEVICES=1 python benchmark_csgo.py --gt data/preprocessed_data/de_dust2/imgs --pred outputs_eval/exp14_dust2_gen_conti/test_20260417_012259/gen_imgs/de_dust2 --all --batch_size 8       ``
    **benchmark_csgo_v1_conti.py after eval_csgo.py** step=7800 test_20260417_012259 4000 test_20260414_001039
    ``      CUDA_VISIBLE_DEVICES=0 python benchmark_csgo_v1_conti.py --gt data/preprocessed_data/de_dust2/imgs --pred outputs_eval/exp14_dust2_gen_conti/test_20260414_001039/gen_imgs/de_dust2 --batch_size 8 --device cuda --paired_size 448 --data_dir data/preprocessed_data --map_name de_dust2 --frame_diff_threshold 2 --min_track_len 4 --clip_length 16 --clip_stride 16 --fvd_size 224       ``


### exp14_1_gen
- exp14_gen

- epoch=100

**train_csgo.py**
```     CUDA_VISIBLE_DEVICES=0 MASTER_ADDR=127.0.0.1 MASTER_PORT=29518 RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 python train_csgo.py --csgo_config csgo_configs/exp14_1_gen.yaml --deepspeed deepspeed_scripts/zero0.json --model_name_or_path UniLIP-1B --unilip_factor 10.6 --mllm_hf_path OpenGVLab/InternVL3-1B-hf --version internvl --data_type "mix" --csgo_image_folder data/preprocessed_data --mm_use_im_start_end False --mm_use_im_patch_token False --bf16 True --output_dir outputs/csgo_1b/exp14_1_gen --num_train_epochs 100 --per_device_train_batch_size 128 --per_device_eval_batch_size 128 --gradient_accumulation_steps 1 --eval_strategy "no" --save_strategy "steps" --save_steps 4000 --save_total_limit 3 --learning_rate 1e-4 --weight_decay 0. --warmup_ratio 0.003 --lr_scheduler_type "cosine_with_min_lr" --model_max_length 1024 --logging_steps 1 --tf32 True --gradient_checkpointing True --dataloader_num_workers 4 --lazy_preprocess True --n_query 256 --n_und_query 0 --report_to wandb --fix_dit False --fix_connect False --fix_llm True      ```
**eval_csgo.py** fst step=4000 8000 12000 20000 46800 36000 28000
``    CUDA_VISIBLE_DEVICES=0 python eval_csgo.py --csgo_config csgo_configs/test/exp14_1_gen.yaml      ``
    **benchmark_csgo.py**
    ``      CUDA_VISIBLE_DEVICES=1 python benchmark_csgo.py --gt data/preprocessed_data/de_dust2/imgs --pred outputs_eval/exp14_1_gen/test_20260415_135216/gen_imgs/de_dust2 --all --batch_size 8       ``
    ``      CUDA_VISIBLE_DEVICES=1 python benchmark_csgo.py --gt data/preprocessed_data/de_nuke/imgs --pred outputs_eval/exp14_1_gen/test_20260415_135216/gen_imgs/de_nuke --all --batch_size 8       ``
    ``      CUDA_VISIBLE_DEVICES=1 python benchmark_csgo.py --gt data/preprocessed_data/de_ancient/imgs --pred outputs_eval/exp14_1_gen/test_20260415_135216/gen_imgs/de_ancient --all --batch_size 8       ``
**continuous gen** fst step=4000 8000 12000 20000 46800 36000 28000
```   CUDA_VISIBLE_DEVICES=1 python eval_csgo.py --csgo_config csgo_configs/test/exp14_1_gen_conti.yaml      ```
    **benchmark_csgo.py**
    ``      CUDA_VISIBLE_DEVICES=1 python benchmark_csgo.py --gt data/preprocessed_data/de_dust2/imgs --pred outputs_eval/exp14_1_gen_conti/test_20260415_165319/gen_imgs/de_dust2 --all --batch_size 8       ``
    ``      CUDA_VISIBLE_DEVICES=1 python benchmark_csgo.py --gt data/preprocessed_data/de_nuke/imgs --pred outputs_eval/exp14_1_gen_conti/test_20260415_165319/gen_imgs/de_nuke --all --batch_size 8       ``
    ``      CUDA_VISIBLE_DEVICES=1 python benchmark_csgo.py --gt data/preprocessed_data/de_ancient/imgs --pred outputs_eval/exp14_1_gen_conti/test_20260415_165319/gen_imgs/de_ancient --all --batch_size 8       ``
    **frames to video**
    ``    python frames_to_video.py --img_dir outputs_eval/exp14_1_gen_conti/test_20260411_153404/gen_imgs/de_dust2 --gt_dir data/preprocessed_data/de_dust2/imgs --output_dir outputs_eval/exp14_1_gen_conti/test_20260411_153404/gen_compared_videos/de_dust2 --max_duration 10        ``
    ``    python frames_to_video.py --img_dir outputs_eval/exp14_1_gen_conti/test_20260411_153404/gen_imgs/de_nuke --gt_dir data/preprocessed_data/de_nuke/imgs --output_dir outputs_eval/exp14_1_gen_conti/test_20260411_153404/gen_compared_videos/de_nuke --max_duration 10        ``
    ``    python frames_to_video.py --img_dir outputs_eval/exp14_1_gen_conti/test_20260411_153404/gen_imgs/de_ancient --gt_dir data/preprocessed_data/de_ancient/imgs --output_dir outputs_eval/exp14_1_gen_conti/test_20260411_153404/gen_compared_videos/de_ancient --max_duration 10        ``

### exp14_1_dust2_gen
- exp14_1_gen

- dust2

**train_csgo.py**
```     CUDA_VISIBLE_DEVICES=4 MASTER_ADDR=127.0.0.1 MASTER_PORT=29518 RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 python train_csgo.py --csgo_config csgo_configs/exp14_1_dust2_gen.yaml --deepspeed deepspeed_scripts/zero0.json --model_name_or_path UniLIP-1B --unilip_factor 10.6 --mllm_hf_path OpenGVLab/InternVL3-1B-hf --version internvl --data_type "mix" --csgo_image_folder data/preprocessed_data --mm_use_im_start_end False --mm_use_im_patch_token False --bf16 True --output_dir outputs/csgo_1b/exp14_1_dust2_gen --num_train_epochs 100 --per_device_train_batch_size 128 --per_device_eval_batch_size 128 --gradient_accumulation_steps 1 --eval_strategy "no" --save_strategy "steps" --save_steps 4000 --save_total_limit 4 --learning_rate 1e-4 --weight_decay 0. --warmup_ratio 0.003 --lr_scheduler_type "cosine_with_min_lr" --model_max_length 1024 --logging_steps 1 --tf32 True --gradient_checkpointing True --dataloader_num_workers 4 --lazy_preprocess True --n_query 256 --n_und_query 0 --report_to wandb --fix_dit False --fix_connect False --fix_llm True      ```
**eval_csgo.py** step=15600
``    CUDA_VISIBLE_DEVICES=0 python eval_csgo.py --csgo_config csgo_configs/test/exp14_1_dust2_gen.yaml      ``
  **benchmark_csgo.py**
    ``      CUDA_VISIBLE_DEVICES=1 python benchmark_csgo.py --gt data/preprocessed_data/de_dust2/imgs --pred outputs_eval/exp14_1_dust2_gen/test_20260414_183840/gen_imgs/de_dust2 --all --batch_size 8       ``
    **benchmark_csgo_v1.py after eval_csgo.py** step=15600
    ``      CUDA_VISIBLE_DEVICES=0 python benchmark_csgo_v1.py --gt data/preprocessed_data/de_dust2/imgs --pred outputs_eval/exp14_1_dust2_gen/test_20260414_183840/gen_imgs/de_dust2 --batch_size 8 --device cuda --paired_size 448 --data_dir data/preprocessed_data --map_name de_dust2       ``
**continuous gen** step=15600
```   CUDA_VISIBLE_DEVICES=1 python eval_csgo.py --csgo_config csgo_configs/test/exp14_1_dust2_gen_conti.yaml      ```
    **benchmark_csgo.py**
    ``      CUDA_VISIBLE_DEVICES=0 python benchmark_csgo.py --gt data/preprocessed_data/de_dust2/imgs --pred outputs_eval/exp14_1_dust2_gen_conti/test_20260414_183844/gen_imgs/de_dust2 --all --batch_size 8       ``
    **benchmark_csgo_v1_conti.py after eval_csgo.py** step=15600
    ``      CUDA_VISIBLE_DEVICES=0 python benchmark_csgo_v1_conti.py --gt data/preprocessed_data/de_dust2/imgs --pred outputs_eval/exp14_1_dust2_gen_conti/test_20260414_183844/gen_imgs/de_dust2 --batch_size 8 --device cuda --paired_size 448 --data_dir data/preprocessed_data --map_name de_dust2 --frame_diff_threshold 2 --min_track_len 4 --clip_length 16 --clip_stride 16 --fvd_size 224       ``

### exp14_loc
- 约等于 exp4_12_2 + 无lora + 30 epoch
- 仅 loc 数据
- img_size: 448
- is_lora: False
- use_pi05_action_dit: True
- is_action_dit_dense_timestep: True
- is_action_dit_projector: True
- action_dit_projector_lr: 0.0005
- action_dit_lr: 0.0001
- is_aciton_dit_vae_small_init: False

**train_csgo.py**
```     CUDA_VISIBLE_DEVICES=2 MASTER_ADDR=127.0.0.1 MASTER_PORT=29519 RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 python train_csgo.py --csgo_config csgo_configs/exp14_loc.yaml --deepspeed deepspeed_scripts/zero0.json --model_name_or_path UniLIP-1B --unilip_factor 10.6 --mllm_hf_path OpenGVLab/InternVL3-1B-hf --version internvl --data_type "mix" --csgo_image_folder data/preprocessed_data --mm_use_im_start_end False --mm_use_im_patch_token False --bf16 True --output_dir outputs/csgo_1b/exp14_loc --num_train_epochs 30 --per_device_train_batch_size 128 --per_device_eval_batch_size 128 --gradient_accumulation_steps 1 --eval_strategy "no" --save_strategy "steps" --save_steps 4000 --save_total_limit 3 --learning_rate 1e-4 --weight_decay 0. --warmup_ratio 0.003 --lr_scheduler_type "cosine_with_min_lr" --model_max_length 1024 --logging_steps 1 --tf32 True --gradient_checkpointing True --dataloader_num_workers 4 --lazy_preprocess True --n_query 256 --n_und_query 0 --report_to wandb --fix_dit True --fix_connect True --fix_llm True --fix_vit True     ```
**eval_csgo_loc.py** fst: step=4000 14040 8000对比exp17_2
```    CUDA_VISIBLE_DEVICES=1 python eval_csgo_loc.py --csgo_config csgo_configs/test/exp14_loc.yaml      ```

### exp14_dust2_loc
- exp14_loc

- dust2
- epoch=50
**train_csgo.py**
```     CUDA_VISIBLE_DEVICES=4 MASTER_ADDR=127.0.0.1 MASTER_PORT=29519 RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 python train_csgo.py --csgo_config csgo_configs/exp14_dust2_loc.yaml --deepspeed deepspeed_scripts/zero0.json --model_name_or_path UniLIP-1B --unilip_factor 10.6 --mllm_hf_path OpenGVLab/InternVL3-1B-hf --version internvl --data_type "mix" --csgo_image_folder data/preprocessed_data --mm_use_im_start_end False --mm_use_im_patch_token False --bf16 True --output_dir outputs/csgo_1b/exp14_dust2_loc --num_train_epochs 50 --per_device_train_batch_size 128 --per_device_eval_batch_size 128 --gradient_accumulation_steps 1 --eval_strategy "no" --save_strategy "steps" --save_steps 4000 --save_total_limit 4 --learning_rate 1e-4 --weight_decay 0. --warmup_ratio 0.003 --lr_scheduler_type "cosine_with_min_lr" --model_max_length 1024 --logging_steps 1 --tf32 True --gradient_checkpointing True --dataloader_num_workers 4 --lazy_preprocess True --n_query 256 --n_und_query 0 --report_to wandb --fix_dit True --fix_connect True --fix_llm True --fix_vit True     ```
**eval_csgo_loc.py** step=7800 4000
```    CUDA_VISIBLE_DEVICES=1 python eval_csgo_loc.py --csgo_config csgo_configs/test/exp14_dust2_loc.yaml      ```


## exp15
- exp16-style training objective
- loc head from exp4_12_2 pretrain_path: "outputs/csgo_1b/exp4_12_2/checkpoint-23450/model.safetensors"
- gen head from exp2 resume_ckpt_path: "outputs/csgo_1b/exp2/checkpoint-46800/model.safetensors"
- is_multi_task_balanced: False
- task_mix_ratio: 0.0; only gen samples
- alpha_loc_loss: 0.0
- is_loc_aux_loss: True
- alpha_loc_aux_loss: 100
- use_pi05_action_dit: True
- is_action_dit_dense_timestep: True
- is_action_dit_projector: True
- action_dit_projector_lr: 0.0005
- is_aciton_dit_vae_small_init: False
- img_size: 448
- is_lora: False

**train_csgo.py**
```     CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=29512 train_csgo.py --csgo_config csgo_configs/exp15.yaml --deepspeed deepspeed_scripts/zero0.json --model_name_or_path UniLIP-1B --unilip_factor 10.6 --mllm_hf_path OpenGVLab/InternVL3-1B-hf --version internvl --data_type "mix" --csgo_image_folder data/preprocessed_data --mm_use_im_start_end False --mm_use_im_patch_token False --bf16 True --output_dir outputs/csgo_1b/exp15 --num_train_epochs 10 --per_device_train_batch_size 8 --per_device_eval_batch_size 8 --gradient_accumulation_steps 16 --eval_strategy "no" --save_strategy "steps" --save_steps 2000 --save_total_limit 3 --learning_rate 1e-4 --weight_decay 0. --warmup_ratio 0.003 --lr_scheduler_type "cosine_with_min_lr" --model_max_length 1024 --logging_steps 1 --tf32 True --gradient_checkpointing True --dataloader_num_workers 4 --lazy_preprocess True --n_query 256 --n_und_query 0 --report_to wandb --fix_dit False --fix_connect False --fix_llm True --pretrain_path outputs/csgo_1b/exp4_12_2/checkpoint-23400/model.safetensors       ```
**eval_csgo.py** fst: step=4690
```    CUDA_VISIBLE_DEVICES=0 python eval_csgo.py --csgo_config csgo_configs/test/exp15_gen.yaml      ```
    **benchmark_csgo.py**
    ``      CUDA_VISIBLE_DEVICES=1 python benchmark_csgo.py --gt data/preprocessed_data/de_dust2/imgs --pred outputs_eval/exp15_gen/test_20260407_182115/gen_imgs/de_dust2 --all --batch_size 8       ``
    ``      CUDA_VISIBLE_DEVICES=1 python benchmark_csgo.py --gt data/preprocessed_data/de_nuke/imgs --pred outputs_eval/exp15_gen/test_20260407_182115/gen_imgs/de_nuke --all --batch_size 8       ``
    ``      CUDA_VISIBLE_DEVICES=1 python benchmark_csgo.py --gt data/preprocessed_data/de_ancient/imgs --pred outputs_eval/exp15_gen/test_20260407_182115/gen_imgs/de_ancient --all --batch_size 8       ``
**continuous gen**  fst: step=4690
```    CUDA_VISIBLE_DEVICES=0 python eval_csgo.py --csgo_config csgo_configs/test/exp15_gen_conti.yaml      ```
    **benchmark_csgo.py**
    ``      CUDA_VISIBLE_DEVICES=1 python benchmark_csgo.py --gt data/preprocessed_data/de_dust2/imgs --pred outputs_eval/exp15_gen_conti/test_20260407_182118/gen_imgs/de_dust2 --all --batch_size 8       ``
    ``      CUDA_VISIBLE_DEVICES=1 python benchmark_csgo.py --gt data/preprocessed_data/de_nuke/imgs --pred outputs_eval/exp15_gen_conti/test_20260407_182118/gen_imgs/de_nuke --all --batch_size 8       ``
    ``      CUDA_VISIBLE_DEVICES=0 python benchmark_csgo.py --gt data/preprocessed_data/de_ancient/imgs --pred outputs_eval/exp15_gen_conti/test_20260407_182118/gen_imgs/de_ancient --all --batch_size 8       ``
    **frames to video**
    ``    python frames_to_video.py --img_dir outputs_eval/exp15_gen_conti/test_20260407_182118/gen_imgs/de_dust2 --gt_dir data/preprocessed_data/de_dust2/imgs --output_dir outputs_eval/exp15_gen_conti/test_20260407_182118/gen_compared_videos/de_dust2 --max_duration 10        ``
    ``    python frames_to_video.py --img_dir outputs_eval/exp15_gen_conti/test_20260407_182118/gen_imgs/de_nuke --gt_dir data/preprocessed_data/de_nuke/imgs --output_dir outputs_eval/exp15_gen_conti/test_20260407_182118/gen_compared_videos/de_nuke --max_duration 10        ``
    ``    python frames_to_video.py --img_dir outputs_eval/exp15_gen_conti/test_20260407_182118/gen_imgs/de_ancient --gt_dir data/preprocessed_data/de_ancient/imgs --output_dir outputs_eval/exp15_gen_conti/test_20260407_182118/gen_compared_videos/de_ancient --max_duration 10        ``


### exp15_1
- exp11-style joint loc/gen training
- loc head from exp4_12_2 pretrain_path: "outputs/csgo_1b/exp4_12_2/checkpoint-23450/model.safetensors"
- gen head from exp2 resume_ckpt_path: "outputs/csgo_1b/exp2/checkpoint-46800/model.safetensors"
- is_multi_task_balanced: True
- alpha_loc_loss: 2
- is_loc_aux_loss: True
- alpha_loc_aux_loss: 100
- use_pi05_action_dit: True
- is_action_dit_dense_timestep: True
- is_action_dit_projector: True
- action_dit_projector_lr: 0.0005
- is_aciton_dit_vae_small_init: False
- img_size: 448
- is_lora: False

**train_csgo.py**
```     CUDA_VISIBLE_DEVICES=4 torchrun --nproc_per_node=1 --master_port=29513 train_csgo.py --csgo_config csgo_configs/exp15_1.yaml --deepspeed deepspeed_scripts/zero0.json --model_name_or_path UniLIP-1B --unilip_factor 10.6 --mllm_hf_path OpenGVLab/InternVL3-1B-hf --version internvl --data_type "mix" --csgo_image_folder data/preprocessed_data --mm_use_im_start_end False --mm_use_im_patch_token False --bf16 True --output_dir outputs/csgo_1b/exp15_1 --num_train_epochs 10 --per_device_train_batch_size 8 --per_device_eval_batch_size 8 --gradient_accumulation_steps 16 --eval_strategy "no" --save_strategy "steps" --save_steps 2000 --save_total_limit 3 --learning_rate 1e-4 --weight_decay 0. --warmup_ratio 0.003 --lr_scheduler_type "cosine_with_min_lr" --model_max_length 1024 --logging_steps 1 --tf32 True --gradient_checkpointing True --dataloader_num_workers 4 --lazy_preprocess True --n_query 256 --n_und_query 0 --report_to wandb --fix_dit False --fix_connect False --fix_llm True --pretrain_path outputs/csgo_1b/exp4_12_2/checkpoint-23400/model.safetensors       ```
**eval_csgo_loc.py** fst: step=4690
```    CUDA_VISIBLE_DEVICES=0 python eval_csgo_loc.py --csgo_config csgo_configs/test/exp15_1_loc.yaml      ```
**eval_csgo.py** fst: step=4690
```    CUDA_VISIBLE_DEVICES=0 python eval_csgo.py --csgo_config csgo_configs/test/exp15_1_gen.yaml      ```
    **benchmark_csgo.py**
    ``      CUDA_VISIBLE_DEVICES=1 python benchmark_csgo.py --gt data/preprocessed_data/de_dust2/imgs --pred outputs_eval/exp15_1_gen/test_20260408_164146/gen_imgs/de_dust2 --all --batch_size 8       ``
    ``      CUDA_VISIBLE_DEVICES=1 python benchmark_csgo.py --gt data/preprocessed_data/de_nuke/imgs --pred outputs_eval/exp15_1_gen/test_20260408_164146/gen_imgs/de_nuke --all --batch_size 8       ``
    ``      CUDA_VISIBLE_DEVICES=1 python benchmark_csgo.py --gt data/preprocessed_data/de_ancient/imgs --pred outputs_eval/exp15_1_gen/test_20260408_164146/gen_imgs/de_ancient --all --batch_size 8       ``
**continuous gen**  fst: step=4690
```    CUDA_VISIBLE_DEVICES=1 python eval_csgo.py --csgo_config csgo_configs/test/exp15_1_gen_conti.yaml      ```
    **benchmark_csgo.py**
    ``      CUDA_VISIBLE_DEVICES=0 python benchmark_csgo.py --gt data/preprocessed_data/de_dust2/imgs --pred outputs_eval/exp15_1_gen_conti/test_20260408_164151/gen_imgs/de_dust2 --all --batch_size 8       ``
    ``      CUDA_VISIBLE_DEVICES=0 python benchmark_csgo.py --gt data/preprocessed_data/de_nuke/imgs --pred outputs_eval/exp15_1_gen_conti/test_20260408_164151/gen_imgs/de_nuke --all --batch_size 8       ``
    ``      CUDA_VISIBLE_DEVICES=0 python benchmark_csgo.py --gt data/preprocessed_data/de_ancient/imgs --pred outputs_eval/exp15_1_gen_conti/test_20260408_164151/gen_imgs/de_ancient --all --batch_size 8       ``
    **frames to video**
    ``    python frames_to_video.py --img_dir outputs_eval/exp15_1_gen_conti/test_20260408_164151/gen_imgs/de_dust2 --gt_dir data/preprocessed_data/de_dust2/imgs --output_dir outputs_eval/exp15_1_gen_conti/test_20260408_164151/gen_compared_videos/de_dust2 --max_duration 10        ``
    ``    python frames_to_video.py --img_dir outputs_eval/exp15_1_gen_conti/test_20260408_164151/gen_imgs/de_nuke --gt_dir data/preprocessed_data/de_nuke/imgs --output_dir outputs_eval/exp15_1_gen_conti/test_20260408_164151/gen_compared_videos/de_nuke --max_duration 10        ``
    ``    python frames_to_video.py --img_dir outputs_eval/exp15_1_gen_conti/test_20260408_164151/gen_imgs/de_ancient --gt_dir data/preprocessed_data/de_ancient/imgs --output_dir outputs_eval/exp15_1_gen_conti/test_20260408_164151/gen_compared_videos/de_ancient --max_duration 10        ``

### exp15_2
- dynamic alpha of exp15
- loc head from exp4_12_2 pretrain_path: "outputs/csgo_1b/exp4_12_2/checkpoint-23450/model.safetensors"
- gen head from exp2 resume_ckpt_path: "outputs/csgo_1b/exp2/checkpoint-46800/model.safetensors"
- is_multi_task_balanced: False
- task_mix_ratio: 0.0; only gen samples
- alpha_loc_loss: 0.0
- is_loc_aux_loss: True
- alpha_loc_aux schedule steps: [0, 500, 2000, 4500]
- alpha_loc_aux schedule values: [0.0, 10.0, 50.0, 100.0]
- use_pi05_action_dit: True
- is_action_dit_dense_timestep: True
- is_action_dit_projector: True
- action_dit_projector_lr: 0.0005
- is_aciton_dit_vae_small_init: False
- img_size: 448
- is_lora: False

**train_csgo.py**
```     CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=29515 train_csgo.py --csgo_config csgo_configs/exp15_2.yaml --deepspeed deepspeed_scripts/zero0.json --model_name_or_path UniLIP-1B --unilip_factor 10.6 --mllm_hf_path OpenGVLab/InternVL3-1B-hf --version internvl --data_type "mix" --csgo_image_folder data/preprocessed_data --mm_use_im_start_end False --mm_use_im_patch_token False --bf16 True --output_dir outputs/csgo_1b/exp15_2 --num_train_epochs 10 --per_device_train_batch_size 8 --per_device_eval_batch_size 8 --gradient_accumulation_steps 16 --eval_strategy "no" --save_strategy "steps" --save_steps 2000 --save_total_limit 3 --learning_rate 1e-4 --weight_decay 0. --warmup_ratio 0.003 --lr_scheduler_type "cosine_with_min_lr" --model_max_length 1024 --logging_steps 1 --tf32 True --gradient_checkpointing True --dataloader_num_workers 4 --lazy_preprocess True --n_query 256 --n_und_query 0 --report_to wandb --fix_dit False --fix_connect False --fix_llm True --pretrain_path outputs/csgo_1b/exp4_12_2/checkpoint-23400/model.safetensors       ```
**eval_csgo.py** fst: step=2000 4690
```    CUDA_VISIBLE_DEVICES=1 python eval_csgo.py --csgo_config csgo_configs/test/exp15_2_gen.yaml      ```
    **benchmark_csgo.py**
    ``      CUDA_VISIBLE_DEVICES=1 python benchmark_csgo.py --gt data/preprocessed_data/de_dust2/imgs --pred outputs_eval/exp15_2_gen/test_20260404_142249/gen_imgs/de_dust2 --all --batch_size 8       ``
    ``      CUDA_VISIBLE_DEVICES=1 python benchmark_csgo.py --gt data/preprocessed_data/de_nuke/imgs --pred outputs_eval/exp15_2_gen/test_20260404_142249/gen_imgs/de_nuke --all --batch_size 8       ``
    ``      CUDA_VISIBLE_DEVICES=1 python benchmark_csgo.py --gt data/preprocessed_data/de_ancient/imgs --pred outputs_eval/exp15_2_gen/test_20260404_142249/gen_imgs/de_ancient --all --batch_size 8       ``
**continuous gen**  fst: step=2000 4690
```    CUDA_VISIBLE_DEVICES=0 python eval_csgo.py --csgo_config csgo_configs/test/exp15_2_gen_conti.yaml      ```
    **benchmark_csgo.py**
    ``      CUDA_VISIBLE_DEVICES=1 python benchmark_csgo.py --gt data/preprocessed_data/de_dust2/imgs --pred outputs_eval/exp15_2_gen_conti/test_20260404_142259/gen_imgs/de_dust2 --all --batch_size 8       ``
    ``      CUDA_VISIBLE_DEVICES=0 python benchmark_csgo.py --gt data/preprocessed_data/de_nuke/imgs --pred outputs_eval/exp15_2_gen_conti/test_20260404_142259/gen_imgs/de_nuke --all --batch_size 8       ``
    ``      CUDA_VISIBLE_DEVICES=0 python benchmark_csgo.py --gt data/preprocessed_data/de_ancient/imgs --pred outputs_eval/exp15_2_gen_conti/test_20260404_142259/gen_imgs/de_ancient --all --batch_size 8       ``
    **frames to video**
    ``    python frames_to_video.py --img_dir outputs_eval/exp15_2_gen_conti/test_20260404_142259/gen_imgs/de_dust2 --gt_dir data/preprocessed_data/de_dust2/imgs --output_dir outputs_eval/exp15_2_gen_conti/test_20260404_142259/gen_compared_videos/de_dust2 --max_duration 10        ``
    ``    python frames_to_video.py --img_dir outputs_eval/exp15_2_gen_conti/test_20260404_142259/gen_imgs/de_nuke --gt_dir data/preprocessed_data/de_nuke/imgs --output_dir outputs_eval/exp15_2_gen_conti/test_20260404_142259/gen_compared_videos/de_nuke --max_duration 10        ``
    ``    python frames_to_video.py --img_dir outputs_eval/exp15_2_gen_conti/test_20260404_142259/gen_imgs/de_ancient --gt_dir data/preprocessed_data/de_ancient/imgs --output_dir outputs_eval/exp15_2_gen_conti/test_20260404_142259/gen_compared_videos/de_ancient --max_duration 10        ``

### exp15_3
- dynamic alpha of exp15_1
- loc head from exp4_12_2 pretrain_path: "outputs/csgo_1b/exp4_12_2/checkpoint-23450/model.safetensors"
- gen head from exp2 resume_ckpt_path: "outputs/csgo_1b/exp2/checkpoint-46800/model.safetensors"
- is_multi_task_balanced: True
- alpha_loc_loss: 2
- is_loc_aux_loss: True
- alpha_loc_aux schedule steps: [0, 500, 2000, 4500]
- alpha_loc_aux schedule values: [0.0, 20.0, 100.0, 200.0]
- use_pi05_action_dit: True
- is_action_dit_dense_timestep: True
- is_action_dit_projector: True
- action_dit_projector_lr: 0.0005
- is_aciton_dit_vae_small_init: False
- img_size: 448
- is_lora: False

**train_csgo.py**
```     CUDA_VISIBLE_DEVICES=4 torchrun --nproc_per_node=1 --master_port=29516 train_csgo.py --csgo_config csgo_configs/exp15_3.yaml --deepspeed deepspeed_scripts/zero0.json --model_name_or_path UniLIP-1B --unilip_factor 10.6 --mllm_hf_path OpenGVLab/InternVL3-1B-hf --version internvl --data_type "mix" --csgo_image_folder data/preprocessed_data --mm_use_im_start_end False --mm_use_im_patch_token False --bf16 True --output_dir outputs/csgo_1b/exp15_3 --num_train_epochs 10 --per_device_train_batch_size 8 --per_device_eval_batch_size 8 --gradient_accumulation_steps 16 --eval_strategy "no" --save_strategy "steps" --save_steps 2000 --save_total_limit 3 --learning_rate 1e-4 --weight_decay 0. --warmup_ratio 0.003 --lr_scheduler_type "cosine_with_min_lr" --model_max_length 1024 --logging_steps 1 --tf32 True --gradient_checkpointing True --dataloader_num_workers 4 --lazy_preprocess True --n_query 256 --n_und_query 0 --report_to wandb --fix_dit False --fix_connect False --fix_llm True --pretrain_path outputs/csgo_1b/exp4_12_2/checkpoint-23400/model.safetensors       ```
**eval_csgo_loc.py** fst: step=4690
```    CUDA_VISIBLE_DEVICES=1 python eval_csgo_loc.py --csgo_config csgo_configs/test/exp15_3_loc.yaml      ```
**eval_csgo.py** fst: step=4690
```    CUDA_VISIBLE_DEVICES=1 python eval_csgo.py --csgo_config csgo_configs/test/exp15_3_gen.yaml      ```
    **benchmark_csgo.py**
    ``      CUDA_VISIBLE_DEVICES=1 python benchmark_csgo.py --gt data/preprocessed_data/de_dust2/imgs --pred outputs_eval/exp15_3_gen/test_20260407_182233/gen_imgs/de_dust2 --all --batch_size 8       ``
    ``      CUDA_VISIBLE_DEVICES=1 python benchmark_csgo.py --gt data/preprocessed_data/de_nuke/imgs --pred outputs_eval/exp15_3_gen/test_20260407_182233/gen_imgs/de_nuke --all --batch_size 8       ``
    ``      CUDA_VISIBLE_DEVICES=1 python benchmark_csgo.py --gt data/preprocessed_data/de_ancient/imgs --pred outputs_eval/exp15_3_gen/test_20260407_182233/gen_imgs/de_ancient --all --batch_size 8       ``
**continuous gen**  fst: step=4690
```    CUDA_VISIBLE_DEVICES=1 python eval_csgo.py --csgo_config csgo_configs/test/exp15_3_gen_conti.yaml      ```
    **benchmark_csgo.py**
    ``      CUDA_VISIBLE_DEVICES=0 python benchmark_csgo.py --gt data/preprocessed_data/de_dust2/imgs --pred outputs_eval/exp15_3_gen_conti/test_20260407_182238/gen_imgs/de_dust2 --all --batch_size 8       ``
    ``      CUDA_VISIBLE_DEVICES=0 python benchmark_csgo.py --gt data/preprocessed_data/de_nuke/imgs --pred outputs_eval/exp15_3_gen_conti/test_20260407_182238/gen_imgs/de_nuke --all --batch_size 8       ``
    ``      CUDA_VISIBLE_DEVICES=0 python benchmark_csgo.py --gt data/preprocessed_data/de_ancient/imgs --pred outputs_eval/exp15_3_gen_conti/test_20260407_182238/gen_imgs/de_ancient --all --batch_size 8       ``
    **frames to video**
    ``    python frames_to_video.py --img_dir outputs_eval/exp15_3_gen_conti/test_20260407_182238/gen_imgs/de_dust2 --gt_dir data/preprocessed_data/de_dust2/imgs --output_dir outputs_eval/exp15_3_gen_conti/test_20260407_182238/gen_compared_videos/de_dust2 --max_duration 10        ``
    ``    python frames_to_video.py --img_dir outputs_eval/exp15_3_gen_conti/test_20260407_182238/gen_imgs/de_nuke --gt_dir data/preprocessed_data/de_nuke/imgs --output_dir outputs_eval/exp15_3_gen_conti/test_20260407_182238/gen_compared_videos/de_nuke --max_duration 10        ``
    ``    python frames_to_video.py --img_dir outputs_eval/exp15_3_gen_conti/test_20260407_182238/gen_imgs/de_ancient --gt_dir data/preprocessed_data/de_ancient/imgs --output_dir outputs_eval/exp15_3_gen_conti/test_20260407_182238/gen_compared_videos/de_ancient --max_duration 10        ``



## exp17
- joint loc/gen train from UniLIP-1B
- is_multi_task_balanced: True
- alpha_loc_loss: 2
- is_loc_aux_loss=True
- alpha_loc_aux_loss=0.1
- use_pi05_action_dit: True
- is_action_dit_dense_timestep: True
- is_action_dit_projector: True
- action_dit_projector_lr: 0.0005
- is_aciton_dit_vae_small_init: False
- img_size: 448
- is_lora: False

**train_csgo.py**
```     CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=29514 train_csgo.py --csgo_config csgo_configs/exp17.yaml --deepspeed deepspeed_scripts/zero0.json --model_name_or_path UniLIP-1B --unilip_factor 10.6 --mllm_hf_path OpenGVLab/InternVL3-1B-hf --version internvl --data_type "mix" --csgo_image_folder data/preprocessed_data --mm_use_im_start_end False --mm_use_im_patch_token False --bf16 True --output_dir outputs/csgo_1b/exp17 --num_train_epochs 100 --per_device_train_batch_size 8 --per_device_eval_batch_size 8 --gradient_accumulation_steps 16 --eval_strategy "no" --save_strategy "steps" --save_steps 4000 --save_total_limit 3 --learning_rate 1e-4 --weight_decay 0. --warmup_ratio 0.003 --lr_scheduler_type "cosine_with_min_lr" --model_max_length 1024 --logging_steps 1 --tf32 True --gradient_checkpointing True --dataloader_num_workers 4 --lazy_preprocess True --n_query 256 --n_und_query 0 --report_to wandb --fix_dit False --fix_connect False --fix_llm True       ```

### exp17_1
- dynamic alpha of exp17
- joint loc/gen train from scratch UniLIP-1B
- is_multi_task_balanced: True
- alpha_loc_loss: 2
- is_loc_aux_loss: True
- alpha_loc_aux schedule steps: [0, 3000, 4000, 5000, 10000]
- alpha_loc_aux schedule values: [0.0, 1.0, 2.0, 5.0, 10.0]
- use_pi05_action_dit: True
- is_action_dit_dense_timestep: True
- is_action_dit_projector: True
- action_dit_projector_lr: 0.0005
- is_aciton_dit_vae_small_init: False
- img_size: 448
- is_lora: False

**train_csgo.py**
```     CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=29517 train_csgo.py --csgo_config csgo_configs/exp17_1.yaml --deepspeed deepspeed_scripts/zero0.json --model_name_or_path UniLIP-1B --unilip_factor 10.6 --mllm_hf_path OpenGVLab/InternVL3-1B-hf --version internvl --data_type "mix" --csgo_image_folder data/preprocessed_data --mm_use_im_start_end False --mm_use_im_patch_token False --bf16 True --output_dir outputs/csgo_1b/exp17_1 --num_train_epochs 100 --per_device_train_batch_size 8 --per_device_eval_batch_size 8 --gradient_accumulation_steps 16 --eval_strategy "no" --save_strategy "steps" --save_steps 4000 --save_total_limit 3 --learning_rate 1e-4 --weight_decay 0. --warmup_ratio 0.003 --lr_scheduler_type "cosine_with_min_lr" --model_max_length 1024 --logging_steps 1 --tf32 True --gradient_checkpointing True --dataloader_num_workers 4 --lazy_preprocess True --n_query 256 --n_und_query 0 --report_to wandb --fix_dit False --fix_connect False --fix_llm True       ```

### exp17_2
- exp17_1 + dynamic alpha_loc schedule + new loc lr grouping
- joint loc/gen train from scratch UniLIP-1B
- is_multi_task_balanced: True
- alpha_loc_loss: 2
- alpha_loc schedule steps: [0, 10000, 18000, 28000]
- alpha_loc schedule values: [2.0, 5.0, 10.0, 20.0]
- is_loc_aux_loss: True
- alpha_loc_aux schedule steps: [0, 3000, 4000, 5000, 10000]
- alpha_loc_aux schedule values: [0.0, 1.0, 2.0, 5.0, 10.0]
- use_pi05_action_dit: True
- is_action_dit_dense_timestep: True
- is_action_dit_projector: True
- action_dit_projector_lr: 0.0005
- action_dit_lr: 0.0001
- is_aciton_dit_vae_small_init: False
- img_size: 448
- is_lora: False

**train_csgo.py**
```     CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node=1 --master_port=29520 train_csgo.py --csgo_config csgo_configs/exp17_2.yaml --deepspeed deepspeed_scripts/zero0.json --model_name_or_path UniLIP-1B --unilip_factor 10.6 --mllm_hf_path OpenGVLab/InternVL3-1B-hf --version internvl --data_type "mix" --csgo_image_folder data/preprocessed_data --mm_use_im_start_end False --mm_use_im_patch_token False --bf16 True --output_dir outputs/csgo_1b/exp17_2 --num_train_epochs 100 --per_device_train_batch_size 8 --per_device_eval_batch_size 8 --gradient_accumulation_steps 16 --eval_strategy "no" --save_strategy "steps" --save_steps 4000 --save_total_limit 3 --learning_rate 1e-4 --weight_decay 0. --warmup_ratio 0.003 --lr_scheduler_type "cosine_with_min_lr" --model_max_length 1024 --logging_steps 1 --tf32 True --gradient_checkpointing True --dataloader_num_workers 4 --lazy_preprocess True --n_query 256 --n_und_query 0 --report_to wandb --fix_dit False --fix_connect False --fix_llm True       ```
**eval_csgo_loc.py** 67: step=4000 8000对比exp14 12000 20000 28000 36000
```    CUDA_VISIBLE_DEVICES=1 python eval_csgo_loc.py --csgo_config csgo_configs/test/exp17_2_loc.yaml      ```
**eval_csgo.py** 67: step=4000 8000对比exp14 12000 20000 28000 36000
```    CUDA_VISIBLE_DEVICES=1 python eval_csgo.py --csgo_config csgo_configs/test/exp17_2_gen.yaml      ```
    **benchmark_csgo.py**
    ``      CUDA_VISIBLE_DEVICES=1 python benchmark_csgo.py --gt data/preprocessed_data/de_dust2/imgs --pred outputs_eval/exp17_2_gen/test_20260417_144932/gen_imgs/de_dust2 --all --batch_size 8       ``
    ``      CUDA_VISIBLE_DEVICES=1 python benchmark_csgo.py --gt data/preprocessed_data/de_nuke/imgs --pred outputs_eval/exp17_2_gen/test_20260417_144932/gen_imgs/de_nuke --all --batch_size 8       ``
    ``      CUDA_VISIBLE_DEVICES=1 python benchmark_csgo.py --gt data/preprocessed_data/de_ancient/imgs --pred outputs_eval/exp17_2_gen/test_20260417_144932/gen_imgs/de_ancient --all --batch_size 8       ``
**continuous gen**  67: step=4000 8000对比exp14 12000 20000 28000 36000
```    CUDA_VISIBLE_DEVICES=1 python eval_csgo.py --csgo_config csgo_configs/test/exp17_2_gen_conti.yaml      ```
    **benchmark_csgo.py**
    ``      CUDA_VISIBLE_DEVICES=1 python benchmark_csgo.py --gt data/preprocessed_data/de_dust2/imgs --pred outputs_eval/exp17_2_gen_conti/test_20260417_144930/gen_imgs/de_dust2 --all --batch_size 8       ``
    ``      CUDA_VISIBLE_DEVICES=1 python benchmark_csgo.py --gt data/preprocessed_data/de_nuke/imgs --pred outputs_eval/exp17_2_gen_conti/test_20260417_144930/gen_imgs/de_nuke --all --batch_size 8       ``
    ``      CUDA_VISIBLE_DEVICES=1 python benchmark_csgo.py --gt data/preprocessed_data/de_ancient/imgs --pred outputs_eval/exp17_2_gen_conti/test_20260417_144930/gen_imgs/de_ancient --all --batch_size 8       ``
    **frames to video**
    ``    python frames_to_video.py --img_dir outputs_eval/exp17_2_gen_conti/test_20260411_153010/gen_imgs/de_dust2 --gt_dir data/preprocessed_data/de_dust2/imgs --output_dir outputs_eval/exp17_2_gen_conti/test_20260411_153010/gen_compared_videos/de_dust2 --max_duration 10        ``
    ``    python frames_to_video.py --img_dir outputs_eval/exp17_2_gen_conti/test_20260411_153010/gen_imgs/de_nuke --gt_dir data/preprocessed_data/de_nuke/imgs --output_dir outputs_eval/exp17_2_gen_conti/test_20260411_153010/gen_compared_videos/de_nuke --max_duration 10        ``
    ``    python frames_to_video.py --img_dir outputs_eval/exp17_2_gen_conti/test_20260411_153010/gen_imgs/de_ancient --gt_dir data/preprocessed_data/de_ancient/imgs --output_dir outputs_eval/exp17_2_gen_conti/test_20260411_153010/gen_compared_videos/de_ancient --max_duration 10        ``

### exp17_2_dust2
- exp17_2

- dust2

**train_csgo.py**
```     CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=29520 train_csgo.py --csgo_config csgo_configs/exp17_2_dust2.yaml --deepspeed deepspeed_scripts/zero0.json --model_name_or_path UniLIP-1B --unilip_factor 10.6 --mllm_hf_path OpenGVLab/InternVL3-1B-hf --version internvl --data_type "mix" --csgo_image_folder data/preprocessed_data --mm_use_im_start_end False --mm_use_im_patch_token False --bf16 True --output_dir outputs/csgo_1b/exp17_2_dust2 --num_train_epochs 100 --per_device_train_batch_size 8 --per_device_eval_batch_size 8 --gradient_accumulation_steps 16 --eval_strategy "no" --save_strategy "steps" --save_steps 4000 --save_total_limit 3 --learning_rate 1e-4 --weight_decay 0. --warmup_ratio 0.003 --lr_scheduler_type "cosine_with_min_lr" --model_max_length 1024 --logging_steps 1 --tf32 True --gradient_checkpointing True --dataloader_num_workers 4 --lazy_preprocess True --n_query 256 --n_und_query 0 --report_to wandb --fix_dit False --fix_connect False --fix_llm True       ```
**eval_csgo_loc.py** 67: step=4000 8000
**eval_csgo_loc.py** 67: step=4000 8000
```    CUDA_VISIBLE_DEVICES=0 python eval_csgo_loc.py --csgo_config csgo_configs/test/exp17_2_dust2_loc.yaml      ```
**eval_csgo.py** 67: step=4000 8000
**eval_csgo.py** 67: step=4000 8000
```    CUDA_VISIBLE_DEVICES=0 python eval_csgo.py --csgo_config csgo_configs/test/exp17_2_dust2_gen.yaml      ```
    **benchmark_csgo.py**
    ``      CUDA_VISIBLE_DEVICES=0 python benchmark_csgo.py --gt data/preprocessed_data/de_dust2/imgs --pred outputs_eval/exp17_2_dust2_gen/test_20260417_013924/gen_imgs/de_dust2 --all --batch_size 8       ``
**continuous gen**  67: step=4000 8000
```    CUDA_VISIBLE_DEVICES=0 python eval_csgo.py --csgo_config csgo_configs/test/exp17_2_dust2_gen_conti.yaml      ```
    **benchmark_csgo.py**
    ``      CUDA_VISIBLE_DEVICES=0 python benchmark_csgo.py --gt data/preprocessed_data/de_dust2/imgs --pred outputs_eval/exp17_2_dust2_gen_conti/test_20260417_125804/gen_imgs/de_dust2 --all --batch_size 8       ``

### exp19_dust2
- exp17_2_dust2 + aux_loc step-level periodic gate
- base: `exp17_2_dust2`
- alpha_loc schedule steps: [0, 28000]
- alpha_loc schedule values: [2.0, 40.0]
- is_loc_aux_loss: True
- alpha_loc_aux schedule steps: [0, 2000, 10000]
- alpha_loc_aux schedule values: [0.0, 0.0, 10.0]
- is_loc_aux_step_gate: True
- loc_aux_gate_cycle_steps: 300
- loc_aux_gate_on_steps: 100
- loc_aux_gate_start_step: 0
- current code behavior:
  - effective alpha is `scheduled_alpha_loc_aux * periodic_gate`
  - when effective alpha is zero, `forward_for_aux_loc_loss()` is skipped
  - when effective alpha is zero and there is no `loc_repa_loss`, `pred_pixels_input` decode is also skipped
- purpose: test whether periodic aux supervision can reduce negative transfer to generation while keeping localization assistance
- dust2

**train_csgo.py**
```     CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=29532 train_csgo.py --csgo_config csgo_configs/exp19_dust2.yaml --deepspeed deepspeed_scripts/zero0.json --model_name_or_path UniLIP-1B --unilip_factor 10.6 --mllm_hf_path OpenGVLab/InternVL3-1B-hf --version internvl --data_type "mix" --csgo_image_folder data/preprocessed_data --mm_use_im_start_end False --mm_use_im_patch_token False --bf16 True --output_dir outputs/csgo_1b/exp19_dust2 --num_train_epochs 100 --per_device_train_batch_size 8 --per_device_eval_batch_size 8 --gradient_accumulation_steps 16 --eval_strategy "no" --save_strategy "steps" --save_steps 2000 --save_total_limit 3 --learning_rate 1e-4 --weight_decay 0. --warmup_ratio 0.003 --lr_scheduler_type "cosine_with_min_lr" --model_max_length 1024 --logging_steps 1 --tf32 True --gradient_checkpointing True --dataloader_num_workers 4 --lazy_preprocess True --n_query 256 --n_und_query 0 --report_to wandb --fix_dit False --fix_connect False --fix_llm True       ```
**eval_csgo_loc.py** : step=
```    CUDA_VISIBLE_DEVICES=0 python eval_csgo_loc.py --csgo_config csgo_configs/test/exp19_dust2_loc.yaml      ```
**eval_csgo.py** : step=
```    CUDA_VISIBLE_DEVICES=0 python eval_csgo.py --csgo_config csgo_configs/test/exp19_dust2_gen.yaml      ```
    **benchmark_csgo.py**
    ``      CUDA_VISIBLE_DEVICES=0 python benchmark_csgo.py --gt data/preprocessed_data/de_dust2/imgs --pred outputs_eval/exp19_dust2_gen/test_/gen_imgs/de_dust2 --all --batch_size 8       ``
**continuous gen**  : step=
```    CUDA_VISIBLE_DEVICES=0 python eval_csgo.py --csgo_config csgo_configs/test/exp19_dust2_gen_conti.yaml      ```
    **benchmark_csgo.py**
    ``      CUDA_VISIBLE_DEVICES=0 python benchmark_csgo.py --gt data/preprocessed_data/de_dust2/imgs --pred outputs_eval/exp19_dust2_gen_conti/test_/gen_imgs/de_dust2 --all --batch_size 8       ``
**frames to video**
    ``    python frames_to_video.py --img_dir outputs_eval/exp19_dust2_gen_conti/test_/gen_imgs/de_dust2 --gt_dir data/preprocessed_data/de_dust2/imgs --output_dir outputs_eval/exp19_dust2_gen_conti/test_/gen_compared_videos/de_dust2 --max_duration 10        ``

### exp20_dust2
- exp17_2_dust2 + noisy loc distribution matching
- base: `exp17_2_dust2`
- is_noisy_loc_loss: True
- noisy_loc_ratio: 0.3
- noisy_loc_image_source: `latent_space`
- noisy_loc_sigma_sampling: `gen_matched`
- noisy_loc_weight_type: `linear_1m_sigma`
- design:
  - 在 batch 内随机抽样 loc 子集
  - 将对应 loc FPS 图像编码到 target latent 空间
  - 使用和生成分支一致的 sigma/timestep 采样加噪
  - decode 回 pixel space 替换 noisy loc 子集
  - 仅对 noisy 子集覆盖 `1 - sigma` 权重
- purpose: 让定位头也在带噪输入分布上学习，以减少 `loc_aux_loss` 路径和定位主任务之间的输入分布偏差
- dust2

**train_csgo.py**
```     CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=29533 train_csgo.py --csgo_config csgo_configs/exp20_dust2.yaml --deepspeed deepspeed_scripts/zero0.json --model_name_or_path UniLIP-1B --unilip_factor 10.6 --mllm_hf_path OpenGVLab/InternVL3-1B-hf --version internvl --data_type "mix" --csgo_image_folder data/preprocessed_data --mm_use_im_start_end False --mm_use_im_patch_token False --bf16 True --output_dir outputs/csgo_1b/exp20_dust2 --num_train_epochs 100 --per_device_train_batch_size 8 --per_device_eval_batch_size 8 --gradient_accumulation_steps 16 --eval_strategy "no" --save_strategy "steps" --save_steps 2000 --save_total_limit 3 --learning_rate 1e-4 --weight_decay 0. --warmup_ratio 0.003 --lr_scheduler_type "cosine_with_min_lr" --model_max_length 1024 --logging_steps 1 --tf32 True --gradient_checkpointing True --dataloader_num_workers 4 --lazy_preprocess True --n_query 256 --n_und_query 0 --report_to wandb --fix_dit False --fix_connect False --fix_llm True       ```
**eval_csgo_loc.py** : step=
```    CUDA_VISIBLE_DEVICES=0 python eval_csgo_loc.py --csgo_config csgo_configs/test/exp20_dust2_loc.yaml      ```
**eval_csgo.py** : step=
```    CUDA_VISIBLE_DEVICES=0 python eval_csgo.py --csgo_config csgo_configs/test/exp20_dust2_gen.yaml      ```
    **benchmark_csgo.py**
    ``      CUDA_VISIBLE_DEVICES=0 python benchmark_csgo.py --gt data/preprocessed_data/de_dust2/imgs --pred outputs_eval/exp20_dust2_gen/test_/gen_imgs/de_dust2 --all --batch_size 8       ``
**eval_csgo_v1.py** : step=
```    CUDA_VISIBLE_DEVICES=0 python eval_csgo_v1.py --csgo_config csgo_configs/test/exp20_dust2_gen.yaml      ```
    **benchmark_csgo_v1.py after eval_csgo.py**
    ``      CUDA_VISIBLE_DEVICES=0 python benchmark_csgo_v1.py --gt data/preprocessed_data/de_dust2/imgs --pred outputs_eval/exp20_dust2_gen/test_<timestamp>/gen_imgs/de_dust2 --batch_size 8 --device cuda --paired_size 448 --data_dir data/preprocessed_data --map_name de_dust2       ``
**continuous gen**  : step=
```    CUDA_VISIBLE_DEVICES=0 python eval_csgo.py --csgo_config csgo_configs/test/exp20_dust2_gen_conti.yaml      ```
    **benchmark_csgo.py**
    ``      CUDA_VISIBLE_DEVICES=0 python benchmark_csgo.py --gt data/preprocessed_data/de_dust2/imgs --pred outputs_eval/exp20_dust2_gen_conti/test_/gen_imgs/de_dust2 --all --batch_size 8       ``
**eval_csgo_v1_conti.py** : step=
```    CUDA_VISIBLE_DEVICES=0 python eval_csgo_v1_conti.py --csgo_config csgo_configs/test/exp20_dust2_gen_conti.yaml      ```
    **benchmark_csgo_v1_conti.py after eval_csgo.py**
    ``      CUDA_VISIBLE_DEVICES=0 python benchmark_csgo_v1_conti.py --gt data/preprocessed_data/de_dust2/imgs --pred outputs_eval/exp20_dust2_gen_conti/test_<timestamp>/gen_imgs/de_dust2 --batch_size 8 --device cuda --paired_size 448 --data_dir data/preprocessed_data --map_name de_dust2 --frame_diff_threshold 2 --min_track_len 4 --clip_length 16 --clip_stride 16 --fvd_size 224       ``
    **frames to video**
    ``    python frames_to_video.py --img_dir outputs_eval/exp20_dust2_gen_conti/test_/gen_imgs/de_dust2 --gt_dir data/preprocessed_data/de_dust2/imgs --output_dir outputs_eval/exp20_dust2_gen_conti/test_/gen_compared_videos/de_dust2 --max_duration 10        ``

### exp17_3
- exp17_2

- train_mm_projector_only: True
- mm_projector_lr: 1.0e-5

**train_csgo.py**
```     CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=29520 train_csgo.py --csgo_config csgo_configs/exp17_3.yaml --deepspeed deepspeed_scripts/zero0.json --model_name_or_path UniLIP-1B --unilip_factor 10.6 --mllm_hf_path OpenGVLab/InternVL3-1B-hf --version internvl --data_type "mix" --csgo_image_folder data/preprocessed_data --mm_use_im_start_end False --mm_use_im_patch_token False --bf16 True --output_dir outputs/csgo_1b/exp17_3 --num_train_epochs 100 --per_device_train_batch_size 8 --per_device_eval_batch_size 8 --gradient_accumulation_steps 16 --eval_strategy "no" --save_strategy "steps" --save_steps 4000 --save_total_limit 3 --learning_rate 1e-4 --weight_decay 0. --warmup_ratio 0.003 --lr_scheduler_type "cosine_with_min_lr" --model_max_length 1024 --logging_steps 1 --tf32 True --gradient_checkpointing True --dataloader_num_workers 4 --lazy_preprocess True --n_query 256 --n_und_query 0 --report_to wandb --fix_dit False --fix_connect False --fix_llm True       ```
**eval_csgo_loc.py** 67: step=4000
```    CUDA_VISIBLE_DEVICES=0 python eval_csgo_loc.py --csgo_config csgo_configs/test/exp17_3_loc.yaml      ```
**eval_csgo.py** 67: step=4000
```    CUDA_VISIBLE_DEVICES=0 python eval_csgo.py --csgo_config csgo_configs/test/exp17_3_gen.yaml      ```
    **benchmark_csgo.py**
    ``      CUDA_VISIBLE_DEVICES=0 python benchmark_csgo.py --gt data/preprocessed_data/de_dust2/imgs --pred outputs_eval/exp17_3_gen/test_20260408_163009/gen_imgs/de_dust2 --all --batch_size 8       ``
    ``      CUDA_VISIBLE_DEVICES=0 python benchmark_csgo.py --gt data/preprocessed_data/de_nuke/imgs --pred outputs_eval/exp17_3_gen/test_20260408_163009/gen_imgs/de_nuke --all --batch_size 8       ``
    ``      CUDA_VISIBLE_DEVICES=0 python benchmark_csgo.py --gt data/preprocessed_data/de_ancient/imgs --pred outputs_eval/exp17_3_gen/test_20260408_163009/gen_imgs/de_ancient --all --batch_size 8       ``
**continuous gen**  67: step=4000
```    CUDA_VISIBLE_DEVICES=0 python eval_csgo.py --csgo_config csgo_configs/test/exp17_3_gen_conti.yaml      ```
    **benchmark_csgo.py**
    ``      CUDA_VISIBLE_DEVICES=0 python benchmark_csgo.py --gt data/preprocessed_data/de_dust2/imgs --pred outputs_eval/exp17_3_gen_conti/test_20260408_163010/gen_imgs/de_dust2 --all --batch_size 8       ``
    ``      CUDA_VISIBLE_DEVICES=0 python benchmark_csgo.py --gt data/preprocessed_data/de_nuke/imgs --pred outputs_eval/exp17_3_gen_conti/test_20260408_163010/gen_imgs/de_nuke --all --batch_size 8       ``
    ``      CUDA_VISIBLE_DEVICES=0 python benchmark_csgo.py --gt data/preprocessed_data/de_ancient/imgs --pred outputs_eval/exp17_3_gen_conti/test_20260408_163010/gen_imgs/de_ancient --all --batch_size 8       ``
    **frames to video**
    ``    python frames_to_video.py --img_dir outputs_eval/exp17_3_gen_conti/test_20260408_163010/gen_imgs/de_dust2 --gt_dir data/preprocessed_data/de_dust2/imgs --output_dir outputs_eval/exp17_3_gen_conti/test_20260408_163010/gen_compared_videos/de_dust2 --max_duration 10        ``
    ``    python frames_to_video.py --img_dir outputs_eval/exp17_3_gen_conti/test_20260408_163010/gen_imgs/de_nuke --gt_dir data/preprocessed_data/de_nuke/imgs --output_dir outputs_eval/exp17_3_gen_conti/test_20260408_163010/gen_compared_videos/de_nuke --max_duration 10        ``
    ``    python frames_to_video.py --img_dir outputs_eval/exp17_3_gen_conti/test_20260408_163010/gen_imgs/de_ancient --gt_dir data/preprocessed_data/de_ancient/imgs --output_dir outputs_eval/exp17_3_gen_conti/test_20260408_163010/gen_compared_videos/de_ancient --max_duration 10        ``


### exp17_4
- exp17_2

- train_shared_llm_tail_only: True
- shared_llm_tail_num_layers: 2
- shared_llm_tail_lr: 1.0e-5

**train_csgo.py**
```     CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=29522 train_csgo.py --csgo_config csgo_configs/exp17_4.yaml --deepspeed deepspeed_scripts/zero0.json --model_name_or_path UniLIP-1B --unilip_factor 10.6 --mllm_hf_path OpenGVLab/InternVL3-1B-hf --version internvl --data_type "mix" --csgo_image_folder data/preprocessed_data --mm_use_im_start_end False --mm_use_im_patch_token False --bf16 True --output_dir outputs/csgo_1b/exp17_4 --num_train_epochs 100 --per_device_train_batch_size 8 --per_device_eval_batch_size 8 --gradient_accumulation_steps 16 --eval_strategy "no" --save_strategy "steps" --save_steps 2000 --save_total_limit 3 --learning_rate 1e-4 --weight_decay 0. --warmup_ratio 0.003 --lr_scheduler_type "cosine_with_min_lr" --model_max_length 1024 --logging_steps 1 --tf32 True --gradient_checkpointing True --dataloader_num_workers 4 --lazy_preprocess True --n_query 256 --n_und_query 0 --report_to wandb --fix_dit False --fix_connect False --fix_llm True       ```
**eval_csgo_loc.py** : step=
```    CUDA_VISIBLE_DEVICES=0 python eval_csgo_loc.py --csgo_config csgo_configs/test/exp17_4_loc.yaml      ```
**eval_csgo.py** : step=
```    CUDA_VISIBLE_DEVICES=0 python eval_csgo.py --csgo_config csgo_configs/test/exp17_4_gen.yaml      ```
    **benchmark_csgo.py**
**continuous gen**  : step=
```    CUDA_VISIBLE_DEVICES=0 python eval_csgo.py --csgo_config csgo_configs/test/exp17_4_gen_conti.yaml      ```
    **benchmark_csgo.py**
    **frames to video**

### exp17_4_dust2
- exp17_4

- dust2

**train_csgo.py**
```     CUDA_VISIBLE_DEVICES=3 torchrun --nproc_per_node=1 --master_port=29522 train_csgo.py --csgo_config csgo_configs/exp17_4_dust2.yaml --deepspeed deepspeed_scripts/zero0.json --model_name_or_path UniLIP-1B --unilip_factor 10.6 --mllm_hf_path OpenGVLab/InternVL3-1B-hf --version internvl --data_type "mix" --csgo_image_folder data/preprocessed_data --mm_use_im_start_end False --mm_use_im_patch_token False --bf16 True --output_dir outputs/csgo_1b/exp17_4_dust2 --num_train_epochs 100 --per_device_train_batch_size 8 --per_device_eval_batch_size 8 --gradient_accumulation_steps 16 --eval_strategy "no" --save_strategy "steps" --save_steps 2000 --save_total_limit 3 --learning_rate 1e-4 --weight_decay 0. --warmup_ratio 0.003 --lr_scheduler_type "cosine_with_min_lr" --model_max_length 1024 --logging_steps 1 --tf32 True --gradient_checkpointing True --dataloader_num_workers 4 --lazy_preprocess True --n_query 256 --n_und_query 0 --report_to wandb --fix_dit False --fix_connect False --fix_llm True       ```
**eval_csgo_loc.py** : step=4000 8000
```    CUDA_VISIBLE_DEVICES=0 python eval_csgo_loc.py --csgo_config csgo_configs/test/exp17_4_dust2_loc.yaml      ```
**eval_csgo.py** : step=4000 8000
```    CUDA_VISIBLE_DEVICES=0 python eval_csgo.py --csgo_config csgo_configs/test/exp17_4_dust2_gen.yaml      ```
    **benchmark_csgo.py**
    ``      CUDA_VISIBLE_DEVICES=0 python benchmark_csgo.py --gt data/preprocessed_data/de_dust2/imgs --pred outputs_eval/exp17_4_dust2_gen/test_20260420_222815/gen_imgs/de_dust2 --all --batch_size 8       ``
**continuous gen**  : step=4000 8000
```    CUDA_VISIBLE_DEVICES=0 python eval_csgo.py --csgo_config csgo_configs/test/exp17_4_dust2_gen_conti.yaml      ```
    **benchmark_csgo.py**
    ``      CUDA_VISIBLE_DEVICES=0 python benchmark_csgo.py --gt data/preprocessed_data/de_dust2/imgs --pred outputs_eval/exp17_4_dust2_gen_conti/test_20260420_222819/gen_imgs/de_dust2 --all --batch_size 8       ``
    **frames to video**

### exp17_4_1_dust2
- exp17_4_dust2

- llm.layers[-6:]

**train_csgo.py**
```     CUDA_VISIBLE_DEVICES=3 torchrun --nproc_per_node=1 --master_port=29522 train_csgo.py --csgo_config csgo_configs/exp17_4_1_dust2.yaml --deepspeed deepspeed_scripts/zero0.json --model_name_or_path UniLIP-1B --unilip_factor 10.6 --mllm_hf_path OpenGVLab/InternVL3-1B-hf --version internvl --data_type "mix" --csgo_image_folder data/preprocessed_data --mm_use_im_start_end False --mm_use_im_patch_token False --bf16 True --output_dir outputs/csgo_1b/exp17_4_1_dust2 --num_train_epochs 100 --per_device_train_batch_size 8 --per_device_eval_batch_size 8 --gradient_accumulation_steps 16 --eval_strategy "no" --save_strategy "steps" --save_steps 2000 --save_total_limit 3 --learning_rate 1e-4 --weight_decay 0. --warmup_ratio 0.003 --lr_scheduler_type "cosine_with_min_lr" --model_max_length 1024 --logging_steps 1 --tf32 True --gradient_checkpointing True --dataloader_num_workers 4 --lazy_preprocess True --n_query 256 --n_und_query 0 --report_to wandb --fix_dit False --fix_connect False --fix_llm True       ```
**eval_csgo_loc.py** : step=2000 4000
```    CUDA_VISIBLE_DEVICES=0 python eval_csgo_loc.py --csgo_config csgo_configs/test/exp17_4_1_dust2_loc.yaml      ```
**eval_csgo.py** : step=2000 4000
```    CUDA_VISIBLE_DEVICES=0 python eval_csgo.py --csgo_config csgo_configs/test/exp17_4_1_dust2_gen.yaml      ```
    **benchmark_csgo.py**
    ``      CUDA_VISIBLE_DEVICES=0 python benchmark_csgo.py --gt data/preprocessed_data/de_dust2/imgs --pred outputs_eval/exp17_4_1_dust2_gen/test_/gen_imgs/de_dust2 --all --batch_size 8       ``
**continuous gen**  : step=2000 4000
```    CUDA_VISIBLE_DEVICES=0 python eval_csgo.py --csgo_config csgo_configs/test/exp17_4_1_dust2_gen_conti.yaml      ```
    **benchmark_csgo.py**
    ``      CUDA_VISIBLE_DEVICES=0 python benchmark_csgo.py --gt data/preprocessed_data/de_dust2/imgs --pred outputs_eval/exp17_4_1_dust2_gen_conti/test_/gen_imgs/de_dust2 --all --batch_size 8       ``
    **frames to video**

### exp17_5_dust2
- exp17_2_dust2 + independent loc-aware REPA loss
- teacher: exp14_dust2_loc checkpoint-7800
- student feature: current loc_aux path after action_dit_projector, FPS prefix tokens only
- is_loc_repa_loss: True
- alpha_loc_repa_loss: 0.1
- loc_repa_feature_type: action_prefix_tokens
- loc_repa_loss_type: cosine
- loc_repa_timestep_weight: linear_1m_sigma
- dust2

**train_csgo.py**
```     CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node=1 --master_port=29523 train_csgo.py --csgo_config csgo_configs/exp17_5_dust2.yaml --deepspeed deepspeed_scripts/zero0.json --model_name_or_path UniLIP-1B --unilip_factor 10.6 --mllm_hf_path OpenGVLab/InternVL3-1B-hf --version internvl --data_type "mix" --csgo_image_folder data/preprocessed_data --mm_use_im_start_end False --mm_use_im_patch_token False --bf16 True --output_dir outputs/csgo_1b/exp17_5_dust2 --num_train_epochs 100 --per_device_train_batch_size 8 --per_device_eval_batch_size 8 --gradient_accumulation_steps 16 --eval_strategy "no" --save_strategy "steps" --save_steps 2000 --save_total_limit 3 --learning_rate 1e-4 --weight_decay 0. --warmup_ratio 0.003 --lr_scheduler_type "cosine_with_min_lr" --model_max_length 1024 --logging_steps 1 --tf32 True --gradient_checkpointing True --dataloader_num_workers 4 --lazy_preprocess True --n_query 256 --n_und_query 0 --report_to wandb --fix_dit False --fix_connect False --fix_llm True       ```
**eval_csgo_loc.py** : step=2000 4000 6000 8000
```    CUDA_VISIBLE_DEVICES=1 python eval_csgo_loc.py --csgo_config csgo_configs/test/exp17_5_dust2_loc.yaml      ```
**eval_csgo.py** : step=2000 4000 6000 8000
```    CUDA_VISIBLE_DEVICES=0 python eval_csgo.py --csgo_config csgo_configs/test/exp17_5_dust2_gen.yaml      ```
    **benchmark_csgo.py**
    ``      CUDA_VISIBLE_DEVICES=0 python benchmark_csgo.py --gt data/preprocessed_data/de_dust2/imgs --pred outputs_eval/exp17_5_dust2_gen/test_20260420_161211/gen_imgs/de_dust2 --all --batch_size 8       ``
**continuous gen**  : step=2000 4000 6000 8000
```    CUDA_VISIBLE_DEVICES=1 python eval_csgo.py --csgo_config csgo_configs/test/exp17_5_dust2_gen_conti.yaml      ```
    **benchmark_csgo.py**
    ``      CUDA_VISIBLE_DEVICES=0 python benchmark_csgo.py --gt data/preprocessed_data/de_dust2/imgs --pred outputs_eval/exp17_5_dust2_gen_conti/test_20260420_161236/gen_imgs/de_dust2 --all --batch_size 8       ``
    **frames to video**
    ``    python frames_to_video.py --img_dir outputs_eval/exp17_5_dust2_gen_conti/test_/gen_imgs/de_dust2 --gt_dir data/preprocessed_data/de_dust2/imgs --output_dir outputs_eval/exp17_5_dust2_gen_conti/test_/gen_compared_videos/de_dust2 --max_duration 10        ``

### exp17_6_dust2
- exp17_5_dust2 + train_shared_llm_tail_only
- teacher: exp14_dust2_loc checkpoint-7800
- student feature: current loc_aux path after action_dit_projector, FPS prefix tokens only
- is_loc_repa_loss: True
- alpha_loc_repa_loss: 0.1
- loc_repa_feature_type: action_prefix_tokens
- loc_repa_loss_type: cosine
- loc_repa_timestep_weight: linear_1m_sigma
- train_shared_llm_tail_only: True
- shared_llm_tail_num_layers: 2
- shared_llm_tail_lr: 1.0e-5
- dust2

**train_csgo.py**
```     CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=29524 train_csgo.py --csgo_config csgo_configs/exp17_6_dust2.yaml --deepspeed deepspeed_scripts/zero0.json --model_name_or_path UniLIP-1B --unilip_factor 10.6 --mllm_hf_path OpenGVLab/InternVL3-1B-hf --version internvl --data_type "mix" --csgo_image_folder data/preprocessed_data --mm_use_im_start_end False --mm_use_im_patch_token False --bf16 True --output_dir outputs/csgo_1b/exp17_6_dust2 --num_train_epochs 100 --per_device_train_batch_size 8 --per_device_eval_batch_size 8 --gradient_accumulation_steps 16 --eval_strategy "no" --save_strategy "steps" --save_steps 2000 --save_total_limit 3 --learning_rate 1e-4 --weight_decay 0. --warmup_ratio 0.003 --lr_scheduler_type "cosine_with_min_lr" --model_max_length 1024 --logging_steps 1 --tf32 True --gradient_checkpointing True --dataloader_num_workers 4 --lazy_preprocess True --n_query 256 --n_und_query 0 --report_to wandb --fix_dit False --fix_connect False --fix_llm True       ```
**eval_csgo_loc.py** : step=2000 4000 8000 6000
```    CUDA_VISIBLE_DEVICES=0 python eval_csgo_loc.py --csgo_config csgo_configs/test/exp17_6_dust2_loc.yaml      ```
**eval_csgo.py** : step=2000 4000 8000 6000
```    CUDA_VISIBLE_DEVICES=0 python eval_csgo.py --csgo_config csgo_configs/test/exp17_6_dust2_gen.yaml      ```
    **benchmark_csgo.py**
    ``      CUDA_VISIBLE_DEVICES=0 python benchmark_csgo.py --gt data/preprocessed_data/de_dust2/imgs --pred outputs_eval/exp17_6_dust2_gen/test_20260420_161122/gen_imgs/de_dust2 --all --batch_size 8       ``
**continuous gen**  : step=2000 4000 8000 6000
```    CUDA_VISIBLE_DEVICES=0 python eval_csgo.py --csgo_config csgo_configs/test/exp17_6_dust2_gen_conti.yaml      ```
    **benchmark_csgo.py**
    ``      CUDA_VISIBLE_DEVICES=0 python benchmark_csgo.py --gt data/preprocessed_data/de_dust2/imgs --pred outputs_eval/exp17_6_dust2_gen_conti/test_20260420_161123/gen_imgs/de_dust2 --all --batch_size 8       ``
    **frames to video**
    ``    python frames_to_video.py --img_dir outputs_eval/exp17_6_dust2_gen_conti/test_/gen_imgs/de_dust2 --gt_dir data/preprocessed_data/de_dust2/imgs --output_dir outputs_eval/exp17_6_dust2_gen_conti/test_/gen_compared_videos/de_dust2 --max_duration 10        ``

### exp17_6_1_dust2
- exp17_6_dust2 + deeper shared LLM tail
- teacher: exp14_dust2_loc checkpoint-7800
- student feature: current loc_aux path after action_dit_projector, FPS prefix tokens only
- is_loc_repa_loss: True
- alpha_loc_repa_loss: 0.1
- loc_repa_feature_type: action_prefix_tokens
- loc_repa_loss_type: cosine
- loc_repa_timestep_weight: linear_1m_sigma
- train_shared_llm_tail_only: True
- shared_llm_tail_num_layers: 6
- shared_llm_tail_lr: 1.0e-5
- purpose: compare 6-layer shared tail full finetune against exp17_6_dust2 2-layer shared tail
- dust2

**train_csgo.py**
```     CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=29525 train_csgo.py --csgo_config csgo_configs/exp17_6_1_dust2.yaml --deepspeed deepspeed_scripts/zero0.json --model_name_or_path UniLIP-1B --unilip_factor 10.6 --mllm_hf_path OpenGVLab/InternVL3-1B-hf --version internvl --data_type "mix" --csgo_image_folder data/preprocessed_data --mm_use_im_start_end False --mm_use_im_patch_token False --bf16 True --output_dir outputs/csgo_1b/exp17_6_1_dust2 --num_train_epochs 100 --per_device_train_batch_size 8 --per_device_eval_batch_size 8 --gradient_accumulation_steps 16 --eval_strategy "no" --save_strategy "steps" --save_steps 2000 --save_total_limit 3 --learning_rate 1e-4 --weight_decay 0. --warmup_ratio 0.003 --lr_scheduler_type "cosine_with_min_lr" --model_max_length 1024 --logging_steps 1 --tf32 True --gradient_checkpointing True --dataloader_num_workers 4 --lazy_preprocess True --n_query 256 --n_und_query 0 --report_to wandb --fix_dit False --fix_connect False --fix_llm True       ```
**eval_csgo_loc.py** : step=4000 2000 6000 8000
```    CUDA_VISIBLE_DEVICES=0 python eval_csgo_loc.py --csgo_config csgo_configs/test/exp17_6_1_dust2_loc.yaml      ```
**eval_csgo.py** : step=4000 2000 6000 8000
```    CUDA_VISIBLE_DEVICES=0 python eval_csgo.py --csgo_config csgo_configs/test/exp17_6_1_dust2_gen.yaml      ```
    **benchmark_csgo.py**
    ``      CUDA_VISIBLE_DEVICES=0 python benchmark_csgo.py --gt data/preprocessed_data/de_dust2/imgs --pred outputs_eval/exp17_6_1_dust2_gen/test_20260423_140416/gen_imgs/de_dust2 --all --batch_size 8       ``
**continuous gen**  : step=4000 2000 6000 8000
```    CUDA_VISIBLE_DEVICES=0 python eval_csgo.py --csgo_config csgo_configs/test/exp17_6_1_dust2_gen_conti.yaml      ```
    **benchmark_csgo.py**
    ``      CUDA_VISIBLE_DEVICES=0 python benchmark_csgo.py --gt data/preprocessed_data/de_dust2/imgs --pred outputs_eval/exp17_6_1_dust2_gen_conti/test_20260423_140417/gen_imgs/de_dust2 --all --batch_size 8       ``
    **frames to video**
    ``    python frames_to_video.py --img_dir outputs_eval/exp17_6_1_dust2_gen_conti/test_/gen_imgs/de_dust2 --gt_dir data/preprocessed_data/de_dust2/imgs --output_dir outputs_eval/exp17_6_1_dust2_gen_conti/test_/gen_compared_videos/de_dust2 --max_duration 10        ``

### exp17_6_2_dust2 todo
- exp17_6_dust2 + 12-layer shared LLM tail with lora_only
- teacher: exp14_dust2_loc checkpoint-7800
- student feature: current loc_aux path after action_dit_projector, FPS prefix tokens only
- is_loc_repa_loss: True
- alpha_loc_repa_loss: 0.1
- loc_repa_feature_type: action_prefix_tokens
- loc_repa_loss_type: cosine
- loc_repa_timestep_weight: linear_1m_sigma
- train_shared_llm_tail_only: True
- shared_llm_tail_num_layers: 12
- shared_llm_tail_lr: 1.0e-5
- shared_llm_tail_lora_enabled: True
- shared_llm_tail_lora_mode: lora_only
- shared_llm_tail_lora_r: 16
- shared_llm_tail_lora_alpha: 32
- shared_llm_tail_lora_dropout: 0.05
- shared_llm_tail_lora_lr: 1.0e-4
- purpose: resource-constrained deeper shared tail alternative; not a pure 6-layer vs 12-layer depth comparison
- dust2

**train_csgo.py**
```     CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=29526 train_csgo.py --csgo_config csgo_configs/exp17_6_2_dust2.yaml --deepspeed deepspeed_scripts/zero0.json --model_name_or_path UniLIP-1B --unilip_factor 10.6 --mllm_hf_path OpenGVLab/InternVL3-1B-hf --version internvl --data_type "mix" --csgo_image_folder data/preprocessed_data --mm_use_im_start_end False --mm_use_im_patch_token False --bf16 True --output_dir outputs/csgo_1b/exp17_6_2_dust2 --num_train_epochs 100 --per_device_train_batch_size 8 --per_device_eval_batch_size 8 --gradient_accumulation_steps 16 --eval_strategy "no" --save_strategy "steps" --save_steps 2000 --save_total_limit 3 --learning_rate 1e-4 --weight_decay 0. --warmup_ratio 0.003 --lr_scheduler_type "cosine_with_min_lr" --model_max_length 1024 --logging_steps 1 --tf32 True --gradient_checkpointing True --dataloader_num_workers 4 --lazy_preprocess True --n_query 256 --n_und_query 0 --report_to wandb --fix_dit False --fix_connect False --fix_llm True       ```
**eval_csgo_loc.py** : step=
```    CUDA_VISIBLE_DEVICES=0 python eval_csgo_loc.py --csgo_config csgo_configs/test/exp17_6_2_dust2_loc.yaml      ```
**eval_csgo.py** : step=
```    CUDA_VISIBLE_DEVICES=0 python eval_csgo.py --csgo_config csgo_configs/test/exp17_6_2_dust2_gen.yaml      ```
    **benchmark_csgo.py**
    ``      CUDA_VISIBLE_DEVICES=0 python benchmark_csgo.py --gt data/preprocessed_data/de_dust2/imgs --pred outputs_eval/exp17_6_2_dust2_gen/test_/gen_imgs/de_dust2 --all --batch_size 8       ``
**continuous gen**  : step=
```    CUDA_VISIBLE_DEVICES=0 python eval_csgo.py --csgo_config csgo_configs/test/exp17_6_2_dust2_gen_conti.yaml      ```
    **benchmark_csgo.py**
    ``      CUDA_VISIBLE_DEVICES=0 python benchmark_csgo.py --gt data/preprocessed_data/de_dust2/imgs --pred outputs_eval/exp17_6_2_dust2_gen_conti/test_/gen_imgs/de_dust2 --all --batch_size 8       ``
    **frames to video**
    ``    python frames_to_video.py --img_dir outputs_eval/exp17_6_2_dust2_gen_conti/test_/gen_imgs/de_dust2 --gt_dir data/preprocessed_data/de_dust2/imgs --output_dir outputs_eval/exp17_6_2_dust2_gen_conti/test_/gen_compared_videos/de_dust2 --max_duration 10        ``

### exp17_7_dust2
- exp17_2_dust2 + traditional REPA
- teacher: DINOv2 base on clean target image
- teacher input size: 224
- student feature: Sana DiT block index 6 hidden states
- align type: patch_wise
- repa projector: three-layer MLP with SiLU
- repa_detach_condition: True
- is_repa_loss: True
- alpha_repa_loss: 0.5
- is_loc_aux_loss: False
- purpose: verify the standard REPA main branch before aux_loc / teacher / shared-tail ablations
- dust2

**train_csgo.py**
```     CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node=1 --master_port=29527 train_csgo.py --csgo_config csgo_configs/exp17_7_dust2.yaml --deepspeed deepspeed_scripts/zero0.json --model_name_or_path UniLIP-1B --unilip_factor 10.6 --mllm_hf_path OpenGVLab/InternVL3-1B-hf --version internvl --data_type "mix" --csgo_image_folder data/preprocessed_data --mm_use_im_start_end False --mm_use_im_patch_token False --bf16 True --output_dir outputs/csgo_1b/exp17_7_dust2 --num_train_epochs 100 --per_device_train_batch_size 8 --per_device_eval_batch_size 8 --gradient_accumulation_steps 16 --eval_strategy "no" --save_strategy "steps" --save_steps 2000 --save_total_limit 3 --learning_rate 1e-4 --weight_decay 0. --warmup_ratio 0.003 --lr_scheduler_type "cosine_with_min_lr" --model_max_length 1024 --logging_steps 1 --tf32 True --gradient_checkpointing True --dataloader_num_workers 4 --lazy_preprocess True --n_query 256 --n_und_query 0 --report_to wandb --fix_dit False --fix_connect False --fix_llm True       ```
**eval_csgo_loc.py** : step=4000 2000 6000 8000
```    CUDA_VISIBLE_DEVICES=0 python eval_csgo_loc.py --csgo_config csgo_configs/test/exp17_7_dust2_loc.yaml      ```
**eval_csgo.py** : step=4000 2000 6000 8000
```    CUDA_VISIBLE_DEVICES=0 python eval_csgo.py --csgo_config csgo_configs/test/exp17_7_dust2_gen.yaml      ```
    **benchmark_csgo.py**
    ``      CUDA_VISIBLE_DEVICES=0 python benchmark_csgo.py --gt data/preprocessed_data/de_dust2/imgs --pred outputs_eval/exp17_7_dust2_gen/test_20260423_220713/gen_imgs/de_dust2 --all --batch_size 8       ``
**continuous gen**  : step=4000 2000 6000 8000
```    CUDA_VISIBLE_DEVICES=0 python eval_csgo.py --csgo_config csgo_configs/test/exp17_7_dust2_gen_conti.yaml      ```
    **benchmark_csgo.py**
    ``      CUDA_VISIBLE_DEVICES=0 python benchmark_csgo.py --gt data/preprocessed_data/de_dust2/imgs --pred outputs_eval/exp17_7_dust2_gen_conti/test_20260423_140351/gen_imgs/de_dust2 --all --batch_size 8       ``
    **frames to video**
    ``    python frames_to_video.py --img_dir outputs_eval/exp17_7_dust2_gen_conti/test_/gen_imgs/de_dust2 --gt_dir data/preprocessed_data/de_dust2/imgs --output_dir outputs_eval/exp17_7_dust2_gen_conti/test_/gen_compared_videos/de_dust2 --max_duration 10        ``

### exp17_8_dust2
- exp17_7_dust2 + aux_loc
- teacher: DINOv2 base on clean target image
- teacher input size: 224
- student feature: Sana DiT block index 6 hidden states
- align type: patch_wise
- repa projector: three-layer MLP with SiLU
- repa_detach_condition: True
- is_repa_loss: True
- alpha_repa_loss: 0.5
- is_loc_aux_loss: True
- purpose: test whether traditional REPA and aux_loc are complementary
- dust2

**train_csgo.py**
```     CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node=1 --master_port=29528 train_csgo.py --csgo_config csgo_configs/exp17_8_dust2.yaml --deepspeed deepspeed_scripts/zero0.json --model_name_or_path UniLIP-1B --unilip_factor 10.6 --mllm_hf_path OpenGVLab/InternVL3-1B-hf --version internvl --data_type "mix" --csgo_image_folder data/preprocessed_data --mm_use_im_start_end False --mm_use_im_patch_token False --bf16 True --output_dir outputs/csgo_1b/exp17_8_dust2 --num_train_epochs 100 --per_device_train_batch_size 8 --per_device_eval_batch_size 8 --gradient_accumulation_steps 16 --eval_strategy "no" --save_strategy "steps" --save_steps 2000 --save_total_limit 3 --learning_rate 1e-4 --weight_decay 0. --warmup_ratio 0.003 --lr_scheduler_type "cosine_with_min_lr" --model_max_length 1024 --logging_steps 1 --tf32 True --gradient_checkpointing True --dataloader_num_workers 4 --lazy_preprocess True --n_query 256 --n_und_query 0 --report_to wandb --fix_dit False --fix_connect False --fix_llm True       ```
**eval_csgo_loc.py** : step=
```    CUDA_VISIBLE_DEVICES=0 python eval_csgo_loc.py --csgo_config csgo_configs/test/exp17_8_dust2_loc.yaml      ```
**eval_csgo.py** : step=
```    CUDA_VISIBLE_DEVICES=0 python eval_csgo.py --csgo_config csgo_configs/test/exp17_8_dust2_gen.yaml      ```
    **benchmark_csgo.py**
    ``      CUDA_VISIBLE_DEVICES=0 python benchmark_csgo.py --gt data/preprocessed_data/de_dust2/imgs --pred outputs_eval/exp17_8_dust2_gen/test_/gen_imgs/de_dust2 --all --batch_size 8       ``
**continuous gen**  : step=
```    CUDA_VISIBLE_DEVICES=0 python eval_csgo.py --csgo_config csgo_configs/test/exp17_8_dust2_gen_conti.yaml      ```
    **benchmark_csgo.py**
    ``      CUDA_VISIBLE_DEVICES=0 python benchmark_csgo.py --gt data/preprocessed_data/de_dust2/imgs --pred outputs_eval/exp17_8_dust2_gen_conti/test_/gen_imgs/de_dust2 --all --batch_size 8       ``
    **frames to video**
    ``    python frames_to_video.py --img_dir outputs_eval/exp17_8_dust2_gen_conti/test_/gen_imgs/de_dust2 --gt_dir data/preprocessed_data/de_dust2/imgs --output_dir outputs_eval/exp17_8_dust2_gen_conti/test_/gen_compared_videos/de_dust2 --max_duration 10        ``

### exp17_9_dust2
- exp17_2_dust2 + traditional REPA
- teacher: frozen UniLIP vision branch on clean target image
- teacher path: vision_tower + multi_modal_projector copied from current UniLIP model
- teacher input size: 448
- teacher hidden size: 896
- student feature: Sana DiT block index 6 hidden states
- align type: patch_wise
- repa projector: three-layer MLP with SiLU
- repa_detach_condition: True
- is_repa_loss: True
- alpha_repa_loss: 0.5
- is_loc_aux_loss: False
- purpose: verify whether domain-specific UniLIP vision teacher is stronger than DINOv2 for traditional REPA
- dust2

**train_csgo.py**
```     CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node=1 --master_port=29529 train_csgo.py --csgo_config csgo_configs/exp17_9_dust2.yaml --deepspeed deepspeed_scripts/zero0.json --model_name_or_path UniLIP-1B --unilip_factor 10.6 --mllm_hf_path OpenGVLab/InternVL3-1B-hf --version internvl --data_type "mix" --csgo_image_folder data/preprocessed_data --mm_use_im_start_end False --mm_use_im_patch_token False --bf16 True --output_dir outputs/csgo_1b/exp17_9_dust2 --num_train_epochs 100 --per_device_train_batch_size 8 --per_device_eval_batch_size 8 --gradient_accumulation_steps 16 --eval_strategy "no" --save_strategy "steps" --save_steps 2000 --save_total_limit 3 --learning_rate 1e-4 --weight_decay 0. --warmup_ratio 0.003 --lr_scheduler_type "cosine_with_min_lr" --model_max_length 1024 --logging_steps 1 --tf32 True --gradient_checkpointing True --dataloader_num_workers 4 --lazy_preprocess True --n_query 256 --n_und_query 0 --report_to wandb --fix_dit False --fix_connect False --fix_llm True       ```
**eval_csgo_loc.py** : step=
```    CUDA_VISIBLE_DEVICES=0 python eval_csgo_loc.py --csgo_config csgo_configs/test/exp17_9_dust2_loc.yaml      ```
**eval_csgo.py** : step=
```    CUDA_VISIBLE_DEVICES=0 python eval_csgo.py --csgo_config csgo_configs/test/exp17_9_dust2_gen.yaml      ```
    **benchmark_csgo.py**
    ``      CUDA_VISIBLE_DEVICES=0 python benchmark_csgo.py --gt data/preprocessed_data/de_dust2/imgs --pred outputs_eval/exp17_9_dust2_gen/test_/gen_imgs/de_dust2 --all --batch_size 8       ``
**continuous gen**  : step=
```    CUDA_VISIBLE_DEVICES=0 python eval_csgo.py --csgo_config csgo_configs/test/exp17_9_dust2_gen_conti.yaml      ```
    **benchmark_csgo.py**
    ``      CUDA_VISIBLE_DEVICES=0 python benchmark_csgo.py --gt data/preprocessed_data/de_dust2/imgs --pred outputs_eval/exp17_9_dust2_gen_conti/test_/gen_imgs/de_dust2 --all --batch_size 8       ``
    **frames to video**
    ``    python frames_to_video.py --img_dir outputs_eval/exp17_9_dust2_gen_conti/test_/gen_imgs/de_dust2 --gt_dir data/preprocessed_data/de_dust2/imgs --output_dir outputs_eval/exp17_9_dust2_gen_conti/test_/gen_compared_videos/de_dust2 --max_duration 10        ``

### exp17_10_dust2
- exp17_9_dust2 + aux_loc
- teacher: frozen UniLIP vision branch on clean target image
- teacher path: vision_tower + multi_modal_projector copied from current UniLIP model
- teacher input size: 448
- teacher hidden size: 896
- student feature: Sana DiT block index 6 hidden states
- align type: patch_wise
- repa projector: three-layer MLP with SiLU
- repa_detach_condition: True
- is_repa_loss: True
- alpha_repa_loss: 0.5
- is_loc_aux_loss: True
- purpose: test whether UniLIP vision teacher and aux_loc are complementary in traditional REPA
- dust2

**train_csgo.py**
```     CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node=1 --master_port=29530 train_csgo.py --csgo_config csgo_configs/exp17_10_dust2.yaml --deepspeed deepspeed_scripts/zero0.json --model_name_or_path UniLIP-1B --unilip_factor 10.6 --mllm_hf_path OpenGVLab/InternVL3-1B-hf --version internvl --data_type "mix" --csgo_image_folder data/preprocessed_data --mm_use_im_start_end False --mm_use_im_patch_token False --bf16 True --output_dir outputs/csgo_1b/exp17_10_dust2 --num_train_epochs 100 --per_device_train_batch_size 8 --per_device_eval_batch_size 8 --gradient_accumulation_steps 16 --eval_strategy "no" --save_strategy "steps" --save_steps 2000 --save_total_limit 3 --learning_rate 1e-4 --weight_decay 0. --warmup_ratio 0.003 --lr_scheduler_type "cosine_with_min_lr" --model_max_length 1024 --logging_steps 1 --tf32 True --gradient_checkpointing True --dataloader_num_workers 4 --lazy_preprocess True --n_query 256 --n_und_query 0 --report_to wandb --fix_dit False --fix_connect False --fix_llm True       ```
**eval_csgo_loc.py** : step=
```    CUDA_VISIBLE_DEVICES=0 python eval_csgo_loc.py --csgo_config csgo_configs/test/exp17_10_dust2_loc.yaml      ```
**eval_csgo.py** : step=
```    CUDA_VISIBLE_DEVICES=0 python eval_csgo.py --csgo_config csgo_configs/test/exp17_10_dust2_gen.yaml      ```
    **benchmark_csgo.py**
    ``      CUDA_VISIBLE_DEVICES=0 python benchmark_csgo.py --gt data/preprocessed_data/de_dust2/imgs --pred outputs_eval/exp17_10_dust2_gen/test_/gen_imgs/de_dust2 --all --batch_size 8       ``
**continuous gen**  : step=
```    CUDA_VISIBLE_DEVICES=0 python eval_csgo.py --csgo_config csgo_configs/test/exp17_10_dust2_gen_conti.yaml      ```
    **benchmark_csgo.py**
    ``      CUDA_VISIBLE_DEVICES=0 python benchmark_csgo.py --gt data/preprocessed_data/de_dust2/imgs --pred outputs_eval/exp17_10_dust2_gen_conti/test_/gen_imgs/de_dust2 --all --batch_size 8       ``
    **frames to video**
    ``    python frames_to_video.py --img_dir outputs_eval/exp17_10_dust2_gen_conti/test_/gen_imgs/de_dust2 --gt_dir data/preprocessed_data/de_dust2/imgs --output_dir outputs_eval/exp17_10_dust2_gen_conti/test_/gen_compared_videos/de_dust2 --max_duration 10        ``

### exp17_11_dust2
- exp17_2_dust2 + iREPA baseline
- teacher: DINOv2 base on clean target image
- teacher input size: 224
- teacher hidden size: 768
- student feature: Sana DiT block index 6 hidden states
- align type: patch_wise
- repa projector: conv + spatial norm
- repa_projector_type: `conv_spatialnorm`
- repa_use_spatial_norm: True
- repa_conv_kernel_size: 3
- repa_spatial_norm_gamma: 1.0
- repa_detach_condition: True
- is_repa_loss: True
- alpha_repa_loss: 0.5
- is_loc_aux_loss: False
- purpose: verify whether iREPA can directly replace classical REPA as the main method
- dust2

**train_csgo.py**
```     CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=29531 train_csgo.py --csgo_config csgo_configs/exp17_11_dust2.yaml --deepspeed deepspeed_scripts/zero0.json --model_name_or_path UniLIP-1B --unilip_factor 10.6 --mllm_hf_path OpenGVLab/InternVL3-1B-hf --version internvl --data_type "mix" --csgo_image_folder data/preprocessed_data --mm_use_im_start_end False --mm_use_im_patch_token False --bf16 True --output_dir outputs/csgo_1b/exp17_11_dust2 --num_train_epochs 100 --per_device_train_batch_size 8 --per_device_eval_batch_size 8 --gradient_accumulation_steps 16 --eval_strategy "no" --save_strategy "steps" --save_steps 2000 --save_total_limit 3 --learning_rate 1e-4 --weight_decay 0. --warmup_ratio 0.003 --lr_scheduler_type "cosine_with_min_lr" --model_max_length 1024 --logging_steps 1 --tf32 True --gradient_checkpointing True --dataloader_num_workers 4 --lazy_preprocess True --n_query 256 --n_und_query 0 --report_to wandb --fix_dit False --fix_connect False --fix_llm True       ```
**eval_csgo_loc.py** : step=
```    CUDA_VISIBLE_DEVICES=0 python eval_csgo_loc.py --csgo_config csgo_configs/test/exp17_11_dust2_loc.yaml      ```
**eval_csgo.py** : step=
```    CUDA_VISIBLE_DEVICES=0 python eval_csgo.py --csgo_config csgo_configs/test/exp17_11_dust2_gen.yaml      ```
    **benchmark_csgo.py**
    ``      CUDA_VISIBLE_DEVICES=0 python benchmark_csgo.py --gt data/preprocessed_data/de_dust2/imgs --pred outputs_eval/exp17_11_dust2_gen/test_/gen_imgs/de_dust2 --all --batch_size 8       ``
**continuous gen**  : step=
```    CUDA_VISIBLE_DEVICES=0 python eval_csgo.py --csgo_config csgo_configs/test/exp17_11_dust2_gen_conti.yaml      ```
    **benchmark_csgo.py**
    ``      CUDA_VISIBLE_DEVICES=0 python benchmark_csgo.py --gt data/preprocessed_data/de_dust2/imgs --pred outputs_eval/exp17_11_dust2_gen_conti/test_/gen_imgs/de_dust2 --all --batch_size 8       ``
    **frames to video**
    ``    python frames_to_video.py --img_dir outputs_eval/exp17_11_dust2_gen_conti/test_/gen_imgs/de_dust2 --gt_dir data/preprocessed_data/de_dust2/imgs --output_dir outputs_eval/exp17_11_dust2_gen_conti/test_/gen_compared_videos/de_dust2 --max_duration 10        ``

### exp18
- exp17_2 + warm-start joint training
- base & loc_head init from exp14_loc checkpoint-14040
- gen_head init from exp14_gen checkpoint-14040
- is_multi_task_balanced: True
- alpha_loc_loss: 2
- alpha_loc schedule steps: [0, 10000, 18000, 28000]
- alpha_loc schedule values: [2.0, 5.0, 10.0, 20.0]
- is_loc_aux_loss: True
- alpha_loc_aux schedule steps: [0, 3000, 4000, 5000, 10000]
- alpha_loc_aux schedule values: [0.0, 1.0, 2.0, 5.0, 10.0]
- use_pi05_action_dit: True
- is_action_dit_dense_timestep: True
- is_action_dit_projector: True
- action_dit_projector_lr: 0.0005
- action_dit_lr: 0.0001
- img_size: 448
- is_lora: False
- joint stage train epochs=70

**train_csgo.py**
```     CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=29521 train_csgo.py --csgo_config csgo_configs/exp18.yaml --deepspeed deepspeed_scripts/zero0.json --model_name_or_path UniLIP-1B --unilip_factor 10.6 --mllm_hf_path OpenGVLab/InternVL3-1B-hf --version internvl --data_type "mix" --csgo_image_folder data/preprocessed_data --mm_use_im_start_end False --mm_use_im_patch_token False --bf16 True --output_dir outputs/csgo_1b/exp18 --num_train_epochs 70 --per_device_train_batch_size 8 --per_device_eval_batch_size 8 --gradient_accumulation_steps 16 --eval_strategy "no" --save_strategy "steps" --save_steps 4000 --save_total_limit 3 --learning_rate 1e-4 --weight_decay 0. --warmup_ratio 0.003 --lr_scheduler_type "cosine_with_min_lr" --model_max_length 1024 --logging_steps 1 --tf32 True --gradient_checkpointing True --dataloader_num_workers 4 --lazy_preprocess True --n_query 256 --n_und_query 0 --report_to wandb --fix_dit False --fix_connect False --fix_llm True       ```
**eval_csgo_loc.py** 4000 8000 12000 16000 20000
```    CUDA_VISIBLE_DEVICES=0 python eval_csgo_loc.py --csgo_config csgo_configs/test/exp18_loc.yaml      ```
**eval_csgo.py** 4000 8000 12000  16000 20000
```    CUDA_VISIBLE_DEVICES=0 python eval_csgo.py --csgo_config csgo_configs/test/exp18_gen.yaml      ```
    **benchmark_csgo.py**
    ``      CUDA_VISIBLE_DEVICES=0 python benchmark_csgo.py --gt data/preprocessed_data/de_dust2/imgs --pred outputs_eval/exp18_gen/test_20260413_235240/gen_imgs/de_dust2 --all --batch_size 8       ``
    ``      CUDA_VISIBLE_DEVICES=0 python benchmark_csgo.py --gt data/preprocessed_data/de_nuke/imgs --pred outputs_eval/exp18_gen/test_20260413_235240/gen_imgs/de_nuke --all --batch_size 8       ``
    ``      CUDA_VISIBLE_DEVICES=0 python benchmark_csgo.py --gt data/preprocessed_data/de_ancient/imgs --pred outputs_eval/exp18_gen/test_20260413_235240/gen_imgs/de_ancient --all --batch_size 8       ``
**continuous gen** 4000 8000 12000 16000 20000
```    CUDA_VISIBLE_DEVICES=0 python eval_csgo.py --csgo_config csgo_configs/test/exp18_gen_conti.yaml      ```
    **benchmark_csgo.py**
    ``      CUDA_VISIBLE_DEVICES=0 python benchmark_csgo.py --gt data/preprocessed_data/de_dust2/imgs --pred outputs_eval/exp18_gen_conti/test_20260413_235244/gen_imgs/de_dust2 --all --batch_size 8       ``
    ``      CUDA_VISIBLE_DEVICES=0 python benchmark_csgo.py --gt data/preprocessed_data/de_nuke/imgs --pred outputs_eval/exp18_gen_conti/test_20260413_235244/gen_imgs/de_nuke --all --batch_size 8       ``
    ``      CUDA_VISIBLE_DEVICES=0 python benchmark_csgo.py --gt data/preprocessed_data/de_ancient/imgs --pred outputs_eval/exp18_gen_conti/test_20260413_235244/gen_imgs/de_ancient --all --batch_size 8       ``
    **frames to video**
    ``    python frames_to_video.py --img_dir outputs_eval/exp18_gen_conti/test_20260411_152913/gen_imgs/de_dust2 --gt_dir data/preprocessed_data/de_dust2/imgs --output_dir outputs_eval/exp18_gen_conti/test_20260411_152913/gen_compared_videos/de_dust2 --max_duration 10        ``
    ``    python frames_to_video.py --img_dir outputs_eval/exp18_gen_conti/test_20260411_152913/gen_imgs/de_nuke --gt_dir data/preprocessed_data/de_nuke/imgs --output_dir outputs_eval/exp18_gen_conti/test_20260411_152913/gen_compared_videos/de_nuke --max_duration 10        ``
    ``    python frames_to_video.py --img_dir outputs_eval/exp18_gen_conti/test_20260411_152913/gen_imgs/de_ancient --gt_dir data/preprocessed_data/de_ancient/imgs --output_dir outputs_eval/exp18_gen_conti/test_20260411_152913/gen_compared_videos/de_ancient --max_duration 10        ``

### exp18_dust2
- exp18

- dust2

**train_csgo.py**
```     CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node=1 --master_port=29521 train_csgo.py --csgo_config csgo_configs/exp18_dust2.yaml --deepspeed deepspeed_scripts/zero0.json --model_name_or_path UniLIP-1B --unilip_factor 10.6 --mllm_hf_path OpenGVLab/InternVL3-1B-hf --version internvl --data_type "mix" --csgo_image_folder data/preprocessed_data --mm_use_im_start_end False --mm_use_im_patch_token False --bf16 True --output_dir outputs/csgo_1b/exp18_dust2 --num_train_epochs 70 --per_device_train_batch_size 8 --per_device_eval_batch_size 8 --gradient_accumulation_steps 16 --eval_strategy "no" --save_strategy "steps" --save_steps 4000 --save_total_limit 3 --learning_rate 1e-4 --weight_decay 0. --warmup_ratio 0.003 --lr_scheduler_type "cosine_with_min_lr" --model_max_length 1024 --logging_steps 1 --tf32 True --gradient_checkpointing True --dataloader_num_workers 4 --lazy_preprocess True --n_query 256 --n_und_query 0 --report_to wandb --fix_dit False --fix_connect False --fix_llm True       ```
**eval_csgo_loc.py** : step=4000 8000
```    CUDA_VISIBLE_DEVICES=0 python eval_csgo_loc.py --csgo_config csgo_configs/test/exp18_dust2_loc.yaml      ```
**eval_csgo.py** : step=4000 8000
```    CUDA_VISIBLE_DEVICES=0 python eval_csgo.py --csgo_config csgo_configs/test/exp18_dust2_gen.yaml      ```
    **benchmark_csgo.py**
    ``      CUDA_VISIBLE_DEVICES=0 python benchmark_csgo.py --gt data/preprocessed_data/de_dust2/imgs --pred outputs_eval/exp18_dust2_gen/test_20260421_001714/gen_imgs/de_dust2 --all --batch_size 8       ``
**continuous gen**  : step=4000 8000
```    CUDA_VISIBLE_DEVICES=0 python eval_csgo.py --csgo_config csgo_configs/test/exp18_dust2_gen_conti.yaml      ```
    **benchmark_csgo.py**
    ``      CUDA_VISIBLE_DEVICES=0 python benchmark_csgo.py --gt data/preprocessed_data/de_dust2/imgs --pred outputs_eval/exp18_dust2_gen_conti/test_20260421_001719/gen_imgs/de_dust2 --all --batch_size 8       ``
    **frames to video**
    ``    python frames_to_video.py --img_dir outputs_eval/exp18_dust2_gen_conti/test_20260417_012752/gen_imgs/de_dust2 --gt_dir data/preprocessed_data/de_dust2/imgs --output_dir outputs_eval/exp18_dust2_gen_conti/test_20260417_012752/gen_compared_videos/de_dust2 --max_duration 10        ``

### exp18_1_dust2
- exp18_dust2

- gen_init = exp14_1_dust2_gen/checkpoint-8000
- lr = 5e-5
- action_dit_projector_lr: 0.0001
- action_dit_lr: 0.00005

**train_csgo.py**
```     CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node=1 --master_port=29521 train_csgo.py --csgo_config csgo_configs/exp18_1_dust2.yaml --deepspeed deepspeed_scripts/zero0.json --model_name_or_path UniLIP-1B --unilip_factor 10.6 --mllm_hf_path OpenGVLab/InternVL3-1B-hf --version internvl --data_type "mix" --csgo_image_folder data/preprocessed_data --mm_use_im_start_end False --mm_use_im_patch_token False --bf16 True --output_dir outputs/csgo_1b/exp18_1_dust2 --num_train_epochs 70 --per_device_train_batch_size 8 --per_device_eval_batch_size 8 --gradient_accumulation_steps 16 --eval_strategy "no" --save_strategy "steps" --save_steps 2000 --save_total_limit 3 --learning_rate 5e-5 --weight_decay 0. --warmup_ratio 0.003 --lr_scheduler_type "cosine_with_min_lr" --model_max_length 1024 --logging_steps 1 --tf32 True --gradient_checkpointing True --dataloader_num_workers 4 --lazy_preprocess True --n_query 256 --n_und_query 0 --report_to wandb --fix_dit False --fix_connect False --fix_llm True       ```
**eval_csgo_loc.py** : step=2000 4000
```    CUDA_VISIBLE_DEVICES=0 python eval_csgo_loc.py --csgo_config csgo_configs/test/exp18_1_dust2_loc.yaml      ```
**eval_csgo.py** : step=2000 4000
```    CUDA_VISIBLE_DEVICES=0 python eval_csgo.py --csgo_config csgo_configs/test/exp18_1_dust2_gen.yaml      ```
    **benchmark_csgo.py**
    ``      CUDA_VISIBLE_DEVICES=0 python benchmark_csgo.py --gt data/preprocessed_data/de_dust2/imgs --pred outputs_eval/exp18_1_dust2_gen/test_/gen_imgs/de_dust2 --all --batch_size 8       ``
**continuous gen**  : step=2000 4000
```    CUDA_VISIBLE_DEVICES=0 python eval_csgo.py --csgo_config csgo_configs/test/exp18_1_dust2_gen_conti.yaml      ```
    **benchmark_csgo.py**
    ``      CUDA_VISIBLE_DEVICES=0 python benchmark_csgo.py --gt data/preprocessed_data/de_dust2/imgs --pred outputs_eval/exp18_1_dust2_gen_conti/test_/gen_imgs/de_dust2 --all --batch_size 8       ``
    **frames to video**
