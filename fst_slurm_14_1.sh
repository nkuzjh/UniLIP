#!/bin/bash
#SBATCH --job-name=unilip
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --mem=50G
#SBATCH --partition=gbunchQ3
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --output=fst_logs/%j.out  # 将日志统一收纳到 logs 文件夹
#SBATCH --error=fst_logs/%j.err

# 1. 创建日志目录，防止因找不到目录而无法输出日志
mkdir -p fst_logs

# 2. 远程集群 Conda 环境初始化 (直接使用 conda activate 会在非交互 shell 中报错)
# 这里使用通用 hook，或者替换为你自己的路径：source ~/miniconda3/etc/profile.d/conda.sh
eval "$(conda shell.bash hook)"
conda activate UniLIP

# 检查CUDA和GPU状态
echo "=== CUDA和GPU检查 ==="
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda}'); print(f'GPU Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"No GPU\"}'); print(f'BF16 Supported: {torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False}')"

# 检查GPU数量
echo "=== GPU数量检查 ==="
python -c "import torch; print(f'GPU Count: {torch.cuda.device_count()}')"

# 3. 动态生成 Master Port，防止在共享节点上发生端口冲突
MASTER_PORT=$((10000 + $RANDOM % 20000))

# 4. 启动训练
# 移除了 CUDA_VISIBLE_DEVICES=0，SLURM 会自动分配并隔离 GPU，强行指定 0 可能导致找不到卡
torchrun --nproc_per_node=1 --master_port=29518 train_csgo.py --csgo_config csgo_configs/exp14_1_gen.yaml --deepspeed deepspeed_scripts/zero0.json --model_name_or_path UniLIP-1B --unilip_factor 10.6 --mllm_hf_path OpenGVLab/InternVL3-1B-hf --version internvl --data_type "mix" --csgo_image_folder data/preprocessed_data --mm_use_im_start_end False --mm_use_im_patch_token False --bf16 True --output_dir outputs/csgo_1b/exp14_1_gen --num_train_epochs 100 --per_device_train_batch_size 128 --per_device_eval_batch_size 128 --gradient_accumulation_steps 1 --eval_strategy "no" --save_strategy "steps" --save_steps 4000 --save_total_limit 3 --learning_rate 1e-4 --weight_decay 0. --warmup_ratio 0.003 --lr_scheduler_type "cosine_with_min_lr" --model_max_length 1024 --logging_steps 1 --tf32 True --gradient_checkpointing True --dataloader_num_workers 4 --lazy_preprocess True --n_query 256 --n_und_query 0 --report_to wandb --fix_dit False --fix_connect False --fix_llm True