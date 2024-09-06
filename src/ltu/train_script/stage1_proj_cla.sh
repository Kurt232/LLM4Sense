#!/bin/bash
#SBATCH -J alm
#SBATCH -o ./log/%j_alm.txt
#SBATCH --qos=regular
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --partition=a5
#SBATCH --ntasks-per-node=32
#SBATCH --mem=470000
#SBATCH --exclusive

export TRANSFORMERS_CACHE=./hf_cache/
export HF_DATASETS_CACHE=./hf_cache/
num_epochs=6
lr=1e-3

output_dir='/data/wenhao/wjdu/output/proj_cla' + "_${num_epochs}_${lr}"
mkdir -p $output_dir
cp "$0" ${output_dir}/$(date +"%Y-%m-%d-%H-%M-%S").sh

CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master_port=2234 ../finetune.py \
    --base_model '/data/wenhao/wjdu/pretrained_mdls/vicuna_imu1' \
    --data_path '/data/wenhao/wjdu/openaqa/data/hhar/train_toy_cla.json' \
    --output_dir $output_dir \
    --batch_size 256 \
    --micro_batch_size 8 \
    --num_epochs $num_epochs \
    --learning_rate $lr \
    --cutoff_len 108 \
    --val_set_size 0 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target_modules ['dummy'] \
    --train_on_inputs \
    --wandb_run_name ${output_dir} \
    --group_by_length \
    --save_steps 10 \
    --trainable_params proj