#!/bin/bash
#SBATCH -J alm
#SBATCH -o ./log/%j_alm.txt
#SBATCH --qos=regular
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --partition=a6
#SBATCH --ntasks-per-node=32
#SBATCH --mem=470000
#SBATCH --exclusive

export TRANSFORMERS_CACHE=./hf_cache/
export HF_DATASETS_CACHE=./hf_cache/
output_dir='/data/wenhao/wjdu/output/imu_cla_p_l_10/'
mkdir -p $output_dir 
cp "$0" ${output_dir}/$(date +"%Y-%m-%d-%H-%M-%S").sh

CUDA_VISIBLE_DEVICES=0,2,3,4 torchrun --nproc_per_node=4 --master_port=2234 ../finetune1.py \
    --base_model '/data/wenhao/wjdu/pretrained_mdls/vicuna_imu1' \
    --data_path '/data/wenhao/wjdu/openaqa/data/hhar/train_pure_cla.json' \
    --output_dir $output_dir \
    --batch_size 4 \
    --micro_batch_size 1 \
    --num_epochs 10 \
    --learning_rate 1e-6 \
    --cutoff_len 100 \
    --val_set_size 0 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target_modules '[q_proj,v_proj]' \
    --train_on_inputs \
    --group_by_length \
    --wandb_run_name ${output_dir} \
    --save_steps 100 \
    --trainable_params all