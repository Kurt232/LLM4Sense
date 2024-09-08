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
num_epochs=10
output_dir='/data/wenhao/wjdu/output/llasa_stage2/'
mkdir -p $output_dir
cp "$0" ${output_dir}/$(date +"%Y-%m-%d-%H-%M-%S").sh

CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master_port=2237 ../finetune.py \
    --base_model '/data/wenhao/wjdu/output/llasa_stage1/' \
    --data_path '/data/wenhao/wjdu/openaqa/data/hhar/train_pure_cla.json' \
    --output_dir $output_dir \
    --batch_size 256 \
    --micro_batch_size 4 \
    --num_epochs $num_epochs \
    --learning_rate 2e-4 \
    --cutoff_len 100 \
    --val_set_size 0 \
    --lora_r 128 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target_modules '[q_proj,v_proj]' \
    --train_on_inputs \
    --group_by_length \
    --wandb_run_name ${output_dir} \
    --save_steps 20 \
    --trainable_params proj \
    --enable_lora True \

# setting enable_lora as true will fine-tune the llm