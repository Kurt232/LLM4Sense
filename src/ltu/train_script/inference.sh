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
base_model='ltu_cla_p_10_1'
output_dir="../eval_res/${base_model}.json"
# cp "$0" ${output_dir}/$(date +"%Y-%m-%d-%H-%M-%S").sh

CUDA_VISIBLE_DEVICES=4 python ../inference.py \
    --base_model "/data/wenhao/wjdu/output/${base_model}/" \
    --data_path '/data/wenhao/wjdu/openaqa/data/hhar/test_pure_cla.json' \
    --output_dir $output_dir \
    --with_lora True \
    --prompt_template "alpaca_short"