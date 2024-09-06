import sys
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "4"

import os
import fire
import json
import torch
import time
import numpy as np
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
    PeftModel
)
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer, LlamaConfig

from utils.prompter import Prompter

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_data(data: list):
    data = np.array(data, dtype=np.float32)
    acc_norm = 9.8
    data[:, :3] = data[:, :3] / acc_norm
    return torch.from_numpy(data)

def main(
    base_model: str = "",
    data_path: str = "",
    output_dir: str = "",
    with_lora: bool = True,
    prompt_template: str = "alpaca_short",  # The prompt template to use, will default to alpaca.
):
    
    # trick to load checkpoints correctly from HF
    if 'pretrained_mdls' not in base_model: # stage2
        # start from a different model with original vicuna
        # temporally first load the original vicuna, then load the actual checkpoint
        start_model = base_model # need to point to a specific bin file that contains state dict.
        # TODO: change to your vicuna_tltr path
        base_model = '/data/wenhao/wjdu/pretrained_mdls/vicuna_imu1'
        print('Will load from {:s} later, for implementation purpose, first load from {:s}'.format(start_model, base_model))
    else:
        start_model = None
    
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"

    prompter = Prompter(prompt_template)
    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    
    model = LlamaForCausalLM.from_pretrained(
      base_model, 
      device_map="auto", 
      load_in_8bit=False, 
      torch_dtype=torch.bfloat16
    )
    
    if start_model:
        # change it to your model path
        eval_mdl_path = start_model
        encoder_weights = torch.load(os.path.join(eval_mdl_path, 'encoder_model.bin'), map_location='cpu')
        proj_weights = torch.load(os.path.join(eval_mdl_path, 'proj_model.bin'), map_location='cpu')
        model.model.imu_encoder.load_state_dict(encoder_weights)
        model.model.imu_proj.load_state_dict(proj_weights)
    
        if with_lora:
            model = PeftModel.from_pretrained( # ! do not load Lora weights, but it will initialize random weights
                model=model,
                model_id=eval_mdl_path
            )
    
    # config = LoraConfig(
    #     r=8,
    #     lora_alpha=16,
    #     target_modules=["q_proj", "v_proj"],
    #     lora_dropout=0.0,
    #     bias="none",
    #     task_type="CAUSAL_LM",
    # )

    # model = get_peft_model(model, config)
    
    # eval_mdl_path = '/data/wenhao/wjdu/output/imu_cla_10/checkpoint-300/pytorch_model.bin'
    # state_dict = torch.load(eval_mdl_path, map_location='cpu')
    # msg = model.load_state_dict(state_dict, strict=False)
    
    # model = model.to(device)
    
    temp, top_p, top_k = 0.11, 0.95, 500
    
    model.is_parallelizable = True
    model.model_parallel = True
    
    # unwind broken decapoda-research config
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2
    
    model.eval()

    data_json = json.load(open(data_path, 'r'))
    # data_json = json.load(open('/data/wenhao/wjdu/openaqa/data/hhar/test_toy.json', 'r'))
    result_json = []
    for i in range(len(data_json)):
        cur_answer = data_json[i]["output"]
        data_id = data_json[i]["data_id"]
        imu_data = data_json[i]["imu_input"]

        instruction = data_json[i]["instruction"]

        begin_time = time.time()

        prompt = prompter.generate_prompt(instruction, None)
        print('Input prompt: ', prompt)
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)

        imu_input = load_data(imu_data).unsqueeze(0) # set batch_size = 1
        if torch.cuda.is_available() == False:
            pass
        else:
            imu_input = imu_input.bfloat16().to(device)

        generation_config = GenerationConfig(
            do_sample=True,
            temperature=temp,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=1.1,
            max_new_tokens=200,
            bos_token_id=model.config.bos_token_id,
            eos_token_id=model.config.eos_token_id,
            pad_token_id=model.config.pad_token_id,
            num_return_sequences=1
        )

        # Without streaming
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                imu_input=imu_input,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=200,
            )
        s = generation_output.sequences[0]
        output = tokenizer.decode(s)[6:-4]
        end_time = time.time()
        print(output)
        
        print(40*'-', '\nlabel: ', cur_answer)
        print('eclipse time: ', end_time-begin_time, ' seconds.')

        result_json.append({'prompt': instruction, 'pred': output[len(prompt):], 'ref': cur_answer, 'data_id': data_id})
        if os.path.exists('./eval_res') == False:
            os.mkdir('./eval_res')
        with open(output_dir, 'w') as fj:
            json.dump(result_json, fj, indent=1)

if __name__ == "__main__":
    fire.Fire(main)
