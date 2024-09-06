from peft import (
    LoraConfig,
    get_peft_model,
    PeftModel,
    PeftConfig,
)

import copy
import torch
from transformers import LlamaForCausalLM

llama_model = LlamaForCausalLM.from_pretrained(
        'pretrained_mdls/vicuna_imu1', # vicuna_imu1 already saved hhar weights
        load_in_8bit=False,
        torch_dtype=torch.float16,
    ).to('cpu')

# print('model:', model.state_dict().keys())
llama_state_dict = llama_model.state_dict()
origin_state_dict = copy.deepcopy(llama_state_dict)

print(f'{llama_state_dict["model.imu_encoder.transformer.embed.lin.bias"]=}') # all is zero
origin_state_dict['model.imu_encoder.transformer.embed.lin.bias'] = torch.zeros_like(llama_state_dict['model.imu_encoder.transformer.embed.lin.bias']) # change the value
llama_model.load_state_dict(origin_state_dict, strict=False)
# print(msg) # no unexpected keys

new_state_dict = llama_model.state_dict()
print(f'{new_state_dict["model.imu_encoder.transformer.embed.lin.bias"]=}') # all is zero
for key in new_state_dict.keys():
    # print(key)
    if key == 'model.imu_encoder.transformer.embed.lin.bias':
        print(f'{llama_state_dict[key]=}')
        print(f'{new_state_dict[key]=}')
        continue
    assert llama_state_dict[key].equal(new_state_dict[key])


config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules='[q_proj,v_proj]',
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
)
peft_model = get_peft_model(llama_model, config).to('cpu')

# print('lora model:', model.state_dict().keys()) will rename the keys, since add a prefix 'basemodel.'
peft_state_dict = peft_model.state_dict()
print(f'{peft_state_dict["base_model.model.model.imu_encoder.transformer.embed.lin.bias"]=}')

peft_model.save_pretrained('/data/wenhao/wjdu/output/test_save') # successfully saved

imu_encoder_state_dict = peft_model.model.model.imu_encoder.state_dict()
torch.save(imu_encoder_state_dict, '/data/wenhao/wjdu/output/test_save/encoder_model.bin') # can use torch.load to load it

# llama_model1 = LlamaForCausalLM.from_pretrained(
#         'pretrained_mdls/vicuna_imu0', 
#         load_in_8bit=False,
#         torch_dtype=torch.float16,
#     ).to('cpu')

reload_state_dict = torch.load('/data/wenhao/wjdu/output/test_save/encoder_model.bin', map_location='cpu')

target_keys = []
imu_key = ''
for key in reload_state_dict.keys():
    if 'transformer.embed.lin.bias' in key:
        print(key)
        imu_key = key
            
print(f'{reload_state_dict.get(imu_key, None)=}')
# llama_model.load_state_dict(reload_state_dict, strict=False)
# reload_peft_state_dict = llama_model.state_dict()
# print(f'{reload_peft_state_dict["base_model.model.model.imu_encoder.transformer.embed.lin.bias"]=}')


