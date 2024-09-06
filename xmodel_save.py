import torch
import json
import copy
from transformers import LlamaConfig, LlamaModel, LlamaForCausalLM

llama_model = LlamaForCausalLM.from_pretrained(
        'pretrained_mdls/vicuna_imu', # vicuna_imu1 already saved hhar weights
        load_in_8bit=False,
        torch_dtype=torch.float16,
    ).to('cpu')

# print('model:', model.state_dict().keys())
llama_state_dict = llama_model.state_dict()
encoder_state_dict = torch.load('hhar.pt', map_location='cpu')

origin_state_dict = copy.deepcopy(llama_state_dict)

print('encoder_keys:\n', encoder_state_dict.keys())
target_keys = []
for key in llama_state_dict.keys():
    if key.startswith('model.imu_encoder.'):
        target_keys.append(key)

print('target_keys:\n', target_keys)

temp_state_dict = {}
for key in target_keys:
    temp_state_dict[key] = encoder_state_dict[key.replace('model.imu_encoder.', '')]

llama_model.load_state_dict(temp_state_dict, strict=False)
# print(msg) # no unexpected keys

new_state_dict = llama_model.state_dict()
print(f'{origin_state_dict["model.imu_encoder.transformer.embed.lin.bias"]=}') # all is zero
print(f'{new_state_dict["model.imu_encoder.transformer.embed.lin.bias"]=}') # right value
# for key in target_keys:
#     # print(key)
#     assert not origin_state_dict[key].equal(new_state_dict[key])
#     assert encoder_state_dict[key.replace('model.imu_encoder.', '')].equal(new_state_dict[key])
    
llama_model.save_pretrained('/data/wenhao/wjdu/pretrained_mdls/vicuna_imu1')