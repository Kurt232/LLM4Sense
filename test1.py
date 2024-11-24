import torch
import json
from transformers import LlamaConfig, LlamaModel, LlamaForCausalLM

model = LlamaForCausalLM.from_pretrained(
        'pretrained_mdls/vicuna_imu1',
        load_in_8bit=False,
        #torch_dtype=torch.float16
    ).to('cpu')

state_dict = model.state_dict()

print(state_dict['model.imu_encoder.transformer.norm2.gamma'])

_state_dict = torch.load('hhar.pt', weights_only=True, map_location='cpu')

# for k, v in _state_dict.items():
#     state_dict['model.imu_encoder.' + k] = v

# model.load_state_dict(state_dict, strict=False)

# model.save_pretrained('/data/wenhao/wjdu/pretrained_mdls/vicuna_imu2')