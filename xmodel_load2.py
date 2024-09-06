from peft import (
    LoraConfig,
    get_peft_model,
    PeftModel,
    PeftConfig
)

import copy
import torch
from transformers import LlamaForCausalLM, AutoModelForCausalLM

llama_model = LlamaForCausalLM.from_pretrained(
        'pretrained_mdls/vicuna_imu1', # vicuna_imu1 already saved hhar weights
        # vicuna_imu1 compare with vicuna_imu, the only difference the config has the weights details
        load_in_8bit=False,
        torch_dtype=torch.float16,
    ).to('cpu')

# print('model:', model.state_dict().keys())
llama_state_dict = llama_model.state_dict()
encoder_state_dict = torch.load('hhar.pt', map_location='cpu')
print(type(encoder_state_dict))
print('encoder_keys:\n', encoder_state_dict.keys())
target_keys = []
for key in llama_state_dict.keys():
    if key.startswith('model.imu_encoder.'):
        target_keys.append(key)

# print('target_keys:\n', target_keys)

# ! check vicuna_imu1 has already saved hhar weights
for key in target_keys:
    if not llama_state_dict[key].equal(encoder_state_dict[key.replace('model.imu_encoder.', '')].to(torch.float16)):
        print(key)

peft_model_id = "/data/wenhao/wjdu/output/test_save"
config = PeftConfig.from_pretrained(peft_model_id)
inference_model = LlamaForCausalLM.from_pretrained(config.base_model_name_or_path)
model = PeftModel.from_pretrained(inference_model, peft_model_id)

inference_state_dict = inference_model.state_dict()
print(f'{inference_state_dict["base_model.model.model.imu_encoder.transformer.embed.lin.bias"]=}')
