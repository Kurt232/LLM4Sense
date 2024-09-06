from peft import (
    LoraConfig,
    get_peft_model,
    PeftModel,
    PeftConfig
)

import copy
import torch
from transformers import LlamaForCausalLM

###
peft_model_id = '/data/wenhao/wjdu/output/test_save'
peft_config = PeftConfig.from_pretrained(peft_model_id)
llama_model = LlamaForCausalLM.from_pretrained(
        peft_config.base_model_name_or_path,
    ).to('cpu')
peft_model = PeftModel.from_pretrained(
        model=llama_model,
        model_id=peft_model_id, # original imu encoder
    ).to('cpu')

## can't load
# llama_model = LlamaForCausalLM.from_pretrained( # can't load
#         'pretrained_mdls/vicuna_imu',
#     ).to('cpu')
# peft_model = PeftModel.from_pretrained(
#         model=llama_model,
#         model_id='/data/wenhao/wjdu/output/test_save', # original imu encoder
#     ).to('cpu')
peft_state_dict = peft_model.state_dict()
print(f'{len(peft_state_dict.keys())=}')
encoder_weight_keys = []
for key in peft_state_dict.keys():
    if 'imu' in key:
        encoder_weight_keys.append(key)

# print(f'{encoder_weight_keys=}')
print(f'{peft_state_dict["base_model.model.model.imu_encoder.transformer.embed.lin.bias"]=}')

# make sure the model can load

### compare the finetuned model
## can't load
# llama_model1 = LlamaForCausalLM.from_pretrained(
#         'pretrained_mdls/vicuna_imu1',
#     ).to('cpu')
# peft_model1 = PeftModel.from_pretrained(
#         model=llama_model1,
#         model_id='/data/wenhao/wjdu/output/imu_toy_20', # finetuned imu encoder
#     ).to('cpu')
####
peft_model_id1 = '/data/wenhao/wjdu/output/imu_toy_20'
peft_config1 = PeftConfig.from_pretrained(peft_model_id1)
peft_config1.base_model_name_or_path = 'pretrained_mdls/vicuna_imu1'
llama_model1 = LlamaForCausalLM.from_pretrained(
        peft_config1.base_model_name_or_path,
    ).to('cpu')
peft_model1 = PeftModel.from_pretrained(
        model=llama_model1,
        model_id=peft_model_id1, # original imu encoder
    ).to('cpu')
peft_state_dict1 = peft_model1.state_dict()
print(f'{peft_state_dict1["base_model.model.model.imu_encoder.transformer.embed.lin.bias"]=}')
    