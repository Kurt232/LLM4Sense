from peft import (
    LoraConfig,
    PeftConfig,
    PeftModel,
    get_peft_model,
)

import copy
import torch
from transformers import LlamaForCausalLM

model = LlamaForCausalLM.from_pretrained(
    'pretrained_mdls/vicuna_imu',
    device_map="auto", 
    load_in_8bit=False, 
    torch_dtype=torch.float32
  )

config = LoraConfig(
      r=8,
      lora_alpha=16,
      target_modules=["q_proj", "v_proj"],
      lora_dropout=0.0,
      bias="none",
      task_type="CAUSAL_LM",
  )

model = get_peft_model(model, config)
eval_mdl_path = '/data/wenhao/wjdu/output/test_save/adapter_model.bin'
state_dict = torch.load(eval_mdl_path, map_location='cpu')

print(state_dict.keys())
imu_keys = []
for key in state_dict.keys():
  if 'imu' in key:
    imu_keys.append(key)
    
print(imu_keys)
