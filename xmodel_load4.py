import torch
from transformers import LlamaForCausalLM
from peft import (
    LoraConfig,
    get_peft_model,
    PeftModel,
    PeftConfig,
)

original_encoder_weights = torch.load('./hhar.pt', map_location='cpu')
print(original_encoder_weights.keys())

encoder_weights = torch.load('/data/wenhao/wjdu/output/imu_cla_10/checkpoint-300/pytorch_model.bin', map_location='cpu')
for key in encoder_weights.keys():
    if 'transformer.attn.proj_v.weight' in key:
        print(encoder_weights[key])
# print(f'{encoder_weights["transformer.attn.proj_v.weight"]=}')
print(f'{original_encoder_weights["transformer.attn.proj_v.weight"]=}')

# print('-' * 30)
# print(f'{original_encoder_weights["transformer.embed.pos_embed.weight"]=}')
# print(f'{encoder_weights["transformer.embed.pos_embed.weight"]=}')

# print(original_encoder_weights["transformer.embed.pos_embed.weight"].dtype)
# print(encoder_weights["transformer.embed.pos_embed.weight"].dtype)

