import torch

adapter = torch.load("/data/wenhao/wjdu/output/imu_cla_10/adapter_model.bin", map_location='cpu')

print(adapter.keys())
