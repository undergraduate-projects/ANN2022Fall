import torch
import os

model = torch.load(os.path.join("pretrain", "van_b2.pth"), map_location=torch.device('cpu'))

# transform odict to dict
model_dict = {}
for k, v in model["state_dict"].items():
    model_dict[k] = v
torch.save(model_dict, os.path.join("pretrain", "van_b2_base.pth"))