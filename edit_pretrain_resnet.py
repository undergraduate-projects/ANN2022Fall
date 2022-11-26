import torch
import os

model = torch.load(os.path.join("pretrain", "resnet101-imagenet.pth"), map_location=torch.device('cpu'))

odict_items = model.items()
# transform odict_items to dict
model_dict = {}
for k, v in odict_items:
    model_dict[k] = v

torch.save(model_dict,os.path.join("pretrain", "resnet101_base.pth"))