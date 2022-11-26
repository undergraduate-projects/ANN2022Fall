# CCNet implemented by Jittor

## Preprocess 
### Dataset
Please download [ADE dataset](https://groups.csail.mit.edu/vision/datasets/ADE20K/) and make sure the file structures are as follow:
```
ADE20K
|-images/
|-index_ade20k.mat
|-index_ad220k.pkl
|-objects.txt
```
You need to run `python3 edit-dataset.py` to generate data list for ADE20K dataset, after that the file structures will be:
```
ADE20K
|-images/
|-datalist/
|-index_ade20k.mat
|-index_ad220k.pkl
|-objects.txt
```

### Pretrain
#### ResNet101 Backbone
Please download MIT imagenet pretrained [resnet101-imagenet.pth](http://sceneparsing.csail.mit.edu/model/pretrained_resnet/resnet101-imagenet.pth) and store it at `pretrain/resnet101-imagenet.pth`, then run `python3 edit_pretrain_resnet.py` to get `pretrain/resnet101_base.pth` for Jittor reloading.

#### VAN_b2 Backbone
Please download pytorch version pretrained [van_b2.pth](https://cloud.tsinghua.edu.cn/d/0100f0cea37d41ba8d08/) and store it at `pretrain/van_b2.pth`, then run `python3 edit_pretrain_van.py` to get `pretrain/van_b2_base.pth` for Jittor reloading.

## Run
Run the following instruction, and the output log will be stored at `log/yyyymmdd_hhmmss.log` 
```
./train.sh
```