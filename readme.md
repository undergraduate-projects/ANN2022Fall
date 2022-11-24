# CCNet implemented by Jittor

## Preprocess 
### Dataset
Download ADE dataset and make sure the file structures are as follow:
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
Download pytorch version pretrained ResNet101 and store it at `pretrain/resnet101-imagenet.pth`, then run `python3 edit-pretrain.py` to get `pretrain/resnet101_base.pth` for Jittor reloading.

## Run
Run the following instruction, and the output log will be stored at `log/yyyymmdd_hhmmss.log` 
```
./train.sh
```