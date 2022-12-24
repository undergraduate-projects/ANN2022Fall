import os
import jittor as jt
from jittor.dataset.dataset import Dataset
import numpy as np
from PIL import Image, ImageOps, ImageFilter
import random


def fetch(image_path, label_path):
    with open(image_path, 'rb') as fp:
        image = Image.open(fp).convert('RGB')

    with open(label_path, 'rb') as fp:
        label = Image.open(fp).convert('P')

    return image, label


def scale(image, label):
    SCALES = (0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0)
    ratio = np.random.choice(SCALES)
    w,h = image.size
    nw = (int)(w*ratio)
    nh = (int)(h*ratio)

    image = image.resize((nw, nh), Image.BILINEAR)
    label = label.resize((nw, nh), Image.NEAREST)

    return image, label


def pad(image, label):
    w,h = image.size
    crop_size = 473
    pad_h = max(crop_size - h, 0)
    pad_w = max(crop_size - w, 0)
    image = ImageOps.expand(image, border=(0, 0, pad_w, pad_h), fill=0)
    label = ImageOps.expand(label, border=(0, 0, pad_w, pad_h), fill=255)

    return image, label


def crop(image, label):
    w, h = image.size
    crop_size = 473
    x1 = random.randint(0, w - crop_size)
    y1 = random.randint(0, h - crop_size)
    image = image.crop((x1, y1, x1 + crop_size, y1 + crop_size))
    label = label.crop((x1, y1, x1 + crop_size, y1 + crop_size))

    return image, label


def normalize(image, label):
    mean = (0.485, 0.456, 0.40)
    std = (0.229, 0.224, 0.225)
    image = np.array(image).astype(np.float32)
    label = np.array(label).astype(np.float32)

    image /= 255.0
    image -= mean
    image /= std
    return image, label


def flip(image, label):
    if random.random() < 0.5:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
        label = label.transpose(Image.FLIP_LEFT_RIGHT)
    return image, label


class BaseDataset(Dataset):
    CLASSES = (
        'wall', 'building', 'sky', 'floor', 'tree', 'ceiling', 'road', 'bed ',
        'windowpane', 'grass', 'cabinet', 'sidewalk', 'person', 'earth',
        'door', 'table', 'mountain', 'plant', 'curtain', 'chair', 'car',
        'water', 'painting', 'sofa', 'shelf', 'house', 'sea', 'mirror', 'rug',
        'field', 'armchair', 'seat', 'fence', 'desk', 'rock', 'wardrobe',
        'lamp', 'bathtub', 'railing', 'cushion', 'base', 'box', 'column',
        'signboard', 'chest of drawers', 'counter', 'sand', 'sink',
        'skyscraper', 'fireplace', 'refrigerator', 'grandstand', 'path',
        'stairs', 'runway', 'case', 'pool table', 'pillow', 'screen door',
        'stairway', 'river', 'bridge', 'bookcase', 'blind', 'coffee table',
        'toilet', 'flower', 'book', 'hill', 'bench', 'countertop', 'stove',
        'palm', 'kitchen island', 'computer', 'swivel chair', 'boat', 'bar',
        'arcade machine', 'hovel', 'bus', 'towel', 'light', 'truck', 'tower',
        'chandelier', 'awning', 'streetlight', 'booth', 'television receiver',
        'airplane', 'dirt track', 'apparel', 'pole', 'land', 'bannister',
        'escalator', 'ottoman', 'bottle', 'buffet', 'poster', 'stage', 'van',
        'ship', 'fountain', 'conveyer belt', 'canopy', 'washer', 'plaything',
        'swimming pool', 'stool', 'barrel', 'basket', 'waterfall', 'tent',
        'bag', 'minibike', 'cradle', 'oven', 'ball', 'food', 'step', 'tank',
        'trade name', 'microwave', 'pot', 'animal', 'bicycle', 'lake',
        'dishwasher', 'screen', 'blanket', 'sculpture', 'hood', 'sconce',
        'vase', 'traffic light', 'tray', 'ashcan', 'fan', 'pier', 'crt screen',
        'plate', 'monitor', 'bulletin board', 'shower', 'radiator', 'glass',
        'clock', 'flag')

    PALETTE = [[120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
               [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255],
               [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7],
               [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82],
               [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3],
               [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255],
               [255, 7, 71], [255, 9, 224], [9, 7, 230], [220, 220, 220],
               [255, 9, 92], [112, 9, 255], [8, 255, 214], [7, 255, 224],
               [255, 184, 6], [10, 255, 71], [255, 41, 10], [7, 255, 255],
               [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7],
               [255, 122, 8], [0, 255, 20], [255, 8, 41], [255, 5, 153],
               [6, 51, 255], [235, 12, 255], [160, 150, 20], [0, 163, 255],
               [140, 140, 140], [250, 10, 15], [20, 255, 0], [31, 255, 0],
               [255, 31, 0], [255, 224, 0], [153, 255, 0], [0, 0, 255],
               [255, 71, 0], [0, 235, 255], [0, 173, 255], [31, 0, 255],
               [11, 200, 200], [255, 82, 0], [0, 255, 245], [0, 61, 255],
               [0, 255, 112], [0, 255, 133], [255, 0, 0], [255, 163, 0],
               [255, 102, 0], [194, 255, 0], [0, 143, 255], [51, 255, 0],
               [0, 82, 255], [0, 255, 41], [0, 255, 173], [10, 0, 255],
               [173, 255, 0], [0, 255, 153], [255, 92, 0], [255, 0, 255],
               [255, 0, 245], [255, 0, 102], [255, 173, 0], [255, 0, 20],
               [255, 184, 184], [0, 31, 255], [0, 255, 61], [0, 71, 255],
               [255, 0, 204], [0, 255, 194], [0, 255, 82], [0, 10, 255],
               [0, 112, 255], [51, 0, 255], [0, 194, 255], [0, 122, 255],
               [0, 255, 163], [255, 153, 0], [0, 255, 10], [255, 112, 0],
               [143, 255, 0], [82, 0, 255], [163, 255, 0], [255, 235, 0],
               [8, 184, 170], [133, 0, 255], [0, 255, 92], [184, 0, 255],
               [255, 0, 31], [0, 184, 255], [0, 214, 255], [255, 0, 112],
               [92, 255, 0], [0, 224, 255], [112, 224, 255], [70, 184, 160],
               [163, 0, 255], [153, 0, 255], [71, 255, 0], [255, 0, 163],
               [255, 204, 0], [255, 0, 143], [0, 255, 235], [133, 255, 0],
               [255, 0, 235], [245, 0, 255], [255, 0, 122], [255, 245, 0],
               [10, 190, 212], [214, 255, 0], [0, 204, 255], [20, 0, 255],
               [255, 255, 0], [0, 153, 255], [0, 41, 255], [0, 255, 204],
               [41, 0, 255], [41, 255, 0], [173, 0, 255], [0, 245, 255],
               [71, 0, 255], [122, 0, 255], [0, 255, 184], [0, 92, 255],
               [184, 255, 0], [0, 133, 255], [255, 214, 0], [25, 194, 194],
               [102, 255, 0], [92, 0, 255]]

    def __init__(self, data_root="ADE20K", split='training', batch_size=1, shuffle=False):
        super().__init__()
        ''' total_len , batch_size, shuffle must be set '''
        self.batch_size = batch_size
        self.shuffle = shuffle

        assert split in ['training', 'validation']
        self.data_list_path = os.path.join(data_root, 'datalist', split + '.txt')
        self.image_path = []
        self.label_path = []

        with open(self.data_list_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                image, annotation = line.rstrip("\n").split(" ")
                # check os type, windows or linux
                if os.name == 'nt':
                    image = image.replace('/', '\\')
                    annotation = annotation.replace('/', '\\')
                elif os.name == 'posix':
                    image = image.replace('\\', '/')
                    annotation = annotation.replace('\\', '/')
                assert os.path.isfile(image)
                assert os.path.isfile(annotation)
                self.image_path.append(image)
                self.label_path.append(annotation)
        self.total_len = len(self.image_path)

        # set_attrs must be called to set batch size total len and shuffle like __len__ function in pytorch
        self.set_attrs(batch_size = self.batch_size, total_len = self.total_len, shuffle = self.shuffle) # bs , total_len, shuffle


    def __getitem__(self, image_id):
        return NotImplementedError


class TrainDataset(BaseDataset):
    def __init__(self, data_root="ADE20K", split='training', batch_size=1, shuffle=False):
        super(TrainDataset, self).__init__(data_root, split, batch_size, shuffle)

    def __getitem__(self, image_id):
        image_path = self.image_path[image_id]
        label_path = self.label_path[image_id]
        image, label = fetch(image_path, label_path)
        image, label = scale(image, label)
        image, label = pad(image, label)
        image, label = crop(image, label)
        image, label = flip(image, label)
        image, label = normalize(image, label)
        image = np.array(image).astype(np.float).transpose(2, 0, 1)
        image = jt.array(image)
        label = jt.array(np.array(label).astype(np.int))
        return image, label


class ValDataset(BaseDataset):
    def __init__(self, data_root="ADE20K", split='validation', batch_size=1, shuffle=False):
        super(ValDataset, self).__init__(data_root, split, batch_size, shuffle)
        
    def __getitem__(self, image_id):
        image_path = self.image_path[image_id]
        label_path = self.label_path[image_id]

        image, label = fetch(image_path, label_path)
        image, label = scale(image, label)
        image, label = pad(image, label)
        image, label = crop(image, label)
        image, label = flip(image, label)
        image, label = normalize(image, label)

        image = np.array(image).astype(np.float).transpose(2, 0, 1)
        image = jt.array(image)
        label = jt.array(np.array(label).astype(np.int))

        return image, label