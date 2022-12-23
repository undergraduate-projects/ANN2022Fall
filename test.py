import jittor as jt
from jittor import init
from jittor import nn

import argparse
import scipy
from scipy import ndimage
import cv2
import numpy as np
import sys
import json
from networks.ccnet import ResNet, Bottleneck
from dataset.ade_dataset import TrainDataset
from collections import OrderedDict
import os
import scipy.ndimage as nd
from math import ceil
from PIL import Image as PILImage


IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

DATA_DIRECTORY = 'ADE20K'
DATA_LIST_PATH = './dataset/list/cityscapes/test.lst'
IGNORE_LABEL = 255
NUM_CLASSES = 19
INPUT_SIZE = '769,769'
RESTORE_FROM = './deeplab_resnet.ckpt'

def get_arguments():
    """Parse all the arguments provided from the CLI.
    Returns:
        A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description='DeepLabLFOV Network')
    parser.add_argument('--data-dir', type=str, default=DATA_DIRECTORY, help='Path to the directory containing the PASCAL VOC dataset.')
    parser.add_argument('--data-list', type=str, default=DATA_LIST_PATH, help='Path to the file listing the images in the dataset.')
    parser.add_argument('--ignore-label', type=int, default=IGNORE_LABEL, help='The index of the label to ignore during the training.')
    parser.add_argument('--num-classes', type=int, default=NUM_CLASSES, help='Number of classes to predict (including background).')
    parser.add_argument('--restore-from', type=str, default=RESTORE_FROM, help='Where restore model parameters from.')
    parser.add_argument('--gpu', type=str, default='0', help='choose gpu device.')
    parser.add_argument('--recurrence', type=int, default=1, help='choose the number of recurrence.')
    parser.add_argument('--input-size', type=str, default=INPUT_SIZE, help='Comma-separated string with height and width of images.')
    parser.add_argument('--whole', type=bool, default=False, help='use whole input size.')
    return parser.parse_args()

def get_palette(num_cls):
    """Returns the color map for visualizing the segmentation mask.
    Args:
        num_cls: Number of classes
    Returns:
        The color map
    """
    n = num_cls
    palette = ([0] * (n * 3))
    for j in range(0, n):
        lab = j
        palette[((j * 3) + 0)] = 0
        palette[((j * 3) + 1)] = 0
        palette[((j * 3) + 2)] = 0
        i = 0
        while lab:
            palette[((j * 3) + 0)] |= (((lab >> 0) & 1) << (7 - i))
            palette[((j * 3) + 1)] |= (((lab >> 1) & 1) << (7 - i))
            palette[((j * 3) + 2)] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
    return palette

def pad_image(img, target_size):
    """Pad an image up to the target size."""
    rows_missing = (target_size[0] - img.shape[2])
    cols_missing = (target_size[1] - img.shape[3])
    padded_img = np.pad(img, ((0, 0), (0, 0), (0, rows_missing), (0, cols_missing)), 'constant')
    return padded_img

def predict_sliding(net, image, tile_size, classes, flip_evaluation, recurrence):
    interp = nn.Upsample(mode='bilinear')
    image_size = image.shape
    overlap = (1 / 3)
    stride = ceil((tile_size[0] * (1 - overlap)))
    tile_rows = int((ceil(((image_size[2] - tile_size[0]) / stride)) + 1))
    tile_cols = int((ceil(((image_size[3] - tile_size[1]) / stride)) + 1))
    print(('Need %i x %i prediction tiles @ stride %i px' % (tile_cols, tile_rows, stride)))
    full_probs = np.zeros((image_size[2], image_size[3], classes))
    count_predictions = np.zeros((image_size[2], image_size[3], classes))
    tile_counter = 0
    for row in range(tile_rows):
        for col in range(tile_cols):
            x1 = int((col * stride))
            y1 = int((row * stride))
            x2 = min((x1 + tile_size[1]), image_size[3])
            y2 = min((y1 + tile_size[0]), image_size[2])
            x1 = max(int((x2 - tile_size[1])), 0)
            y1 = max(int((y2 - tile_size[0])), 0)
            img = image[:, :, y1:y2, x1:x2]
            padded_img = pad_image(img, tile_size)
            tile_counter += 1
            print(('Predicting tile %i' % tile_counter))
            padded_prediction = net(Variable(torch.from_numpy(padded_img), volatile=True).cuda(), recurrence)
            if isinstance(padded_prediction, list):
                padded_prediction = padded_prediction[0]
            padded_prediction = interp(padded_prediction).cpu().data[0].numpy().transpose(1, 2, 0)
            prediction = padded_prediction[0:img.shape[2], 0:img.shape[3], :]
            count_predictions[y1:y2, x1:x2] += 1
            full_probs[y1:y2, x1:x2] += prediction
    full_probs /= count_predictions
    return full_probs

def predict_whole(net, image, tile_size, flip_evaluation, recurrence):
    interp = nn.Upsample(mode='bilinear')
    prediction = net(image.cuda(), recurrence)
    if isinstance(prediction, list):
        prediction = prediction[0]
    prediction = interp(prediction).cpu().data[0].numpy().transpose(1, 2, 0)
    return prediction

def id2trainId(label, id_to_trainid, reverse=False):
    label_copy = label.copy()
    if reverse:
        for (v, k) in id_to_trainid.items():
            label_copy[(label == k)] = v
    else:
        for (k, v) in id_to_trainid.items():
            label_copy[(label == k)] = v
    return label_copy

def main():
    """Create the model and start the evaluation process."""
    args = get_arguments()
    
    #gpu0 = args.gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    (h, w) = map(int, args.input_size.split(','))
    if args.whole:
        input_size = (1024, 2048)
    else:
        input_size = (h, w)
    ignore_label = args.ignore_label
    model = ResNet(Bottleneck,[3, 4, 23, 3], args.num_classes, criterion, args.recurrence)
    saved_state_dict = model.load(args.restore_from)
    model.load_parameters(saved_state_dict)
    model.eval()
    model.cuda()
    testloader = data.DataLoader(TrainDataset(args.data_dir, args.data_list, crop_size=(1024, 2048), mean=IMG_MEAN), batch_size=1, shuffle=False, pin_memory=True)
    data_list = []
    confusion_matrix = np.zeros((args.num_classes, args.num_classes))
    palette = get_palette(256)
    id_to_trainid = {(- 1): ignore_label, 0: ignore_label, 1: ignore_label, 2: ignore_label, 3: ignore_label, 4: ignore_label, 5: ignore_label, 6: ignore_label, 7: 0, 8: 1, 9: ignore_label, 10: ignore_label, 11: 2, 12: 3, 13: 4, 14: ignore_label, 15: ignore_label, 16: ignore_label, 17: 5, 18: ignore_label, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14, 28: 15, 29: ignore_label, 30: ignore_label, 31: 16, 32: 17, 33: 18}
    interp = nn.Upsample(mode='bilinear')
    
    if (not os.path.exists('outputs')):
        os.makedirs('outputs')
    
    for (index, batch) in enumerate(testloader):
        if ((index % 100) == 0):
            print(('%d processd' % index))
        (image, size, name) = batch
        size = size[0].numpy()
        with torch.no_grad():
            if args.whole:
                output = predict_whole(model, image, input_size, True, args.recurrence)
            else:
                output = predict_sliding(model, image.numpy(), input_size, args.num_classes, True, args.recurrence)
        
        seg_pred = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)
        seg_pred = id2trainId(seg_pred, id_to_trainid, reverse=True)
        output_im = PILImage.fromarray(seg_pred)
        output_im.putpalette(palette)
        output_im.save((('outputs/' + name[0]) + '.png'))
        
        
if __name__ == '__main__':
    main()