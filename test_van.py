import argparse
import numpy as np
import os
from tqdm import tqdm

from loss.criterion import CriterionDSN, CriterionOhemDSN
from dataset.ade_dataset import ValDataset
from networks.van import VAN
import jittor as jt
from jittor import nn
import time
from evaluator import Evaluator


IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)

BATCH_SIZE = 1
DATA_DIRECTORY = 'ADE20K'
IGNORE_LABEL = 255
LEARNING_RATE = 1e-2
MOMENTUM = 0.9
NUM_CLASSES = 19
NUM_STEPS = 40000
POWER = 0.9
RANDOM_SEED = 12345
RESTORE_FROM = os.path.join('pretrain', 'van_b2_base.pth')
SAVE_PRED_EVERY = 1
CKPT_DIR = 'checkpoints'
WEIGHT_DECAY = 0.0005


def get_parser():
    parser = argparse.ArgumentParser(description="ResNet + CCNet Training")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--ignore_label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--learning_rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--num_classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--num_steps", type=int, default=NUM_STEPS,
                        help="Number of training steps.")
    parser.add_argument("--random_seed", type=int, default=RANDOM_SEED,
                        help="Random seed to have reproducible results.")
    parser.add_argument("--restore_from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--save_pred_every", type=int, default=SAVE_PRED_EVERY,
                        help="Save summaries and checkpoint every often.")
    parser.add_argument("--ckpt_dir", type=str, default=CKPT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--recurrence", type=int, default=2,
                        help="choose the number of recurrence.")
    parser.add_argument("--ohem", action="store_true",
                        help="use hard negative mining")
    parser.add_argument("--ohem_thres", type=float, default=0.6,
                        help="choose the samples with correct probability underthe threshold.")
    parser.add_argument("--ohem_keep", type=int, default=200000,
                        help="choose the samples with correct probability underthe threshold.")
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    print(args)
    
    seed = args.random_seed
    jt.set_seed(seed)
    
    if jt.has_cuda:
        jt.flags.use_cuda = 1
    
    criterion = CriterionDSN() #CriterionCrossEntropy()

    ckpt_path = os.path.join(args.ckpt_dir, f"van-{args.recurrence}")
    os.makedirs(ckpt_path, exist_ok=True)
    model = VAN(embed_dims=[64, 128, 320, 512],
                depths=[3, 3, 12, 3], 
                num_classes=args.num_classes,
                recurrence=args.recurrence)
    
    ckpts = os.listdir(ckpt_path)
    newest = 0
    if len(ckpts) > 0: # try to load from checkpoints
        newest = max([int(x.split("-")[1].split(".")[0]) for x in ckpts])
        newest_ckpt = os.path.join(ckpt_path, f"VAN-{newest}.pkl")
        print(f"Loading checkpoint {newest_ckpt}")
        model.load(newest_ckpt)
    else: # load from pretrained model
        print(f"Loading pretrained model {args.restore_from}")
        model.load(args.restore_from)

    # data loader
    val_loader = ValDataset(shuffle=True, batch_size=args.batch_size)

    evaluator = Evaluator(args.num_classes)
    
    model.eval()
    
    for idx, (image, target) in enumerate(val_loader):
        image = image.float32()
        output = model(image)
        h, w = target.shape[1:]
        pred = nn.interpolate(output[0], size=(h, w), mode='bilinear', align_corners=True)
        pred = pred.numpy()
        target = target.numpy()
        pred = np.argmax(pred, axis=1)
        evaluator.add_batch(target, pred)
        print ('Test in iteration {}'.format(idx))
    
    mIoU = evaluator.Mean_Intersection_over_Union()

    print ('Testing result miou = {}'.format(mIoU))

    result_path = os.path.join("result", "van-{}".format(args.recurrence))
    os.makedirs(result_path, exist_ok=True)
    with open(os.path.join(result_path, "result.txt"), "w") as f:
        f.write("miou = {}".format(mIoU))


if __name__ == '__main__':
    main()
