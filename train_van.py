import argparse
import numpy as np
import os
from tqdm import tqdm

from tensorboardX import SummaryWriter
from loss.criterion import CriterionDSN, CriterionOhemDSN
from dataset.ade_dataset import TrainDataset
from networks.van import VAN
import jittor as jt


IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)

BATCH_SIZE = 8
DATA_DIRECTORY = 'ADE20K'
IGNORE_LABEL = 255
NUM_CLASSES = 150
NUM_STEPS = 40000
RANDOM_SEED = 12345
RESTORE_FROM = os.path.join('pretrain', 'van_b2_base.pth')
SAVE_NUM_IMAGES = 2
SAVE_PRED_EVERY = 10000
SNAPSHOT_DIR = 'checkpoints'
LEARNING_RATE = 0.00006


def get_parser():
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--is-training", action="store_true",
                        help="Whether to updates the running means and variances during the training.")
    parser.add_argument("--not-restore-last", action="store_true",
                        help="Whether to not restore last (FC) layers.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--start-iters", type=int, default=0,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Number of training steps.")
    parser.add_argument("--print-frequency", type=int, default=50,
                        help="Number of training steps.") 
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                        help="Random seed to have reproducible results.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--save-num-images", type=int, default=SAVE_NUM_IMAGES,
                        help="How many images to save.")
    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                        help="Save summaries and checkpoint every often.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--recurrence", type=int, default=0,
                        help="choose the number of recurrence.")
    parser.add_argument("--ohem", action="store_true",
                        help="use hard negative mining")
    parser.add_argument("--ohem-thres", type=float, default=0.6,
                        help="choose the samples with correct probability underthe threshold.")
    parser.add_argument("--ohem-keep", type=int, default=200000,
                        help="choose the samples with correct probability underthe threshold.")
    return parser


# lr scheduler
def poly_lr_scheduler(opt, init_lr, iter, max_iter):
    new_lr = init_lr * (1 - float(iter) / max_iter) ** 0.9
    opt.lr = new_lr
    return new_lr


def main():
    parser = get_parser()
    args = parser.parse_args()

    seed = args.random_seed
    jt.set_seed(seed)

    if jt.has_cuda:
        jt.flags.use_cuda = 1

    # config network and criterion
    if args.ohem:
        criterion = CriterionOhemDSN(thresh=args.ohem_thres, min_kept=args.ohem_keep)
    else:
        criterion = CriterionDSN() #CriterionCrossEntropy()

    model = VAN(embed_dims=[64, 128, 320, 512],
                depths=[3, 3, 12, 3], 
                num_classes=args.num_classes,
                recurrence=args.recurrence)

    if args.restore_from is not None:
        model.load(args.restore_from)

    optimizer = jt.optim.AdamW(model.parameters(), 
                                lr=LEARNING_RATE, 
                                betas=(0.9, 0.999),
                                weight_decay=0.01)
    # data loader
    train_loader = TrainDataset(shuffle=True, batch_size=args.batch_size)
    optimizer.zero_grad()
    os.makedirs(args.snapshot_dir, exist_ok=True)
    
    for epoch in tqdm(range(args.num_steps)):
        model.train()
        for idx, (image, target) in enumerate(train_loader):
            optimizer.zero_grad()
            lr = poly_lr_scheduler(optimizer, LEARNING_RATE, epoch * len(train_loader) + idx, args.num_steps * len(train_loader))
            image = image.float32()
            pred = model(image)
            loss = criterion(pred, target)
            optimizer.step(loss)
            print('epoch: {}, iter: {}, loss: {}, lr: {}'.format(epoch, idx, loss, lr))

        if epoch % args.save_pred_every == 0:
            print('taking snapshot ...')
            jt.save(model.state_dict(), os.path.join(args.snapshot_dir, f"VAN-{epoch}.pth"))  

if __name__ == '__main__':
    main()