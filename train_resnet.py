import argparse
import numpy as np
import os
from tqdm import tqdm

from loss.criterion import CriterionDSN, CriterionOhemDSN
from dataset.ade_dataset import TrainDataset
from networks.ccnet import ResNet, Bottleneck
import jittor as jt
import time


IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)

BATCH_SIZE = 6
DATA_DIRECTORY = 'ADE20K'
IGNORE_LABEL = 255
LEARNING_RATE = 1e-2
MOMENTUM = 0.9
NUM_CLASSES = 19
NUM_STEPS = 40000
POWER = 0.9
RANDOM_SEED = 12345
RESTORE_FROM = os.path.join('pretrain', 'resnet101_base.pth')
SAVE_PRED_EVERY = 5
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
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
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


# lr scheduler
def poly_lr_scheduler(opt, init_lr, iter, max_iter):
    new_lr = init_lr * (1 - float(iter) / max_iter) ** 0.9
    opt.lr = new_lr
    return new_lr


def main():
    parser = get_parser()
    args = parser.parse_args()
    print(args)
    
    seed = args.random_seed
    jt.set_seed(seed)
    
    if jt.has_cuda:
        jt.flags.use_cuda = 1
    
    # config network and criterion
    # if args.ohem:
    #     criterion = CriterionOhemDSN(thresh=args.ohem_thres, min_kept=args.ohem_keep)
    # else:
    criterion = CriterionDSN() #CriterionCrossEntropy()

    ckpt_path = os.path.join(args.ckpt_dir, f"resnet-{args.recurrence}")
    os.makedirs(ckpt_path, exist_ok=True)
    model = ResNet(Bottleneck,[3, 4, 23, 3], args.num_classes, criterion, args.recurrence)
    
    ckpts = os.listdir(ckpt_path)
    newest = 0
    if len(ckpts) > 0: # try to load from checkpoints
        newest = max([int(x.split("-")[1]) for x in ckpts])
        newest_ckpt = os.path.join(ckpt_path, f"CCNet-{newest}.pth")
        print(f"Loading checkpoint {newest_ckpt}")
        model.load(os.path.join(ckpt_path, newest_ckpt))
    else: # load from pretrained model
        print(f"Loading pretrained model {args.restore_from}")
        model.load(args.restore_from)

    # group weight and config optimizer
    optimizer = jt.optim.SGD(model.parameters(),
                             lr=args.learning_rate,
                             momentum=args.momentum,
                             weight_decay=args.weight_decay)
    # data loader
    train_loader = TrainDataset(shuffle=True, batch_size=args.batch_size)
    
    start_time = time.time()
    num_data = len(train_loader)
    model.train()
    for epoch in tqdm(range(newest, args.num_steps)):
        for idx, (image, target) in enumerate(train_loader):
            lr = poly_lr_scheduler(optimizer, args.learning_rate, epoch * num_data + idx, args.num_steps * num_data)
            image = image.float32()
            loss = model(image, target)
            optimizer.step(loss)
            if idx % 10 == 0:
                time_used = time.time() - start_time
                print('epoch: {}/{}, iter: {}/{}, time: {} s, loss: {}, lr: {}'.format(epoch, args.num_steps, idx, num_data, time_used, loss, lr))

        if epoch % args.save_pred_every == 0:
            print('store checkpoints ...')
            jt.save(model.state_dict(), os.path.join(ckpt_path, f"CCNet-{epoch}.pth"))


if __name__ == '__main__':
    main()
