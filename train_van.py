import argparse
import numpy as np
import os
from tqdm import tqdm

from loss.criterion import CriterionDSN, CriterionOhemDSN
from dataset.ade_dataset import TrainDataset
from networks.van import VAN
import jittor as jt
import time


IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)

BATCH_SIZE = 1
DATA_DIRECTORY = 'ADE20K'
IGNORE_LABEL = 255
NUM_CLASSES = 19
NUM_STEPS = 40000
RANDOM_SEED = 12345
RESTORE_FROM = os.path.join('pretrain', 'van_b2_base.pth')
SAVE_NUM_IMAGES = 2
SAVE_PRED_EVERY = 1
CKPT_DIR = 'checkpoints'
LEARNING_RATE = 0.00006
WEIGHT_DECAY = 0.01

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

    model = VAN(embed_dims=[64, 128, 320, 512],
                depths=[3, 3, 12, 3], 
                num_classes=args.num_classes,
                recurrence=args.recurrence)

    ckpt_path = os.path.join(args.ckpt_dir, f"van-{args.recurrence}")
    os.makedirs(ckpt_path, exist_ok=True)

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

    optimizer = jt.optim.AdamW(model.parameters(), 
                                lr=args.learning_rate, 
                                betas=(0.9, 0.999),
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
            pred = model(image)
            loss = criterion(pred, target)
            optimizer.step(loss)
            if idx % 10 == 0:
                time_used = time.time() - start_time
                print('epoch: {}/{}, iter: {}/{}, time: {} s, loss: {}, lr: {}'.format(epoch, args.num_steps, idx, num_data, time_used, loss, lr))

        if epoch % args.save_pred_every == 0:
            print('store checkpoints ...')
            model.save(os.path.join(ckpt_path, f"VAN-{epoch}.pkl"))

if __name__ == '__main__':
    main()