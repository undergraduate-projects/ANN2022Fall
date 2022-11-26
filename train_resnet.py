import argparse
import numpy as np
import os
from tqdm import tqdm

from tensorboardX import SummaryWriter
from loss.criterion import CriterionDSN, CriterionOhemDSN
from dataset.ade_dataset import TrainDataset
from networks.ccnet import Seg_Model
import jittor as jt


IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)

BATCH_SIZE = 8
DATA_DIRECTORY = 'ADE20K'
IGNORE_LABEL = 255
LEARNING_RATE = 1e-2
MOMENTUM = 0.9
NUM_CLASSES = 19
NUM_STEPS = 40000
POWER = 0.9
RANDOM_SEED = 12345
RESTORE_FROM = os.path.join('pretrain', 'resnet101_base.pth')
SAVE_NUM_IMAGES = 2
SAVE_PRED_EVERY = 10000
SNAPSHOT_DIR = 'snapshots'
WEIGHT_DECAY = 0.0005

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_parser():
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--is-training", action="store_true",
                        help="Whether to updates the running means and variances during the training.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
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
    parser.add_argument("--random-mirror", action="store_true",
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random-scale", action="store_true",
                        help="Whether to randomly scale the inputs during the training.")
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
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--gpu", type=str, default='None',
                        help="choose gpu device.")
    parser.add_argument("--model", type=str, default='None',
                        help="choose model.")
    parser.add_argument("--num-workers", type=int, default=8,
                        help="choose the number of workers.")
    parser.add_argument("--recurrence", type=int, default=0,
                        help="choose the number of recurrence.")

    parser.add_argument("--ohem", type=str2bool, default='False',
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
    """Create the model and start the training."""
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

    model = Seg_Model(
        num_classes=args.num_classes, criterion=criterion,
        pretrained_model=args.restore_from, recurrence=args.recurrence
    )
    # seg_model.init_weights()

    # group weight and config optimizer
    optimizer = jt.optim.SGD(model.parameters(),
                            lr=args.learning_rate, 
                            momentum=args.momentum, 
                            weight_decay=args.weight_decay)
    # data loader
    train_loader = TrainDataset(shuffle=True, batch_size=args.batch_size)
    optimizer.zero_grad()
    os.makedirs(args.snapshot_dir, exist_ok=True)
    
    for epoch in tqdm(range(args.num_steps)):
        model.train()
        for idx, (image, target) in enumerate(train_loader):
            optimizer.zero_grad()
            lr = poly_lr_scheduler(optimizer, args.learning_rate, epoch * len(train_loader) + idx, args.num_steps * len(train_loader))
            image = image.float32()
            loss = model(image, target)
            optimizer.step(loss)
            print('epoch: {}, iter: {}, loss: {}, lr: {}'.format(epoch, idx, loss, lr))

        if epoch % args.save_pred_every == 0:
            print('taking snapshot ...')
            jt.save(model.state_dict(), os.path.join(args.snapshot_dir, f"CCNet-{epoch}.pth"))  

if __name__ == '__main__':
    main()