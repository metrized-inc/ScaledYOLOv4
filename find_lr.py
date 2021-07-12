import argparse
import torch
import torch.optim as optim
import torch.nn.functional as F
import wandb
import yaml
import math

from torch_lr_finder import LRFinder, TrainDataLoaderIter
from utils.datasets import create_dataloader
from models.yolo import Model
from utils.torch_utils import intersect_dicts
from torch.cuda import amp

import torch.optim.lr_scheduler as lr_scheduler
from utils.general import compute_loss, labels_to_class_weights

imgsz = 480
batch_size = 2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CustomTrainIter(TrainDataLoaderIter):
    def inputs_labels_from_batch(self, batch_data):
        imgs, targets, paths, _ = batch_data
        imgs = imgs.to(device, non_blocking=True).float() / 255.0  # uint8 to float32, 0-255 to 0.0-1.0
        targets = targets.to(device)
        return imgs, targets

def sample_lr(opt, hyp):
    # get data from .yaml file
    with open(opt.data) as f:
        data_dict = yaml.load(f, Loader=yaml.FullLoader)
    train_path = data_dict['train']
    nc = int(data_dict['nc'])
    names = data_dict['names']

    # load model
    ckpt = torch.load(opt.weights, map_location=device)  # load checkpoint
    model = Model(opt.cfg or ckpt['model'].yaml, ch=3, nc=nc).to(device)  # create
    exclude = ['anchor'] if opt.cfg else []  # exclude keys
    state_dict = ckpt['model'].float().state_dict()  # to FP32
    state_dict = intersect_dicts(state_dict, model.state_dict(), exclude=exclude)  # intersect
    model.load_state_dict(state_dict, strict=False)  # load

    # Model params
    hyp['cls'] *= nc / 80.  # scale coco-tuned hyp['cls'] to current dataset
    model.nc = nc  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    model.gr = 1.0  # giou loss ratio (obj_loss = 1.0 or giou)
    model.names = names
    gs = int(max(model.stride))  # grid size (max stride)

    # Dataset
    dataloader, dataset = create_dataloader(train_path, imgsz, batch_size, gs, opt, hyp=hyp, augment=True)
    custom_train_iter = CustomTrainIter(dataloader)

    # More model params
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device)  # attach class weights

    # Optimizer
    nbs = 64  # nominal batch size
    accumulate = max(round(nbs / batch_size), 1)  # accumulate loss before optimizing
    hyp['weight_decay'] *= batch_size * accumulate / nbs  # scale weight_decay

    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    for k, v in model.named_parameters():
        v.requires_grad = True
        if '.bias' in k:
            pg2.append(v)  # biases
        elif '.weight' in k and '.bn' not in k:
            pg1.append(v)  # apply weight decay
        else:
            pg0.append(v)  # all else

    optimizer = optim.SGD(pg0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)

    optimizer.add_param_group({'params': pg1, 'weight_decay': hyp['weight_decay']})  # add pg1 with weight_decay
    optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
    print('Optimizer groups: %g .bias, %g conv.weight, %g other' % (len(pg2), len(pg1), len(pg0)))
    del pg0, pg1, pg2

    # Loss (criterion)
    def loss_wrapper(p, targets):
        loss, loss_items = compute_loss(p, targets, model)
        return loss
    criterion = loss_wrapper

    run = wandb.init(project = 'ScaledYOLOv4', entity = 'michelle-aubin', job_type = 'lr_tuning')

    lr_finder = LRFinder(model, optimizer, criterion, device="cuda")
    lr_finder.range_test(custom_train_iter, start_lr=opt.start_lr, end_lr=opt.end_lr, num_iter=opt.num_iter)

    for x, y in zip(lr_finder.history["lr"], lr_finder.history["loss"]):
        wandb.log({"lr": x, "loss": y})

    run.finish()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        type=str,
        help=".yaml data file",
    )
    parser.add_argument('--hyp', type=str, default='', help='hyperparameters path, i.e. data/hyp.scratch.yaml')

    parser.add_argument(
        "--start_lr", type=float, default=1e-4, help="Start learning rate sampling"
    )

    parser.add_argument(
        "--end_lr", type=float, default=10, help="End learning rate sampling"
    )
    parser.add_argument('--cfg', type=str, default='', help='model.yaml path')

    parser.add_argument("--num_iter", type=int, default=500, help="Num of plot samples")

    parser.add_argument('--weights', type=str, default='yolov4-p5.pt', help='initial weights path')

    opt = parser.parse_args()

    opt.hyp = opt.hyp or ('data/hyp.finetune.yaml' if opt.weights else 'data/hyp.scratch.yaml')
    opt.single_cls = False

    with open(opt.hyp) as f:
        hyp = yaml.load(f, Loader=yaml.FullLoader)  # load hyps


    sample_lr(opt, hyp)


if __name__ == "__main__":
    main()