import argparse
import torch
import torch.optim as optim
import torch.nn.functional as F
import wandb
import yaml

from torch_lr_finder import LRFinder
from utils.datasets import create_dataloader
from models.yolo import Model
from utils.torch_utils import intersect_dicts

imgsz = 480
batch_size = 8
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def sample_lr(opt, hyp):
    # get data from .yaml file
    with open(opt.data) as f:
        data_dict = yaml.load(f, Loader=yaml.FullLoader)
    train_path = data_dict['train']
    nc = int(data_dict['nc'])

    # load model
    ckpt = torch.load(opt.weights, map_location=device)  # load checkpoint
    model = Model(opt.cfg or ckpt['model'].yaml, ch=3, nc=nc).to(device)  # create
    exclude = ['anchor'] if opt.cfg else []  # exclude keys
    state_dict = ckpt['model'].float().state_dict()  # to FP32
    state_dict = intersect_dicts(state_dict, model.state_dict(), exclude=exclude)  # intersect
    model.load_state_dict(state_dict, strict=False)  # load

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

    # if opt.adam:
    #     optimizer = optim.Adam(pg0, lr=hyp['lr0'], betas=(hyp['momentum'], 0.999))  # adjust beta1 to momentum
    # else:
    optimizer = optim.SGD(pg0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)

    optimizer.add_param_group({'params': pg1, 'weight_decay': hyp['weight_decay']})  # add pg1 with weight_decay
    optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
    print('Optimizer groups: %g .bias, %g conv.weight, %g other' % (len(pg2), len(pg1), len(pg0)))
    del pg0, pg1, pg2

    dataloader, dataset = create_dataloader(train_path, imgsz, batch_size, gs, opt, hyp=hyp, augment=True)

    run = wandb.init(
        project=settings["project"],
        entity=settings["entity"],
        job_type="lr tuning",
    )


    # Choose optimizer settings
    optimizer = optim.SGD(model.parameters(), lr=args.start_lr)
    criterion = eval(settings["loss_function"])  # Loss function
    lr_finder = LRFinder(model, optimizer, criterion, device="cuda")
    lr_finder.range_test(dataloader, end_lr=args.end_lr, num_iter=args.num_iter)

    for x, y in zip(lr_finder.history["lr"], lr_finder.history["loss"]):
        wandb.log({"lr": x, "loss": y})

    # Save the settings file 
    logSettingsFile(wandb.run.dir, args.settings_path, settings)

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