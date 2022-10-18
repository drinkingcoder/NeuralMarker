from __future__ import print_function, division

import sys
sys.path.append('./core')

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from core.biraft import BiRAFT
from core import datasets
from core.loss import compute_all_loss
import wandb
import random
import yaml
import time
import shutil
from termcolor import colored
from easydict import EasyDict
from torch.cuda.amp import GradScaler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import time
import evaluate
import warnings
warnings.filterwarnings('ignore')


def setup(args, rank, world_size):
    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = args.master_port

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def save_checkpoint(args, model, optimizer=None, step=None, epoch=None):
    CHECKPOINT_PATH = None
    if step is not None:
        CHECKPOINT_PATH = os.path.join(args.local_workspace, 'checkpoint_{:09d}.pth'.format(step))
        torch.save(model.state_dict(), CHECKPOINT_PATH)
        print(colored('[MODEL]: ', 'yellow') + 'Saving the model at iteration {:d}: {:s}'.format(
                step, os.path.basename(CHECKPOINT_PATH)))
    elif epoch is not None:
        CHECKPOINT_PATH = os.path.join(args.local_workspace, 'checkpoint_latest.pth')
        torch.save({'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch}, CHECKPOINT_PATH)
        print(colored('[MODEL]: ', 'yellow') + 'Saving the model at epoch {:d}: {:s}'.format(
                epoch, os.path.basename(CHECKPOINT_PATH)))
    else:
        raise "save model for a certain steps or epochs"
    

def load_checkpoint(args, model, optimizer):
    CHECKPOINT_PATH = os.path.join(args.local_workspace, 'checkpoint_latest.pth')
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    start_epoch = checkpoint['epoch']
    print(colored('[MODEL]: ', 'yellow') + 'Loading the model at epoch: {:d}'.format(start_epoch))

    return start_epoch + 1

def train(rank, world_size, args):
    print(f"Training on rank {rank}.")
    setup(args, rank, world_size)
    torch.cuda.set_device(rank)

    if rank == 0:
        wandb.init(project="NeuralMarker", entity='corr', config=args, name=args.experiment_name)

        wandb.define_metric('val_step')
        wandb.define_metric('validate/*', step='val_step')

    model = BiRAFT(args).to(rank)
    model = DDP(model, device_ids=[rank],
                broadcast_buffers=False,
                find_unused_parameters=True)

    train_loader = datasets.fetch_dataloader(args)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=args.epsilon)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, steps_per_epoch=len(train_loader), 
                                              epochs=args.epochs, pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')

    global_step = 1
    start_epoch = 1
    val_step = 1

    if args.resume:
        start_epoch = load_checkpoint(args, model, optimizer)
        global_step = (start_epoch - 1) * len(train_loader) + 1
        scheduler.last_epoch = global_step

    model.train()
    scaler = GradScaler(enabled=args.mixed_precision)

    tic = toc = time.time()
    for epoch_idx in range(start_epoch, args.epochs + 1):
        for step, input in enumerate(train_loader):
            iter_cost = max(0, time.time() - tic)
            tic = time.time()
            optimizer.zero_grad()
            for key in input.keys():
                if not isinstance(input[key], list):
                    input[key] = input[key].to(rank)

            image1, image2, image3 = input['im1'], input['im2'], input['im3']

            # predict flow
            # flow_ab means flow from a to b
            output_BA = output_AB = output_BB1 = output_B1B = 0
            if args.sed_loss:
                output_BA, output_AB = model(image2, image1, iters=args.iters)
            if args.tnf_loss:
                output_BB1, output_B1B = model(image2, image3, iters=args.iters)

            output = {
                'output_BA' : output_BA,
                'output_AB' : output_AB,
                'output_BB1': output_BB1,
                'output_B1B': output_B1B,
                }

            # compute loss
            loss = compute_all_loss(args, input, output)

            scaler.scale(loss['total_loss']).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            toc = time.time()

            if rank == 0:
                loss_log = {
                    'Step': global_step,
                    'Epoch': epoch_idx,
                    'lr': scheduler.get_lr()[0],
                    'batch_fps': (args.batch_size * args.world_size) / iter_cost,
                    'compute_cost(s)': toc - tic,
                    'iterate_cost(s)': iter_cost
                }
                for key in loss.keys():
                    loss_log.update({'Loss/' + key: loss[key]})
                wandb.log(loss_log)

            global_step += 1

            if global_step % args.save_ckpt_itr_freq == 0:
                if rank == 0:
                    save_checkpoint(args, model, step=global_step)
                    model.eval()
                    if args.validate:
                        evaluate.validate(args, model, val_step, 'synthesis')
                        evaluate.visualization(args, model, val_step, 'visualization')
                    val_step += 1
                    model.train()

        if rank == 0:
            save_checkpoint(args, model, optimizer, epoch=epoch_idx)
            model.eval()
            if args.validate:
                evaluate.validate(args, model, val_step, 'synthesis')
                evaluate.visualization(args, model, val_step, 'visualization')
            model.train()

    cleanup()

def config_prase(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if args.resume is None:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)

        tmp_config_path = os.path.join(os.path.dirname(args.config), "params_tmp.yaml")

        config.update({'resume' : args.resume})
        config.update({'seed' : args.seed})
        config.update({'debug'  : args.debug})
        config.update({'world_size'  : torch.cuda.device_count()})

        if not os.path.exists(config["workspace"]):
            os.mkdir(config["workspace"])

        config['timestamp'] = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        workspace = os.path.join(config["workspace"], config['timestamp'])
        if not os.path.exists(workspace):
            os.mkdir(workspace)
        config["local_workspace"] = workspace

        with open(tmp_config_path, "w") as f:
            print("Dumping extra config file...")
            yaml.dump(config, f)
        
        shutil.copy(tmp_config_path, os.path.join(workspace, "params.yaml"))
    else:
        config_path = os.path.join(args.workspace, args.resume, "params.yaml")
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        config.update({'resume' : args.resume})

    config.update({'experiment_name': ''})
    return EasyDict(config)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./configs/params_default.yaml", type=str)
    parser.add_argument("--workspace", type=str, default='./snapshot', required=False)
    parser.add_argument("--resume", type=str, default=None, required=False)
    parser.add_argument("--debug", action='store_true')
    parser.add_argument("--seed", type=int, default=2022, required=False)
    args = parser.parse_args()

    config = config_prase(args)

    print(colored('\nxxxxxxx', 'yellow'))
    config.experiment_name = input(colored('=====> INPUT THE EXPERIMENT NAME: ', 'yellow'))
    print(colored('xxxxxxx\n', 'yellow'))

    # DDP deploy
    world_size = torch.cuda.device_count()
    mp.spawn(train, args=(world_size, config), nprocs=world_size, join=True)