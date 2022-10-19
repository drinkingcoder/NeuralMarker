import sys

from core.utils.utils import image_flow_warp
sys.path.append('./core')
import os
import torch
from core.datasets import ValidateData
import numpy as np
from termcolor import colored
import wandb
import argparse
from core.raft import RAFT
from tqdm import tqdm

@torch.no_grad()
def validate(args, model, step=None, split=None):
    model.eval()
    dataset_val = ValidateData(args, split)
    epe_list = []
    epes = []
    for val_id in tqdm(range(len(dataset_val))):
        im1, im2, flow_gt, valid_mask = dataset_val[val_id]
        output = model(im1[None].cuda(), im2[None].cuda(), iters=args.iters, test_mode=True)
        flow_pr = output[1]
        epe = torch.sum((flow_pr[0].to(flow_gt.device) - flow_gt)**2, dim=0)[valid_mask].sqrt()
        epe_list.append(epe.view(-1).cpu().tolist())
        epes += [epe.mean()]

    epe_all = np.concatenate(epe_list)
    epe = np.mean(epe_all)
    px1 = np.mean(epe_all<1)
    px3 = np.mean(epe_all<3)
    px5 = np.mean(epe_all<5)

    print(colored('[Validation %s]: ', 'yellow') % (split) +
                  "EPE: %f, 1px: %f, 3px: %f, 5px: %f" % (np.mean(epes), px1, px3, px5))

    if step:
        wandb.log({
            'val_step'      : step,
            split + '/epe'  : epe,
            split + '/1px'  : px1,
            split + '/3px'  : px3,
            split + '/5px'  : px5,
        })


@torch.no_grad()
def visualization(args, model, step=None, split=None):
    model.eval()
    dataset_val = ValidateData(args, split)
    for val_id in range(len(dataset_val)):
        im1, im2, flow_gt, valid_mask = dataset_val[val_id]
        output = model(im1[None].cuda(), im2[None].cuda(), iters=args.iters, test_mode=True)
        flow_pr = output[1]

        im1_vis = im1.permute([1,2,0]).cpu().numpy().astype(np.uint8)
        im2_vis = im2.permute([1,2,0]).cpu().numpy().astype(np.uint8)
        im2_warp_vis = image_flow_warp(im2_vis, flow_pr[0].permute([1,2,0]))

        im_all = np.concatenate([im1_vis, im2_vis, im2_warp_vis], axis=1)[:,:,::-1]

        im_all = wandb.Image(im_all, caption='step: {:d}'.format(step))
        wandb.log({'vis/{:d}'.format(val_id): im_all})
        

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    parser = argparse.ArgumentParser()
    parser.add_argument("--mixed_precision", type=str, default=True, required=False)
    parser.add_argument("--small", type=str, default=False, required=False)
    parser.add_argument("--iters", type=int, default=12, required=False)
    parser.add_argument("--dim_corr", type=int, default=192, required=False)
    parser.add_argument("--dim_corr_coarse", type=int, default=64, required=False)
    parser.add_argument("--dim_corr_all", type=int, default=256, required=False)
    parser.add_argument("--model", type=str, default='./models/cnn_full.biraft.pth')
    parser.add_argument("--fnet", type=str, default='CNN')
    parser.add_argument("--validate_data_dir", type=str, default='./data/flyingmarkers')
    parser.add_argument("--split", type=str, default='test')
    parser.add_argument("--raw_model", action='store_true')
    args = parser.parse_args()

    if args.raw_model:
        model = RAFT(args).cuda()
        model.load_state_dict(torch.load(args.model), False)
    else:
        model = torch.nn.DataParallel(RAFT(args)).cuda()
        model.load_state_dict(torch.load(args.model)["model"], False)

    validate(args, model, step=None, split=args.split)