import numpy as np
import cv2
from tqdm import tqdm
import torch
import pandas as pd
from core.utils.augmentor import Augmentor
from easydict import EasyDict
import os
from core.utils.utils import coords_grid
from core.utils.frame_utils import writeFlow
from core.utils.forward_warp import ForwardWarp
import argparse


def synthesis_data(args):
    data = pd.read_csv(args.csv)
    augmentor = Augmentor(args)
    forward_warp = ForwardWarp()

    H, W = args.image_size
    coords1 = coords_grid(1, H, W, 'cuda:0').contiguous().cuda()

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
        os.mkdir(os.path.join(args.save_dir, 'images'))
        os.mkdir(os.path.join(args.save_dir, 'flows'))
        os.mkdir(os.path.join(args.save_dir, 'masks'))

    for idx in tqdm(range(len(data))):
        item = data.iloc[idx]

        im = cv2.imread(os.path.join(args.root, item['fg_path']))
        bg = cv2.imread(os.path.join(args.root, item['bg_path']))
        tnf_type = item['tnf_type']
        theta = None
        if tnf_type == 'affine':
            theta = item[3:9].tolist()
        elif tnf_type == 'hom':
            theta = item[3:11].tolist()
        elif tnf_type == 'tps':
            theta = item[3:].tolist()
        
        fg = torch.from_numpy(im).permute([2, 0, 1])
        theta = torch.Tensor(theta)
        
        im_warp, grid_map, _, _ = augmentor(fg, tnf_type, theta)

        grid = coords_grid(1, H, W, grid_map.device)[0].permute(1,2,0)
        flow_gt = grid_map - grid

        out = forward_warp(coords1.cuda(), flow_gt[None].permute([0,3,1,2]).cuda())
        coords2 = out[0][0] / out[1][0]
        intensity = torch.norm(out[0][0], dim=0, p=0).cuda()
        mask = intensity != 0
        forward_flow = (coords2 - coords1) * mask

        warp_mask = (grid_map[:,:,0] >= 0) & (grid_map[:,:,0] < W) & \
                    (grid_map[:,:,1] >= 0) & (grid_map[:,:,1] < H)

        mask_for_blend = np.expand_dims(warp_mask, 2).repeat(3, 2)
        im_warp = im_warp.permute([1,2,0]) * mask_for_blend + bg * ~mask_for_blend
        im_warp = torch.clip(im_warp, 0, 255).numpy().astype(np.uint8)

        cv2.imwrite(os.path.join(args.save_dir, 'images/image_{:06d}_0.png'.format(idx)), im)
        cv2.imwrite(os.path.join(args.save_dir, 'images/image_{:06d}_1.png'.format(idx)), im_warp)
        cv2.imwrite(os.path.join(args.save_dir, 'masks/mask_{:06d}.png'.format(idx)), mask.cpu().numpy().astype(np.uint8))
        writeFlow(os.path.join(args.save_dir, 'flows/flow_{:06d}.flo'.format(idx)), forward_flow[0].permute([1,2,0]).cpu())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default='./data/MegaDepth_CAPS', type=str)
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument('--image_size', type=int, nargs='+', default=[480, 640])
    args = parser.parse_args()

    synthesis_data(args)
