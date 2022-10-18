from ast import For
import numpy as np
from pyparsing import Each
import torch
import torch.utils.data as data
from torch.utils.data.distributed import DistributedSampler
import os
import os.path as osp
import time
from tqdm import tqdm
import cv2
import random
from core.utils.utils import entry_convert, fundamental_matrix_gen
from core.utils.frame_utils import readFlow
from core.utils.augmentor import Augmentor
import pandas as pd
from termcolor import colored
import warnings
import random
from glob import glob
from core.utils.utils import coords_grid
from core.utils.forward_warp import ForwardWarp

warnings.filterwarnings("ignore") 

class Image:
    def __init__(self, path, Tcw, K):
        self.im_path = path
        self.Tcw = Tcw
        self.K = K

class PoseDataset(data.Dataset):
    def __init__(self, args):
        self.init_seed = False

        self.args = args
        self.entrydata_list = []
        self.dataset = self.args.dataset
        self.transformation = Augmentor(self.args)
        self.forward_warping = ForwardWarp()
        self.bg_dataset = SequenceDataset(args, 'train')

    def __getitem__(self, index):

        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        Im1, Im2 = self.entrydata_list[index]

        im1 = cv2.imread(Im1.im_path)
        im2 = cv2.imread(Im2.im_path)

        if im1.ndim == 2:
            im1 = np.tile(im1[..., None], (1, 1, 3))
            im2 = np.tile(im2[..., None], (1, 1, 3))
        else:
            im1 = im1[..., :3]
            im2 = im2[..., :3]

        '''
        crop the image
        '''
        ht, wd = im1.shape[:2]
        crop_size = self.args.image_size

        assert (crop_size[1] % 8 == 0 and crop_size[0] % 8 == 0)

        im1 = cv2.resize(im1, (crop_size[1], crop_size[0]), interpolation=cv2.INTER_LINEAR)
        im2 = cv2.resize(im2, (crop_size[1], crop_size[0]), interpolation=cv2.INTER_LINEAR)

        scale = np.diag([crop_size[1] / wd, crop_size[0] / ht, 1.0])

        K1s = scale.dot(Im1.K)
        K2s = scale.dot(Im2.K)

        F = fundamental_matrix_gen(Im1.Tcw, Im2.Tcw, K1s, K2s)

        im1 = torch.from_numpy(im1).permute(2, 0, 1).float()
        im2 = torch.from_numpy(im2).permute(2, 0, 1).float()

        '''
        transformation 
        '''

        im3, backward_map, tnf_type, theta = self.transformation(im2, self.args.tnf_type)

        backward_map = backward_map.permute([2, 0, 1])
        backward_mask = (backward_map[0, :, :] >= 0) & (backward_map[0, :, :] < wd) & \
                        (backward_map[1, :, :] >= 0) & (backward_map[1, :, :] < ht)
        
        coords1 = coords_grid(1, ht, wd, device='cpu').contiguous()
        backward_flow = backward_map[None] - coords1

        out = self.forward_warping(coords1.cuda(), backward_flow.cuda())
        forward_map = out[0][0] / (out[1][0] + (1e-6))
        forward_mask = torch.norm(out[0][0], dim=0, p=0) != 0
        
        bg, bg_path = self.bg_dataset.random_select()
        mask_for_blend = np.expand_dims(backward_mask, 2).repeat(3, 2)
        im3 = im3.permute([1,2,0]) * mask_for_blend + bg * ~mask_for_blend
        im3 = torch.clip(im3, 0, 255).permute([2,0,1]).numpy().astype(np.uint8)

        return {
            'im1'                   : np.ascontiguousarray(im1),
            'intrinsic1'            : np.ascontiguousarray(K1s),
            'Tcw1'                  : np.ascontiguousarray(Im1.Tcw),
            'im2'                   : np.ascontiguousarray(im2),
            'intrinsic2'            : np.ascontiguousarray(K2s),
            'Tcw2'                  : np.ascontiguousarray(Im2.Tcw),
            'fundamental_matrix'    : np.ascontiguousarray(F),
            'im3'                   : np.ascontiguousarray(im3),
            'forward_map'           : np.ascontiguousarray(forward_map.cpu()),
            'forward_mask'          : np.ascontiguousarray(forward_mask.cpu()),
            'backward_map'          : np.ascontiguousarray(backward_map.cpu()),
            'backward_mask'         : np.ascontiguousarray(backward_mask.cpu()),
            'im1_path'              : Im1.im_path,
            'im2_path'              : Im2.im_path,
            'bg_path'               : bg_path,
            'transformation_type'   : tnf_type,
            # 'theta'                 : theta           
        }

    def __len__(self):
        return len(self.entrydata_list)

class MegaDepth_CAPS(PoseDataset):
    def __init__(self, args, split='train'):
        super(MegaDepth_CAPS, self).__init__(args)
        root = os.path.join(args.training_data_dir, args.dataset)
        if not os.path.exists(root):
            raise colored('[Error: ]', 'red') + 'file path {:s} is not exist!'.format(root)

        scenes_list = sorted(os.listdir(osp.join(root, split)))
        scenes_list = [scene for scene in scenes_list if osp.isdir(osp.join(root, split, scene))]
        
        dense_list = []
        for scene in scenes_list:
            denses = os.listdir(osp.join(root, split, scene))
            dense_list += [osp.join(root, split, scene, dense) for dense in denses if osp.isdir(osp.join(root, split, scene, dense))]

        if self.args.debug:
            dense_list = dense_list[:1]

        start_time = time.time()
        for dense_path in tqdm(dense_list, ncols=70):
            im_path = osp.join(dense_path, 'aligned', 'images')
            
            # ignore empty file's warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pairs_list = np.loadtxt(osp.join(dense_path, 'aligned', 'pairs.txt'), dtype=str)
                pose_list  = np.loadtxt(osp.join(dense_path, 'aligned', 'img_cam.txt'), dtype=str)
            
            pose_map = {}
            for item in pose_list:
                pose_map.update({item[0]: item[1:].astype(np.float32)})
            
            if len(pairs_list) > self.args.sample_maxlen:
                index = np.arange(len(pairs_list))
                np.random.shuffle(index)
                pairs_list = np.array(pairs_list)[index[:self.args.sample_maxlen]].tolist()

            for im1_name, im2_name in pairs_list:
                im1_path = osp.join(im_path, im1_name)
                im2_path = osp.join(im_path, im2_name)
                Tcw1, K1 = entry_convert(pose_map[im1_name])
                Tcw2, K2 = entry_convert(pose_map[im2_name])

                # Avoid the different path map to the same image and pose
                if np.linalg.norm(Tcw1 - Tcw2) < self.args.epsilon:
                    continue 
                
                im1 = Image(im1_path, Tcw1.astype(np.float32), K1.astype(np.float32))
                im2 = Image(im2_path, Tcw2.astype(np.float32), K2.astype(np.float32))
                
                self.entrydata_list += [[im1, im2]]
        end_time = time.time()
        
        print('\nLoading dataset cost {:.3f}s, data size {} pairs'.format(end_time - start_time, len(self.entrydata_list)))


class SequenceDataset(data.Dataset):
    def __init__(self, args, split='train'):
        root = os.path.join(args.training_data_dir, args.dataset, split)
        if not os.path.exists(root):
            raise colored('[Error: ]', 'red') + 'file path {:s} is not exist!'.format(root)

        self.images_list = glob(os.path.join(root,'*/*/*/*/*.jpg'))

    def random_select(self):
        index = random.randint(0, len(self.images_list)-1)
        im_path = self.images_list[index]
        im = cv2.imread(im_path, cv2.IMREAD_UNCHANGED)
        return im, im_path

    def __getitem__(self, index):
        pass

    def __len___(self):
        return len(self.images_list)
        
class ValidateData(data.Dataset):
    def __init__(self, args, split):
        self.root = os.path.join(args.validate_data_dir, split)
        if not os.path.exists(self.root):
            raise colored('[Error: ]', 'red') + 'file path {:s} is not exist!'.format(self.root)

        self.images_dir = os.path.join(self.root, 'images')
        self.flows_dir = os.path.join(self.root, 'flows')
        self.masks_dir = os.path.join(self.root, 'masks')


    def __getitem__(self, index):
        im1 = cv2.imread(os.path.join(self.images_dir, 'image_{:06d}_0.png'.format(index)))
        im2 = cv2.imread(os.path.join(self.images_dir, 'image_{:06d}_1.png'.format(index)))
        flow = readFlow(os.path.join(self.flows_dir, 'flow_{:06d}.flo'.format(index)))

        H, W = im1.shape[:2]

        im1 = torch.from_numpy(im1).permute([2, 0, 1])
        im2 = torch.from_numpy(im2).permute([2, 0, 1])
        flow = torch.from_numpy(flow).permute([2, 0, 1])
   
        valid_mask = np.ones([H, W]).astype(bool)
        if os.path.exists(self.masks_dir):
            valid_mask = cv2.imread(os.path.join(self.masks_dir, 'mask_{:06d}.png'.format(index)), cv2.IMREAD_UNCHANGED).astype(bool)
        return im1, im2, flow, valid_mask

    def __len__(self):
        return len(os.listdir(self.flows_dir))

def fetch_dataloader(args, split='train'):
    print(colored('[Data]: ', 'yellow') + 'Loading MegaDepth CAPS')
    dataset = MegaDepth_CAPS(args, split)

    if args.debug_data:
        dataloader = data.DataLoader(
            dataset, 
            batch_size=args.batch_size, 
            pin_memory=False, 
            shuffle=True, 
            num_workers=args.num_workers, 
            drop_last=True)
    else:
        sampler = DistributedSampler(dataset)
        dataloader = data.DataLoader(
            dataset, 
            batch_size=args.batch_size, 
            sampler=sampler)

    return dataloader


if __name__=='__main__':
    from easydict import EasyDict
    args = EasyDict({
        'batch_size'    : 1,
        'num_workers'   : 8,
        'dataset'       : 'MegaDepth_CAPS',
        'data_dir'      : '../data/',
        'image_size'    : [480, 640],
        'debug'         : True,
        'sample_maxlen' : 5000,
        'epsilon'       : 1e-8,
        'tnf_type'      : 'random'
    })

    dataloader = fetch_dataloader(args)

    print('dataset size: ', len(dataloader))