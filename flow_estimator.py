from core.utils.utils import InputPadder, coords_grid
import torch
import cv2
import numpy as np
from core.raft import RAFT
import os 

class Flow_estimator():
    def __init__(self, args):

        self.args = args 
        if not os.path.exists(args.model):            
            raise FileExistsError(args.model)       
        # self.network = torch.nn.DataParallel(RAFT(args)).cuda()
        # self.network.load_state_dict(torch.load(args.model)["model"], False)

        self.network = RAFT(args).cuda()
        self.network.load_state_dict(torch.load(args.model), False)

        #for parameters in self.network.parameters():
        #    print(parameters)

    def estimate(self, im1, im2):
        '''
        Input:
            im1, im2: HxWx3 numpy (it is not necessary for im1 and im2 to be same shape, because im2 will be resized to
            the shape of im1)
        Output:
            flow: Bx2xHxW torch.Tensor
        '''
        h1, w1 = im1.shape[:2]
        h2, w2 = im2.shape[:2]

        im2_resized = cv2.resize(im2, (w1, h1))
        im1, im2 = self.image_pair_process(im1, im2_resized)
        padder = InputPadder(im1.shape)
        im1, im2 = padder.pad(im1[None].cuda(), im2[None].cuda())
        # print(im1)
        flow = self.inference_model(im1, im2)[0]
        # print(flow)
        
        flow = padder.unpad(flow).permute(1, 2, 0).repeat(1, 1, 1, 1).permute([0, 3, 1, 2])
        grid = coords_grid(1, h1, w1).cuda()
        flow += grid
        flow[:,0] = flow[:,0] * w2 / w1
        flow[:,1] = flow[:,1] * h2 / h1
        flow -= grid
        

        return flow

    def image_pair_process(self, img1, img2):
        if len(img1.shape) == 2:
            img1 = np.tile(img1[..., None], (1, 1, 3))
            img2 = np.tile(img2[..., None], (1, 1, 3))

        img1 = img1.astype(np.uint8)[..., :3]
        img2 = img2.astype(np.uint8)[..., :3]

        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        return img1, img2

    @torch.no_grad()
    def inference_model(self, im1, im2):
        self.network.eval()
        output = self.network(im1, im2, iters=self.args.iters, test_mode=True)      
        '''
        if output is tuple:
            print(len(output))
        if output[1] is tuple:
            print(len(output[1]))
        print(output[1][0].shape)
        '''
        return output[1]

