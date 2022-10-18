# code from https://github.com/ignacio-rocco/cnngeometric_pytorch.git


from __future__ import print_function, division
import numpy as np
import torch
from torch.nn.modules.module import Module
from torch.autograd import Variable
import torch.nn.functional as F
from core.utils.utils import expand_dim
import random

class ComposedGeometricTnf(object):
    """
    Composed geometric transfromation (affine+tps)
    """

    def __init__(self, tps_grid_size=3, tps_reg_factor=0, out_h=240, out_w=240,
                 offset_factor=1.0,
                 padding_crop_factor=None,
                 use_cuda=True):
        self.padding_crop_factor = padding_crop_factor

        self.affTnf = GeometricTnf(out_h=out_h, out_w=out_w,
                                   geometric_model='affine',
                                   offset_factor=offset_factor if padding_crop_factor is None else padding_crop_factor,
                                   use_cuda=use_cuda)

        self.tpsTnf = GeometricTnf(out_h=out_h, out_w=out_w,
                                   geometric_model='tps',
                                   tps_grid_size=tps_grid_size,
                                   tps_reg_factor=tps_reg_factor,
                                   offset_factor=offset_factor if padding_crop_factor is None else 1.0,
                                   use_cuda=use_cuda)

    def __call__(self, image_batch, theta_aff, theta_aff_tps, use_cuda=True):
        sampling_grid_aff = self.affTnf(image_batch=None,
                                        theta_batch=theta_aff.view(-1, 2, 3),
                                        return_sampling_grid=True,
                                        return_warped_image=False)

        sampling_grid_aff_tps = self.tpsTnf(image_batch=None,
                                            theta_batch=theta_aff_tps,
                                            return_sampling_grid=True,
                                            return_warped_image=False)

        if self.padding_crop_factor is not None:
            sampling_grid_aff_tps = sampling_grid_aff_tps * self.padding_crop_factor

        # put 1e10 value in region out of bounds of sampling_grid_aff
        in_bound_mask_aff = ((sampling_grid_aff[:, :, :, 0] > -1) * (sampling_grid_aff[:, :, :, 0] < 1) * (
                    sampling_grid_aff[:, :, :, 1] > -1) * (sampling_grid_aff[:, :, :, 1] < 1)).unsqueeze(3)
        in_bound_mask_aff = in_bound_mask_aff.expand_as(sampling_grid_aff)
        sampling_grid_aff = torch.mul(in_bound_mask_aff.float(), sampling_grid_aff)
        sampling_grid_aff = torch.add((in_bound_mask_aff.float() - 1) * (1e10), sampling_grid_aff)

        # compose transformations
        sampling_grid_aff_tps_comp = F.grid_sample(sampling_grid_aff.transpose(2, 3).transpose(1, 2),
                                                   sampling_grid_aff_tps, align_corners=True).transpose(1, 2).transpose(2, 3)

        # put 1e10 value in region out of bounds of sampling_grid_aff_tps_comp
        in_bound_mask_aff_tps = ((sampling_grid_aff_tps[:, :, :, 0] > -1) * (sampling_grid_aff_tps[:, :, :, 0] < 1) * (
                    sampling_grid_aff_tps[:, :, :, 1] > -1) * (sampling_grid_aff_tps[:, :, :, 1] < 1)).unsqueeze(3)
        in_bound_mask_aff_tps = in_bound_mask_aff_tps.expand_as(sampling_grid_aff_tps_comp)
        sampling_grid_aff_tps_comp = torch.mul(in_bound_mask_aff_tps.float(), sampling_grid_aff_tps_comp)
        sampling_grid_aff_tps_comp = torch.add((in_bound_mask_aff_tps.float() - 1) * (1e10), sampling_grid_aff_tps_comp)

        # sample transformed image
        warped_image_batch = F.grid_sample(image_batch, sampling_grid_aff_tps_comp, align_corners=True)

        return warped_image_batch


class GeometricTnf(object):
    """
    Geometric transfromation to an image batch (wrapped in a PyTorch Variable)
    ( can be used with no transformation to perform bilinear resizing )
    """

    def __init__(self, geometric_model='affine', tps_grid_size=3, tps_reg_factor=0, out_h=240, out_w=240,
                 offset_factor=None, use_cuda=True):
        self.out_h = out_h
        self.out_w = out_w
        self.geometric_model = geometric_model
        self.use_cuda = use_cuda
        self.offset_factor = offset_factor

        if geometric_model == 'affine' and offset_factor is None:
            self.gridGen = AffineGridGen(out_h=out_h, out_w=out_w, use_cuda=use_cuda)
        elif geometric_model == 'affine' and offset_factor is not None:
            self.gridGen = AffineGridGenV2(out_h=out_h, out_w=out_w, use_cuda=use_cuda)
        elif geometric_model == 'hom':
            self.gridGen = HomographyGridGen(out_h=out_h, out_w=out_w, use_cuda=use_cuda)
        elif geometric_model == 'tps':
            self.gridGen = TpsGridGen(out_h=out_h, out_w=out_w, grid_size=tps_grid_size,
                                      reg_factor=tps_reg_factor, use_cuda=use_cuda)
        if offset_factor is not None:
            self.gridGen.grid_X = self.gridGen.grid_X / offset_factor
            self.gridGen.grid_Y = self.gridGen.grid_Y / offset_factor

        self.theta_identity = torch.Tensor(np.expand_dims(np.array([[1, 0, 0], [0, 1, 0]]), 0).astype(np.float32))
        if use_cuda:
            self.theta_identity = self.theta_identity.cuda()

    def __call__(self, image_batch, theta_batch=None, out_h=None, out_w=None, return_warped_image=True,
                 return_sampling_grid=False, padding_factor=1.0, crop_factor=1.0):
    
        crop_factor = random.uniform(0.5, 5)
        if image_batch is None:
            b = 1
        else:
            b = image_batch.size(0)
        if theta_batch is None:
            theta_batch = self.theta_identity
            theta_batch = theta_batch.expand(b, 2, 3).contiguous()
            theta_batch = Variable(theta_batch, requires_grad=False)
        # check if output dimensions have been specified at call time and have changed
        if (out_h is not None and out_w is not None) and (out_h != self.out_h or out_w != self.out_w):
            if self.geometric_model == 'affine':
                gridGen = AffineGridGen(out_h, out_w, use_cuda=self.use_cuda)
            elif self.geometric_model == 'hom':
                gridGen = HomographyGridGen(out_h, out_w, use_cuda=self.use_cuda)
            elif self.geometric_model == 'tps':
                gridGen = TpsGridGen(out_h, out_w, use_cuda=self.use_cuda)
        else:
            gridGen = self.gridGen
        sampling_grid = gridGen(theta_batch)

        # rescale grid according to crop_factor and padding_factor
        if padding_factor != 1 or crop_factor != 1:
            sampling_grid = sampling_grid * (padding_factor * crop_factor)
        # rescale grid according to offset_factor
        if self.offset_factor is not None:
            sampling_grid = sampling_grid * self.offset_factor

        if return_sampling_grid and not return_warped_image:
            return sampling_grid
        # sample transformed image
        warped_image_batch = F.grid_sample(image_batch, sampling_grid, align_corners=True)

        if return_sampling_grid and return_warped_image:
            return (warped_image_batch, sampling_grid)

        return warped_image_batch


class SynthPairTnf(object):
    """
    Generate a synthetically warped training pair using an affine transformation.
    """

    def __init__(self, use_cuda=True, geometric_model='affine', crop_factor=9 / 16, output_size=(240, 240),
                 padding_factor=0.5):
        assert isinstance(use_cuda, (bool))
        assert isinstance(crop_factor, (float))
        assert isinstance(output_size, (tuple))
        assert isinstance(padding_factor, (float))
        self.use_cuda = use_cuda
        self.crop_factor = crop_factor
        self.padding_factor = padding_factor
        self.out_h, self.out_w = output_size
        self.rescalingTnf = GeometricTnf('affine', out_h=self.out_h, out_w=self.out_w,
                                         use_cuda=self.use_cuda)
        self.geometricTnf = GeometricTnf(geometric_model, out_h=self.out_h, out_w=self.out_w,
                                         use_cuda=self.use_cuda)

    def __call__(self, batch):
        image_batch, theta_batch = batch['image'], batch['theta']
        if self.use_cuda:
            image_batch = image_batch.cuda()
            theta_batch = theta_batch.cuda()

        b, c, h, w = image_batch.size()

        # generate symmetrically padded image for bigger sampling region
        # image_batch = self.symmetricImagePad(image_batch, self.padding_factor)
        image_batch = self.expandImagePad(image_batch, self.padding_factor)


        # convert to variables
        image_batch = Variable(image_batch, requires_grad=False)
        theta_batch = Variable(theta_batch, requires_grad=False)

        # get cropped image
        cropped_image_batch, cropped_grid = self.rescalingTnf(image_batch=image_batch,
                                                              theta_batch=None,
                                                              padding_factor=self.padding_factor,
                                                              crop_factor=self.crop_factor,
                                                              return_sampling_grid=True)  # Identity is used as no theta given

        # get transformed image
        warped_image_batch, warped_grid = self.geometricTnf(image_batch=image_batch,
                                                            theta_batch=theta_batch,
                                                            padding_factor=self.padding_factor,
                                                            crop_factor=self.crop_factor,
                                                            return_sampling_grid=True)  # Identity is used as no theta given
            
        valid_mask = (warped_grid[:,:,0] >= 0) & (warped_grid[:,:,0] < w) & (warped_grid[:,:,1] >= 0) & (warped_grid[:,:,1] < h)

        return {'source_image': cropped_image_batch,
                'target_image': warped_image_batch,
                'cropped_grid': cropped_grid,
                'warped_grid': warped_grid,
                'valid_mask': valid_mask}

    def symmetricImagePad(self, image_batch, padding_factor):
        b, c, h, w = image_batch.size()
        pad_h, pad_w = int(h * padding_factor), int(w * padding_factor)
        idx_pad_left = torch.LongTensor(range(pad_w - 1, -1, -1))
        idx_pad_right = torch.LongTensor(range(w - 1, w - pad_w - 1, -1))
        idx_pad_top = torch.LongTensor(range(pad_h - 1, -1, -1))
        idx_pad_bottom = torch.LongTensor(range(h - 1, h - pad_h - 1, -1))
        if self.use_cuda:
            idx_pad_left = idx_pad_left.cuda()
            idx_pad_right = idx_pad_right.cuda()
            idx_pad_top = idx_pad_top.cuda()
            idx_pad_bottom = idx_pad_bottom.cuda()
        image_batch = torch.cat((image_batch.index_select(3, idx_pad_left), image_batch,
                                 image_batch.index_select(3, idx_pad_right)), 3)
        image_batch = torch.cat((image_batch.index_select(2, idx_pad_top), image_batch,
                                 image_batch.index_select(2, idx_pad_bottom)), 2)
        return image_batch

    def expandImagePad(self, image_batch, padding_factor):
        b, c, h, w = image_batch.size()
        expand_image = torch.zeros([b, c, h * 2, w * 2])
        expand_image[:, :, h//2: h//2 + h, w//2: w//2 + w] = image_batch

        return expand_image



class SynthTwoStageTnf(SynthPairTnf):
    def __init__(self, use_cuda=True, crop_factor=9 / 16, output_size=(240, 240), padding_factor=0.5):
        super().__init__(use_cuda=use_cuda)
        #        self.aff_reorder_idx=torch.LongTensor([3,2,5,1,0,4])
        self.geometricTnf = ComposedGeometricTnf(padding_crop_factor=padding_factor * crop_factor,
                                                 use_cuda=self.use_cuda)

    def __call__(self, batch):
        image_batch, theta_batch = batch['image'], batch['theta']
        #        theta_aff=torch.index_select(theta_batch[:,:6],1,self.aff_reorder_idx)
        theta_aff = theta_batch[:, :6].contiguous()
        theta_tps = theta_batch[:, 6:]

        if self.use_cuda:
            image_batch = image_batch.cuda()
            theta_aff = theta_aff.cuda()
            theta_tps = theta_tps.cuda()

        b, c, h, w = image_batch.size()

        # generate symmetrically padded image for bigger sampling region
        image_batch = self.symmetricImagePad(image_batch, self.padding_factor)

        # convert to variables
        image_batch = Variable(image_batch, requires_grad=False)
        theta_aff = Variable(theta_aff, requires_grad=False)
        theta_tps = Variable(theta_tps, requires_grad=False)

        # get cropped image
        cropped_image_batch = self.rescalingTnf(image_batch=image_batch,
                                                theta_batch=None,
                                                padding_factor=self.padding_factor,
                                                crop_factor=self.crop_factor)  # Identity is used as no theta given
        # get transformed image
        warped_image_batch = self.geometricTnf(image_batch=image_batch,
                                               theta_aff=theta_aff,
                                               theta_aff_tps=theta_tps)

        return {'source_image': cropped_image_batch, 'target_image': warped_image_batch, 'theta_GT_aff': theta_aff,
                'theta_GT_tps': theta_tps}


class SynthTwoStageTwoPairTnf(SynthPairTnf):
    def __init__(self, use_cuda=True, crop_factor=9 / 16, output_size=(240, 240), padding_factor=0.5):
        super().__init__(use_cuda=use_cuda)
        #        self.aff_reorder_idx=torch.LongTensor([3,2,5,1,0,4])
        self.geometricTnf = ComposedGeometricTnf(padding_crop_factor=padding_factor * crop_factor,
                                                 use_cuda=self.use_cuda)
        self.affTnf = GeometricTnf(geometric_model='affine', out_h=self.out_h, out_w=self.out_w,
                                   use_cuda=self.use_cuda)
        self.tpsTnf = GeometricTnf(geometric_model='tps', out_h=self.out_h, out_w=self.out_w,
                                   use_cuda=self.use_cuda)

    def __call__(self, batch):
        image_batch, theta_batch = batch['image'], batch['theta']
        theta_aff = theta_batch[:, :6].contiguous()
        #        theta_aff=torch.index_select(theta_batch[:,:6],1,self.aff_reorder_idx)
        theta_tps = theta_batch[:, 6:]

        if self.use_cuda:
            image_batch = image_batch.cuda()
            theta_aff = theta_aff.cuda()
            theta_tps = theta_tps.cuda()

        b, c, h, w = image_batch.size()

        # generate symmetrically padded image for bigger sampling region
        image_batch = self.symmetricImagePad(image_batch, self.padding_factor)

        # convert to variables
        image_batch = Variable(image_batch, requires_grad=False)
        theta_aff = Variable(theta_aff, requires_grad=False)
        theta_tps = Variable(theta_tps, requires_grad=False)

        # get cropped image
        cropped_image_batch = self.rescalingTnf(image_batch=image_batch,
                                                theta_batch=None,
                                                padding_factor=self.padding_factor,
                                                crop_factor=self.crop_factor)  # Identity is used as no theta given
        # get transformed image
        target_image_tps = self.geometricTnf(image_batch=image_batch,
                                             theta_aff=theta_aff,
                                             theta_aff_tps=theta_tps)

        target_image_aff = self.affTnf(image_batch=image_batch,
                                       theta_batch=theta_aff,
                                       padding_factor=self.padding_factor,
                                       crop_factor=self.crop_factor)

        source_image_tps = self.affTnf(image_batch=cropped_image_batch,
                                       theta_batch=theta_aff,
                                       padding_factor=1.0,
                                       crop_factor=1.0)

        return {'source_image_aff': cropped_image_batch,
                'target_image_aff': target_image_aff,
                'source_image_tps': source_image_tps,
                'target_image_tps': target_image_tps,
                'theta_GT_aff': theta_aff,
                'theta_GT_tps': theta_tps}


class SynthTwoPairTnf(SynthPairTnf):
    def __init__(self, use_cuda=True, crop_factor=9 / 16, output_size=(240, 240), padding_factor=0.5):
        super().__init__(use_cuda=use_cuda)
        #        self.aff_reorder_idx=torch.LongTensor([3,2,5,1,0,4])
        self.affTnf = GeometricTnf(geometric_model='affine', out_h=self.out_h, out_w=self.out_w,
                                   use_cuda=self.use_cuda)
        self.tpsTnf = GeometricTnf(geometric_model='tps', out_h=self.out_h, out_w=self.out_w,
                                   use_cuda=self.use_cuda)

    def __call__(self, batch):
        image_batch, theta_batch = batch['image'], batch['theta']
        #        theta_aff=torch.index_select(theta_batch[:,:6],1,self.aff_reorder_idx)
        theta_aff = theta_batch[:, :6].contiguous()
        theta_tps = theta_batch[:, 6:]

        if self.use_cuda:
            image_batch = image_batch.cuda()
            theta_aff = theta_aff.cuda()
            theta_tps = theta_tps.cuda()

        b, c, h, w = image_batch.size()

        # generate symmetrically padded image for bigger sampling region
        image_batch = self.symmetricImagePad(image_batch, self.padding_factor)

        # convert to variables
        image_batch = Variable(image_batch, requires_grad=False)
        theta_aff = Variable(theta_aff, requires_grad=False)
        theta_tps = Variable(theta_tps, requires_grad=False)

        # get cropped image
        cropped_image_batch = self.rescalingTnf(image_batch=image_batch,
                                                theta_batch=None,
                                                padding_factor=self.padding_factor,
                                                crop_factor=self.crop_factor)  # Identity is used as no theta given
        # get transformed image
        warped_image_aff = self.affTnf(image_batch=image_batch,
                                       theta_batch=theta_aff,
                                       padding_factor=self.padding_factor,
                                       crop_factor=self.crop_factor)

        warped_image_tps = self.tpsTnf(image_batch=image_batch,
                                       theta_batch=theta_tps,
                                       padding_factor=self.padding_factor,
                                       crop_factor=self.crop_factor)

        return {'source_image': cropped_image_batch, 'target_image_aff': warped_image_aff,
                'target_image_tps': warped_image_tps, 'theta_GT_aff': theta_aff, 'theta_GT_tps': theta_tps}


class AffineGridGen(Module):
    def __init__(self, out_h=240, out_w=240, out_ch=3, use_cuda=True):
        super(AffineGridGen, self).__init__()
        self.out_h = out_h
        self.out_w = out_w
        self.out_ch = out_ch

    def forward(self, theta):
        b = theta.size()[0]
        if not theta.size() == (b, 2, 3):
            theta = theta.view(-1, 2, 3)
        theta = theta.contiguous()
        batch_size = theta.size()[0]
        out_size = torch.Size((batch_size, self.out_ch, self.out_h, self.out_w))
        return F.affine_grid(theta, out_size, align_corners=True)


class AffineGridGenV2(Module):
    def __init__(self, out_h=240, out_w=240, use_cuda=True):
        super(AffineGridGenV2, self).__init__()
        self.out_h, self.out_w = out_h, out_w
        self.use_cuda = use_cuda

        # create grid in numpy
        # self.grid = np.zeros( [self.out_h, self.out_w, 3], dtype=np.float32)
        # sampling grid with dim-0 coords (Y)
        self.grid_X, self.grid_Y = np.meshgrid(np.linspace(-1, 1, out_w), np.linspace(-1, 1, out_h))
        # grid_X,grid_Y: size [1,H,W,1,1]
        self.grid_X = torch.FloatTensor(self.grid_X).unsqueeze(0).unsqueeze(3)
        self.grid_Y = torch.FloatTensor(self.grid_Y).unsqueeze(0).unsqueeze(3)
        self.grid_X = Variable(self.grid_X, requires_grad=False)
        self.grid_Y = Variable(self.grid_Y, requires_grad=False)
        if use_cuda:
            self.grid_X = self.grid_X.cuda()
            self.grid_Y = self.grid_Y.cuda()

    def forward(self, theta):
        b = theta.size(0)
        if not theta.size() == (b, 6):
            theta = theta.view(b, 6)
            theta = theta.contiguous()

        t0 = theta[:, 0].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        t1 = theta[:, 1].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        t2 = theta[:, 2].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        t3 = theta[:, 3].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        t4 = theta[:, 4].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        t5 = theta[:, 5].unsqueeze(1).unsqueeze(2).unsqueeze(3)

        grid_X = expand_dim(self.grid_X, 0, b)
        grid_Y = expand_dim(self.grid_Y, 0, b)
        grid_Xp = grid_X * t0 + grid_Y * t1 + t2
        grid_Yp = grid_X * t3 + grid_Y * t4 + t5

        return torch.cat((grid_Xp, grid_Yp), 3)


class HomographyGridGen(Module):
    def __init__(self, out_h=240, out_w=240, use_cuda=True):
        super(HomographyGridGen, self).__init__()
        self.out_h, self.out_w = out_h, out_w
        self.use_cuda = use_cuda

        # create grid in numpy
        # self.grid = np.zeros( [self.out_h, self.out_w, 3], dtype=np.float32)
        # sampling grid with dim-0 coords (Y)
        self.grid_X, self.grid_Y = np.meshgrid(np.linspace(-1, 1, out_w), np.linspace(-1, 1, out_h))
        # grid_X,grid_Y: size [1,H,W,1,1]
        self.grid_X = torch.FloatTensor(self.grid_X).unsqueeze(0).unsqueeze(3)
        self.grid_Y = torch.FloatTensor(self.grid_Y).unsqueeze(0).unsqueeze(3)
        self.grid_X = Variable(self.grid_X, requires_grad=False)
        self.grid_Y = Variable(self.grid_Y, requires_grad=False)
        if use_cuda:
            self.grid_X = self.grid_X.cuda()
            self.grid_Y = self.grid_Y.cuda()

    def forward(self, theta):
        b = theta.size(0)
        if theta.size(1) == 9:
            H = theta
        else:
            H = homography_mat_from_4_pts(theta)
        h0 = H[:, 0].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        h1 = H[:, 1].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        h2 = H[:, 2].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        h3 = H[:, 3].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        h4 = H[:, 4].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        h5 = H[:, 5].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        h6 = H[:, 6].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        h7 = H[:, 7].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        h8 = H[:, 8].unsqueeze(1).unsqueeze(2).unsqueeze(3)

        grid_X = expand_dim(self.grid_X, 0, b)
        grid_Y = expand_dim(self.grid_Y, 0, b)

        grid_Xp = grid_X * h0 + grid_Y * h1 + h2
        grid_Yp = grid_X * h3 + grid_Y * h4 + h5
        k = grid_X * h6 + grid_Y * h7 + h8

        grid_Xp /= k
        grid_Yp /= k

        return torch.cat((grid_Xp, grid_Yp), 3)


def homography_mat_from_4_pts(theta):
    b = theta.size(0)
    if not theta.size() == (b, 8):
        theta = theta.view(b, 8)
        theta = theta.contiguous()

    xp = theta[:, :4].unsqueeze(2)
    yp = theta[:, 4:].unsqueeze(2)

    x = Variable(torch.FloatTensor([-1, -1, 1, 1])).unsqueeze(1).unsqueeze(0).expand(b, 4, 1)
    y = Variable(torch.FloatTensor([-1, 1, -1, 1])).unsqueeze(1).unsqueeze(0).expand(b, 4, 1)
    z = Variable(torch.zeros(4)).unsqueeze(1).unsqueeze(0).expand(b, 4, 1)
    o = Variable(torch.ones(4)).unsqueeze(1).unsqueeze(0).expand(b, 4, 1)
    single_o = Variable(torch.ones(1)).unsqueeze(1).unsqueeze(0).expand(b, 1, 1)

    if theta.is_cuda:
        x = x.cuda()
        y = y.cuda()
        z = z.cuda()
        o = o.cuda()
        single_o = single_o.cuda()

    A = torch.cat([torch.cat([-x, -y, -o, z, z, z, x * xp, y * xp, xp], 2),
                   torch.cat([z, z, z, -x, -y, -o, x * yp, y * yp, yp], 2)], 1)
    # find homography by assuming h33 = 1 and inverting the linear system
    h = torch.bmm(torch.inverse(A[:, :, :8]), -A[:, :, 8].unsqueeze(2))
    # add h33
    h = torch.cat([h, single_o], 1)
    H = h.squeeze(2)

    return H


class TpsGridGen(Module):
    def __init__(self, out_h=240, out_w=240, use_regular_grid=True, grid_size=3, reg_factor=0, use_cuda=True):
        super(TpsGridGen, self).__init__()
        self.out_h, self.out_w = out_h, out_w
        self.reg_factor = reg_factor
        self.use_cuda = use_cuda

        # create grid in numpy
        # self.grid = np.zeros( [self.out_h, self.out_w, 3], dtype=np.float32)
        # sampling grid with dim-0 coords (Y)
        self.grid_X, self.grid_Y = np.meshgrid(np.linspace(-1, 1, out_w), np.linspace(-1, 1, out_h))
        # grid_X,grid_Y: size [1,H,W,1,1]
        self.grid_X = torch.FloatTensor(self.grid_X).unsqueeze(0).unsqueeze(3)
        self.grid_Y = torch.FloatTensor(self.grid_Y).unsqueeze(0).unsqueeze(3)
        self.grid_X = Variable(self.grid_X, requires_grad=False)
        self.grid_Y = Variable(self.grid_Y, requires_grad=False)
        if use_cuda:
            self.grid_X = self.grid_X.cuda()
            self.grid_Y = self.grid_Y.cuda()

        # initialize regular grid for control points P_i
        if use_regular_grid:
            axis_coords = np.linspace(-1, 1, grid_size)
            self.N = grid_size * grid_size
            P_Y, P_X = np.meshgrid(axis_coords, axis_coords)
            P_X = np.reshape(P_X, (-1, 1))  # size (N,1)
            P_Y = np.reshape(P_Y, (-1, 1))  # size (N,1)
            P_X = torch.FloatTensor(P_X)
            P_Y = torch.FloatTensor(P_Y)
            self.Li = Variable(self.compute_L_inverse(P_X, P_Y).unsqueeze(0), requires_grad=False)
            self.P_X = P_X.unsqueeze(2).unsqueeze(3).unsqueeze(4).transpose(0, 4)
            self.P_Y = P_Y.unsqueeze(2).unsqueeze(3).unsqueeze(4).transpose(0, 4)
            self.P_X = Variable(self.P_X, requires_grad=False)
            self.P_Y = Variable(self.P_Y, requires_grad=False)
            if use_cuda:
                self.P_X = self.P_X.cuda()
                self.P_Y = self.P_Y.cuda()

    def forward(self, theta):
        warped_grid = self.apply_transformation(theta, torch.cat((self.grid_X, self.grid_Y), 3))

        return warped_grid

    def compute_L_inverse(self, X, Y):
        N = X.size()[0]  # num of points (along dim 0)
        # construct matrix K
        Xmat = X.expand(N, N)
        Ymat = Y.expand(N, N)
        P_dist_squared = torch.pow(Xmat - Xmat.transpose(0, 1), 2) + torch.pow(Ymat - Ymat.transpose(0, 1), 2)
        P_dist_squared[P_dist_squared == 0] = 1  # make diagonal 1 to avoid NaN in log computation
        K = torch.mul(P_dist_squared, torch.log(P_dist_squared))
        if self.reg_factor != 0:
            K += torch.eye(K.size(0), K.size(1)) * self.reg_factor
        # construct matrix L
        O = torch.FloatTensor(N, 1).fill_(1)
        Z = torch.FloatTensor(3, 3).fill_(0)
        P = torch.cat((O, X, Y), 1)
        L = torch.cat((torch.cat((K, P), 1), torch.cat((P.transpose(0, 1), Z), 1)), 0)
        Li = torch.inverse(L)
        if self.use_cuda:
            Li = Li.cuda()
        return Li

    def apply_transformation(self, theta, points):
        if theta.dim() == 2:
            theta = theta.unsqueeze(2).unsqueeze(3)
        # points should be in the [B,H,W,2] format,
        # where points[:,:,:,0] are the X coords
        # and points[:,:,:,1] are the Y coords

        # input are the corresponding control points P_i
        batch_size = theta.size()[0]
        # split theta into point coordinates
        Q_X = theta[:, :self.N, :, :].squeeze(3)
        Q_Y = theta[:, self.N:, :, :].squeeze(3)

        # get spatial dimensions of points
        points_b = points.size()[0]
        points_h = points.size()[1]
        points_w = points.size()[2]

        # repeat pre-defined control points along spatial dimensions of points to be transformed
        P_X = self.P_X.expand((1, points_h, points_w, 1, self.N))
        P_Y = self.P_Y.expand((1, points_h, points_w, 1, self.N))

        # compute weigths for non-linear part
        W_X = torch.bmm(self.Li[:, :self.N, :self.N].expand((batch_size, self.N, self.N)), Q_X)
        W_Y = torch.bmm(self.Li[:, :self.N, :self.N].expand((batch_size, self.N, self.N)), Q_Y)
        # reshape
        # W_X,W,Y: size [B,H,W,1,N]
        W_X = W_X.unsqueeze(3).unsqueeze(4).transpose(1, 4).repeat(1, points_h, points_w, 1, 1)
        W_Y = W_Y.unsqueeze(3).unsqueeze(4).transpose(1, 4).repeat(1, points_h, points_w, 1, 1)
        # compute weights for affine part
        A_X = torch.bmm(self.Li[:, self.N:, :self.N].expand((batch_size, 3, self.N)), Q_X)
        A_Y = torch.bmm(self.Li[:, self.N:, :self.N].expand((batch_size, 3, self.N)), Q_Y)
        # reshape
        # A_X,A,Y: size [B,H,W,1,3]
        A_X = A_X.unsqueeze(3).unsqueeze(4).transpose(1, 4).repeat(1, points_h, points_w, 1, 1)
        A_Y = A_Y.unsqueeze(3).unsqueeze(4).transpose(1, 4).repeat(1, points_h, points_w, 1, 1)

        # compute distance P_i - (grid_X,grid_Y)
        # grid is expanded in point dim 4, but not in batch dim 0, as points P_X,P_Y are fixed for all batch
        points_X_for_summation = points[:, :, :, 0].unsqueeze(3).unsqueeze(4).expand(
            points[:, :, :, 0].size() + (1, self.N))
        points_Y_for_summation = points[:, :, :, 1].unsqueeze(3).unsqueeze(4).expand(
            points[:, :, :, 1].size() + (1, self.N))

        if points_b == 1:
            delta_X = points_X_for_summation - P_X
            delta_Y = points_Y_for_summation - P_Y
        else:
            # use expanded P_X,P_Y in batch dimension
            delta_X = points_X_for_summation - P_X.expand_as(points_X_for_summation)
            delta_Y = points_Y_for_summation - P_Y.expand_as(points_Y_for_summation)

        dist_squared = torch.pow(delta_X, 2) + torch.pow(delta_Y, 2)
        # U: size [1,H,W,1,N]
        dist_squared[dist_squared == 0] = 1  # avoid NaN in log computation
        U = torch.mul(dist_squared, torch.log(dist_squared))

        # expand grid in batch dimension if necessary
        points_X_batch = points[:, :, :, 0].unsqueeze(3)
        points_Y_batch = points[:, :, :, 1].unsqueeze(3)
        if points_b == 1:
            points_X_batch = points_X_batch.expand((batch_size,) + points_X_batch.size()[1:])
            points_Y_batch = points_Y_batch.expand((batch_size,) + points_Y_batch.size()[1:])

        points_X_prime = A_X[:, :, :, :, 0] + \
                         torch.mul(A_X[:, :, :, :, 1], points_X_batch) + \
                         torch.mul(A_X[:, :, :, :, 2], points_Y_batch) + \
                         torch.sum(torch.mul(W_X, U.expand_as(W_X)), 4)

        points_Y_prime = A_Y[:, :, :, :, 0] + \
                         torch.mul(A_Y[:, :, :, :, 1], points_X_batch) + \
                         torch.mul(A_Y[:, :, :, :, 2], points_Y_batch) + \
                         torch.sum(torch.mul(W_Y, U.expand_as(W_Y)), 4)

        return torch.cat((points_X_prime, points_Y_prime), 3)

class Theta_gen():
    """
    Synthetically transformed pairs dataset for training with strong supervision
    Args:
            csv_file (string): Path to the csv file with image names and transformations.
            training_image_path (string): Directory with all the images.
            transform (callable): Transformation for post-processing the training pair (eg. image normalization)
    Returns:
            Dict: {'image': full dataset image, 'theta': desired transformation}
    """

    def __init__(self,
                 output_size=(480, 640),
                 geometric_model='affine',
                 random_t=0.5,
                 random_s=0.5,
                 random_alpha=1 / 6,
                 random_t_tps=0.5,
                 four_point_hom=True):

        self.out_h, self.out_w = output_size
        self.random_t = random_t
        self.random_t_tps = random_t_tps
        self.random_alpha = random_alpha
        self.random_s = random_s
        self.four_point_hom = four_point_hom
        self.geometric_model = geometric_model
        self.affineTnf = GeometricTnf(out_h=self.out_h, out_w=self.out_w, use_cuda=False)


    def __call__(self):
        # np.random.seed(1)  # for debugging purposes

        if self.geometric_model == 'affine' or self.geometric_model == 'afftps':
            '''
            rotate: -pi/6 ~ pi/6
            shear: -pi/6 ~ pi/6
            translation: -0.25 ~ 0.25
            lambda: 0.75 ~ 1.25
            '''
            rot_angle = (np.random.rand(1) - 0.5) * 2 * np.pi / 3  # between -np.pi/12 and np.pi/12
            sh_angle = (np.random.rand(1) - 0.5) * 2 * np.pi / 2  # between -np.pi/6 and np.pi/6
            lambda_1 = 1 + (2 * np.random.rand(1) - 1) * 0.2  # between 0.75 and 1.25
            lambda_2 = 1 + (2 * np.random.rand(1) - 1) * 0.2  # between 0.75 and 1.25
            tx = (2 * np.random.rand(1) - 1) * 0.75  # between -0.25 and 0.25
            ty = (2 * np.random.rand(1) - 1) * 0.75

            R_sh = np.array([[np.cos(sh_angle[0]), -np.sin(sh_angle[0])],
                             [np.sin(sh_angle[0]), np.cos(sh_angle[0])]])
            R_alpha = np.array([[np.cos(rot_angle[0]), -np.sin(rot_angle[0])],
                                [np.sin(rot_angle[0]), np.cos(rot_angle[0])]])

            D = np.diag([lambda_1[0], lambda_2[0]])

            A = R_alpha @ R_sh.transpose() @ D @ R_sh

            theta_aff = np.array([A[0, 0], A[0, 1], tx, A[1, 0], A[1, 1], ty],np.float32)
        if self.geometric_model == 'hom':
            theta_hom = np.array([-1, -1, 1, 1, -1, 1, -1, 1],np.float32) 
            theta_hom = theta_hom + (np.random.rand(8) - 0.5) * 2 * self.random_t_tps
        if self.geometric_model == 'tps' or self.geometric_model == 'afftps':
            theta_tps = np.array([-1, -1, -1, 0, 0, 0, 1, 1, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1], dtype=np.float32)
            theta_tps = theta_tps + (np.random.rand(18) - 0.5) * 2 * self.random_t_tps
        if self.geometric_model == 'affine':
            theta = theta_aff
        elif self.geometric_model == 'hom':
            theta = theta_hom
        elif self.geometric_model == 'tps':
            theta = theta_tps
        elif self.geometric_model == 'afftps':
            theta = np.concatenate((theta_aff, theta_tps))

        theta = torch.Tensor(theta.astype(np.float32))

        if self.geometric_model == 'hom' and self.four_point_hom == False:
            theta = homography_mat_from_4_pts(Variable(theta.unsqueeze(0))).squeeze(0).data

        return theta