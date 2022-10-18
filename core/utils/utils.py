import torch
import torch.nn.functional as F
import numpy as np
from scipy import interpolate
from core.utils.forward_warp import ForwardWarp

class InputPadder:
    """ Pads images such that dimensions are divisible by 8 """
    def __init__(self, dims, mode='sintel'):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // 8) + 1) * 8 - self.ht) % 8
        pad_wd = (((self.wd // 8) + 1) * 8 - self.wd) % 8
        if mode == 'sintel':
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, pad_ht//2, pad_ht - pad_ht//2]
        else:
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, 0, pad_ht]

    def pad(self, *inputs):
        return [F.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self,x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht-self._pad[3], self._pad[0], wd-self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]

def pytorch_foward_interpolate(flow):
    def get_gaussian_weights(x, y, x1, x2, y1, y2):
        w11 = torch.exp(-((x - x1)**2 + (y - y1)**2))
        w12 = torch.exp(-((x - x1)**2 + (y - y2)**2))
        w21 = torch.exp(-((x - x2)**2 + (y - y1)**2))
        w22 = torch.exp(-((x - x2)**2 + (y - y2)**2))

        return w11, w12, w21, w22

    def sample_one(img, shiftx, shifty, weight):
        """
		Input:
			-img (N, C, H, W) (flow)
			-shiftx, shifty (N, c, H, W)
		"""

        N, C, H, W = img.size()

        # flatten all (all restored as Tensors)
        flat_shiftx = shiftx.view(-1)
        flat_shifty = shifty.view(-1)
        flat_basex = torch.arange(0, H, requires_grad=False).view(-1, 1)[None, None].cuda().long().repeat(N, C, 1, W).view(-1)
        flat_basey = torch.arange(0, W, requires_grad=False).view(1, -1)[None, None].cuda().long().repeat(N, C, H, 1).view(-1)
        flat_weight = weight.view(-1)
        flat_img = img.view(-1)

        # The corresponding positions in I1
        idxn = torch.arange(0, N, requires_grad=False).view(N, 1, 1, 1).long().cuda().repeat(1, C, H, W).view(-1)
        idxc = torch.arange(0, C, requires_grad=False).view(1, C, 1, 1).long().cuda().repeat(N, 1, H, W).view(-1)
        # ttype = flat_basex.type()
        idxx = flat_shiftx.long() + flat_basex
        idxy = flat_shifty.long() + flat_basey


        # recording the inside part the shifted
        mask = idxx.ge(0) & idxx.lt(H) & idxy.ge(0) & idxy.lt(W)

        # Mask off points out of boundaries
        ids = (idxn*C*H*W + idxc*H*W + idxx*W + idxy)
        ids_mask = torch.masked_select(ids, mask).clone().cuda()

        #(zero part - gt) -> difference
        # difference back propagate -> No influence! Whether we do need mask? mask?
                # put (add) them together
        # Note here! accmulate fla must be true for proper bp
        img_warp = torch.zeros([N*C*H*W, ]).cuda() 
        img_warp.put_(ids_mask, torch.masked_select(flat_img*flat_weight, mask), accumulate=True)

        return img_warp.view(N, C, H, W)

    """
        flow should be a pytorch tensor 
        flow - Bx2xWxH
    """
    flow = flow.detach()
    img = flow

    N, C, ht, wd = flow.shape
    y = flow[:, 0:1 :, :]
    x = flow[:, 1:2, :, :]

    x = x.repeat(1, C, 1, 1)
    y = y.repeat(1, C, 1, 1)

    # Four point of square (x1, y1), (x1, y2), (x2, y1), (y2, y2)
    x1 = torch.floor(x)
    x2 = x1 + 1
    y1 = torch.floor(y)
    y2 = y1 + 1

    # firstly, get gaussian weights
    w11, w12, w21, w22 = get_gaussian_weights(x, y, x1, x2, y1, y2)

    # secondly, sample each weighted corner 
    img11 = sample_one(img, x1, y1, w11)
    img12 = sample_one(img, x1, y2, w12)
    img21 = sample_one(img, x2, y1, w21)
    img22 = sample_one(img, x2, y2, w22)


    imgw = img11 + img12 + img21 + img22

    return imgw

def forward_interpolate(flow):
    flow = flow.detach().cpu().numpy()
    dx, dy = flow[0], flow[1]

    ht, wd = dx.shape
    x0, y0 = np.meshgrid(np.arange(wd), np.arange(ht))

    x1 = x0 + dx
    y1 = y0 + dy
    
    x1 = x1.reshape(-1)
    y1 = y1.reshape(-1)
    dx = dx.reshape(-1)
    dy = dy.reshape(-1)

    valid = (x1 > 0) & (x1 < wd) & (y1 > 0) & (y1 < ht)
    x1 = x1[valid]
    y1 = y1[valid]
    dx = dx[valid]
    dy = dy[valid]

    flow_x = interpolate.griddata(
        (x1, y1), dx, (x0, y0), method='nearest', fill_value=0)

    flow_y = interpolate.griddata(
        (x1, y1), dy, (x0, y0), method='nearest', fill_value=0)

    flow = np.stack([flow_x, flow_y], axis=0)
    return torch.from_numpy(flow).float()


def bilinear_sampler(img, coords, mode='bilinear', mask=False):
    """ Wrapper for grid_sample, uses pixel coordinates """
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1,1], dim=-1)
    xgrid = 2*xgrid/(W-1) - 1
    ygrid = 2*ygrid/(H-1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, align_corners=True)

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()

    return img


def coords_grid(batch, ht, wd, device='cpu'):
    coords = torch.meshgrid(torch.arange(ht, device=device), torch.arange(wd, device=device))
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)


def upflow8(flow, mode='bilinear'):
    new_size = (8 * flow.shape[2], 8 * flow.shape[3])
    return  8 * F.interpolate(flow, size=new_size, mode=mode, align_corners=True)


def entry_convert(entry):
    intrinsic = np.array([entry[2], 0., entry[4],
                          0., entry[3], entry[5],
                          0., 0., 1.]).reshape(3, 3)

    pose = np.eye(4)
    pose[:3, :3] = np.array(entry[6:15]).reshape(3, 3)
    pose[:3, 3] = np.array(entry[15:]).reshape(3)

    return pose, intrinsic    

def fundamental_matrix_gen(Tcw1, Tcw2, K1, K2):

    T21 = Tcw2.dot(np.linalg.inv(Tcw1))

    R = T21[:3, :3]
    t = T21[:3, 3]

    t_antisymmetric = np.zeros([3, 3])
    t_antisymmetric[0][1] = -t[2]
    t_antisymmetric[0][2] = t[1]
    t_antisymmetric[1][0] = t[2]
    t_antisymmetric[1][2] = -t[0]
    t_antisymmetric[2][0] = -t[1]
    t_antisymmetric[2][1] = t[0]

    essential_matrix = t_antisymmetric.dot(R)

    fundamental_matrix = (np.linalg.inv(K2.transpose()).dot(essential_matrix)).dot(np.linalg.inv(K1))

    return fundamental_matrix.astype(np.float32)

def expand_dim(tensor, dim, desired_dim_len):
    sz = list(tensor.size())
    sz[dim] = desired_dim_len
    return tensor.expand(tuple(sz))


def reverse_mapping(flow, grid_map):
    '''
    Input:
        flow: B x 2 x H x W torch.Tensor.cuda, represent the flow from A to B
        grid_map: B x H x W x 2 torch.Tensor.cuda, grid mapping from B to A
    Output:
        reverse_map: B x H x W x 2 torch.Tensor.cuda, grid mapping from A to B
    '''
    _, H, W, _ = grid_map.shape
    grid_norm = grid_map.clone()
    grid_norm[:, :, :, 0] = (grid_norm[:, :, :, 0] * 2 - W + 1) / (W - 1)
    grid_norm[:, :, :, 1] = (grid_norm[:, :, :, 1] * 2 - H + 1) / (H - 1)
    reverse_map = F.grid_sample(flow, grid_norm, align_corners=True).permute([0, 2, 3, 1]) + grid_map
    return reverse_map

def reverse_mask(mask, grid_map):
    '''
    Input:
        mask: B x 1 x H x W torch.Tensor.cuda, represent the mask of A
        grid_map: B x H x W x 2 torch.Tensor.cuda, grid mapping from B to A
    Output:
        reverse_map: B x H x W x 2 torch.Tensor.cuda, grid mapping from A to B
    '''
    _, H, W, _ = grid_map.shape
    grid_norm = grid_map.clone()
    grid_norm[:, :, :, 0] = (grid_norm[:, :, :, 0] * 2 - W + 1) / (W - 1)
    grid_norm[:, :, :, 1] = (grid_norm[:, :, :, 1] * 2 - H + 1) / (H - 1)
    reverse_mask = F.grid_sample(mask, grid_norm, align_corners=True).permute([0, 2, 3, 1])
    return reverse_mask

def refine_grid(grid):
    H, W, _ = grid.shape
    grid[:,:,0] = (grid[:,:,0] + 1/2) * W
    grid[:,:,1] = (grid[:,:,1] + 1/2) * H

    return grid

def convert_flow_to_mapping(flow, output_channel_first=True):
    if not isinstance(flow, np.ndarray):
        # torch tensor
        if len(flow.shape) == 4:
            if flow.shape[1] != 2:
                # size is BxHxWx2
                flow = flow.permute(0, 3, 1, 2)

            B, C, H, W = flow.size()

            xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
            yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
            xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
            yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
            grid = torch.cat((xx, yy), 1).float()

            if flow.is_cuda:
                grid = grid.cuda()
            map = flow + grid  # here also channel first
            if not output_channel_first:
                map = map.permute(0,2,3,1)
        else:
            if flow.shape[0] != 2:
                # size is HxWx2
                flow = flow.permute(2, 0, 1)

            C, H, W = flow.size()

            xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
            yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
            xx = xx.view(1, H, W)
            yy = yy.view(1, H, W)
            grid = torch.cat((xx, yy), 0).float() # attention, concat axis=0 here

            if flow.is_cuda:
                grid = grid.cuda()
            map = flow + grid # here also channel first
            if not output_channel_first:
                map = map.permute(1,2,0).float()
        return map.float()
    else:
        # here numpy arrays
        if len(flow.shape) == 4:
            if flow.shape[3] != 2:
                # size is Bx2xHxW
                flow = flow.transpose(0, 2, 3, 1)
            # BxHxWx2
            b, h_scale, w_scale = flow.shape[:3]
            map = np.copy(flow)
            X, Y = np.meshgrid(np.linspace(0, w_scale - 1, w_scale),
                               np.linspace(0, h_scale - 1, h_scale))
            for i in range(b):
                map[i, :, :, 0] = flow[i, :, :, 0] + X
                map[i, :, :, 1] = flow[i, :, :, 1] + Y
            if output_channel_first:
                map = map.transpose(0,3,1,2)
        else:
            if flow.shape[0] == 2:
                # size is 2xHxW
                flow = flow.transpose(1,2,0)
            # HxWx2
            h_scale, w_scale = flow.shape[:2]
            map = np.copy(flow)
            X, Y = np.meshgrid(np.linspace(0, w_scale - 1, w_scale),
                               np.linspace(0, h_scale - 1, h_scale))

            map[:,:,0] = flow[:,:,0] + X
            map[:,:,1] = flow[:,:,1] + Y
            if output_channel_first:
                map = map.transpose(2,0,1)
        return map.astype(np.float32)

def get_gt_correspondence_mask(flow):
    """Computes the mask of valid flows (that do not match to a pixel outside of the image). """
    mapping = convert_flow_to_mapping(flow, output_channel_first=True)
    print(mapping.shape)
    if isinstance(mapping, np.ndarray):
        if len(mapping.shape) == 4:
            # shape is B,C,H,W
            b, _, h, w = mapping.shape
            mask_x = np.logical_and(mapping[:, 0] > 0, mapping[:, 0] < w)
            mask_y = np.logical_and(mapping[:, 1] > 0, mapping[:, 1] < h)
            mask = np.logical_and(mask_x, mask_y)
        else:
            _, h, w = mapping.shape
            mask_x = np.logical_and(mapping[0] > 0, mapping[0] < w)
            mask_y = np.logical_and(mapping[1] > 0, mapping[1] < h)
            mask = np.logical_and(mask_x, mask_y)
        mask = mask.astype(np.bool) if float(torch.__version__[:3]) >= 1.1 else mask.astype(np.uint8)
    else:
        if len(mapping.shape) == 4:
            # shape is B,C,H,W
            b, _, h, w = mapping.shape
            mask = mapping[:, 0].ge(0) & mapping[:, 0].le(w) & mapping[:, 1].ge(0) & mapping[:, 1].le(h)
        else:
            _, h, w = mapping.shape
            mask = mapping[0].ge(0) & mapping[0].le(w) & mapping[1].ge(0) & mapping[1].le(h)
        mask = mask.bool() if float(torch.__version__[:3]) >= 1.1 else mask.byte()
    return mask

def image_flow_warp(image, flow, padding_mode='zeros'):
    '''
    Input:
        image: HxWx3 numpy
        flow: HxWx2 torch.Tensor
    Output:
        outImg: HxWx3 numpy
    '''
    image = torch.from_numpy(image)
    if image.ndim == 2:
        image = image[None].permute([1,2,0])
    H, W, _ = image.shape
    coords = coords_grid(1, H, W).cuda().float().contiguous()
    flow = flow[None].repeat(1, 1, 1, 1).permute([0, 3, 1, 2]).float().contiguous()
    grid = (flow + coords).permute([0, 2, 3, 1]).contiguous()
    grid[:, :, :, 0] = (grid[:, :, :, 0] * 2 - W + 1) / (W - 1)
    grid[:, :, :, 1] = (grid[:, :, :, 1] * 2 - H + 1) / (H - 1)
    image = image[None].permute([0, 3, 1, 2]).cuda().float()
    outImg = F.grid_sample(image, grid, padding_mode=padding_mode, align_corners=False)[0].cpu().numpy().transpose([1, 2, 0])

    return outImg.astype(np.uint8)

def image_forward_warp(image, flow, padding_mode='zeros'):
    '''
    Input:
        image: HxWx3 numpy
        flow: HxWx2 torch.Tensor
    Output:
        outImg: HxWx3 numpy
    '''
    forward_warp = ForwardWarp()
    image = torch.from_numpy(image).permute([2,0,1])[None].cuda()
    flow = torch.from_numpy(flow).permute([2,0,1])[None].cuda()
    out = forward_warp(image, flow)
    image = (out[0][0] / (out[1][0] + 1e-6)).cpu().permute([1,2,0]).numpy().astype(np.uint8)

    return image