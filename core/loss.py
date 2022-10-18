import kornia
import kornia.geometry.epipolar as epipolar
import torch
import torch.nn.functional as F
from core.utils.utils import coords_grid, reverse_mapping, reverse_mask
import time

def compute_all_loss(args, data, flows_pre):
    # loss_AB means compute loss using flow_ab
    losses = {'sed_BA_loss' : 0,
              'sed_AB_loss' : 0,
              'tnf_BB1_loss': 0,
              'tnf_B1B_loss': 0,
              'total_loss'  : 0}

    flow_BA  = flows_pre['output_BA']
    flow_AB  = flows_pre['output_AB']
    flow_BB1 = flows_pre['output_BB1']
    flow_B1B = flows_pre['output_B1B']

    N = args.iters
    B, _, H, W = data['im1'].shape
    Fm = data['fundamental_matrix'].to(data['im1'].device)

    for i in range(N):
        i_weight = args.gamma ** (N - i - 1)

        if args.sed_loss:
            losses['sed_BA_loss'] += i_weight * compute_symmetrical_epipolar_distance_loss(args, flow_BA[i], Fm.permute(0, 2, 1))
            losses['sed_AB_loss'] += i_weight * compute_symmetrical_epipolar_distance_loss(args, flow_AB[i], Fm)
        if args.tnf_loss:
            losses['tnf_BB1_loss'] += i_weight * compute_transformation_loss(args, data, flow_BB1[i], type='BB1')
            losses['tnf_B1B_loss'] += i_weight * compute_transformation_loss(args, data, flow_B1B[i], type='B1B')

    losses['total_loss'] = losses['sed_BA_loss'] + losses['sed_AB_loss'] + losses['tnf_B1B_loss'] + losses['tnf_BB1_loss']

    return losses

def compute_symmetrical_epipolar_distance_loss(args, flow_ab, Fm):
    B, _, H, W = flow_ab.shape
    pts1 = (coords_grid(B, H, W, flow_ab.device)).permute(0, 2, 3, 1).reshape(B, -1, 2).contiguous()
    pts2 = (coords_grid(B, H, W, flow_ab.device) + flow_ab).permute(0, 2, 3, 1).reshape(B, -1, 2).contiguous()
    dist = symmetrical_epipolar_distance(pts1, pts2, Fm, squared=False)
    if args.clamp:
        dist = torch.clamp(dist, max=args.clamp)
    return dist.mean()

def compute_epipolar_distance_loss(args, flow_ab, Fm):
    B, _, H, W = flow_ab.shape
    pts1 = (coords_grid(B, H, W, flow_ab.device)).permute(0, 2, 3, 1).reshape(B, -1, 2).contiguous()
    pts2 = (coords_grid(B, H, W, flow_ab.device) + flow_ab).permute(0, 2, 3, 1).reshape(B, -1, 2).contiguous()
    dist = epipolar_distance(pts1, pts2, Fm, squared=False)
    if args.clamp:
        dist = torch.clamp(dist, max=args.clamp)
    return dist.mean()

def symmetrical_epipolar_distance(pts1, pts2, Fm, squared=False, eps = 1e-8):
    '''
    copy from kornia
    '''
    if not isinstance(Fm, torch.Tensor):
        raise TypeError(f"Fm type is not a torch.Tensor. Got {type(Fm)}")

    if (len(Fm.shape) != 3) or not Fm.shape[-2:] == (3, 3):
        raise ValueError(f"Fm must be a (*, 3, 3) tensor. Got {Fm.shape}")

    if pts1.size(-1) == 2:
        pts1 = kornia.geometry.convert_points_to_homogeneous(pts1)

    if pts2.size(-1) == 2:
        pts2 = kornia.geometry.convert_points_to_homogeneous(pts2)

    F_t: torch.Tensor = Fm.permute(0, 2, 1)
    line1_in_2: torch.Tensor = pts1 @ F_t
    line2_in_1: torch.Tensor = pts2 @ Fm
    
    numerator: torch.Tensor = (pts2 * line1_in_2).sum(2).pow(2)
    denominator_inv: torch.Tensor = 1.0 / (line1_in_2[..., :2].norm(2, dim=2).pow(2) + eps) + 1.0 / (
        line2_in_1[..., :2].norm(2, dim=2).pow(2) + eps
    )
    out: torch.Tensor = numerator * denominator_inv

    if squared:
        return out
    return (out + eps).sqrt()

def epipolar_distance(pts1, pts2, Fm, squared=False, eps = 1e-8):
    '''
    copy from kornia
    '''
    if not isinstance(Fm, torch.Tensor):
        raise TypeError("Fm type is not a torch.Tensor. Got {}".format(type(Fm)))

    if (len(Fm.shape) != 3) or not Fm.shape[-2:] == (3, 3):
        raise ValueError("Fm must be a (*, 3, 3) tensor. Got {}".format(Fm.shape))

    if pts1.size(-1) == 2:
        pts1 = kornia.geometry.convert_points_to_homogeneous(pts1)

    if pts2.size(-1) == 2:
        pts2 = kornia.geometry.convert_points_to_homogeneous(pts2)

    F_t: torch.Tensor = Fm.permute(0, 2, 1).contiguous()
    line1_in_2: torch.Tensor = pts1 @ F_t

    numerator: torch.Tensor = (pts2 * line1_in_2).sum(2).pow(2)

    denominator_inv: torch.Tensor = 1.0 / (line1_in_2[..., :2].norm(2, dim=2).pow(2) + eps)
    out: torch.Tensor = numerator * denominator_inv

    if squared:
        return out
    return (out + eps).sqrt()

def compute_transformation_loss(args, data, flow_est, type='BB1'):
    device = flow_est.device
    B, C, H, W = flow_est.shape
    gt_map, valid_mask = None, None
    if type == 'BB1':
        gt_map = data['forward_map'].to(device)
        valid_mask = data['forward_mask']
    elif type == 'B1B':
        gt_map = data['backward_map'].to(device)
        valid_mask = data['backward_mask']
    
    est_map = (coords_grid(B, H, W, device) + flow_est)
    epe = torch.norm(gt_map - est_map, dim=1, p=1)
    return epe[valid_mask].mean()
