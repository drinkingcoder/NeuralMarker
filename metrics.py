# metrics
from skimage.metrics import mean_squared_error as mse 
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
# psnr
from skimage.util.dtype import dtype_range
from skimage._shared.utils import warn, check_shape_equality
# ssim
from skimage.util.arraycrop import crop
from scipy.ndimage import uniform_filter, gaussian_filter

from tqdm import tqdm
import numpy as np
import cv2
import os

def PSNR(image_true, image_test, mask, heatmap=False, with_mask=True):
    check_shape_equality(image_true, image_test)    

    if image_true.dtype != image_test.dtype:
        warn("Inputs have mismatched dtype. Setting data_range based on "
                "im_true.", stacklevel=2)
    dmin, dmax = dtype_range[image_true.dtype.type]
    true_min, true_max = np.min(image_true), np.max(image_true)
    if true_max > dmax or true_min < dmin:
        raise ValueError(
            "im_true has intensity values outside the range expected for "
            "its data type. Please manually specify the data_range")
    if true_min >= 0:
        # most common case (255 for uint8, 1 for float)
        data_range = dmax
    else:
        data_range = dmax - dmin
    
    image_true = image_true.astype(np.float64)
    image_test = image_test.astype(np.float64)
    if not with_mask:                      
        error_mask = ((image_true - image_test) ** 2).astype(np.float64)
        err = np.mean(error_mask, dtype=np.float64)
    else:
        cnt = np.count_nonzero(1-mask) * image_true.shape[2]
        error_mask = ((image_true*(1-mask) - image_test*(1-mask))**2).astype(np.float64)
        sum = np.sum(error_mask)    
        err = sum/cnt        
        
    score = 10 * np.log10((data_range ** 2) / err)
    if heatmap:                     
        error_mask = np.mean(error_mask, axis=2)   
        # np.save('./error_mask.npy', error_mask)

        x,y = np.nonzero(error_mask)
        error_mask[x,y] = 10 * np.log10((data_range ** 2) / error_mask[x,y])        
        error_mask = error_mask / 30
        
        error_mask = (error_mask * 255).astype(np.uint8)
        error_heatmap = cv2.applyColorMap(error_mask, cv2.COLORMAP_JET)*(1-mask)
        return score, error_heatmap
    else:
        return score

from scipy.ndimage import uniform_filter
def SSIM(im1, im2, mask, multichannel=True, heatmap=False, with_mask=True):
    check_shape_equality(im1, im2)

    if multichannel:
        # loop over channels
        nch = im1.shape[-1]
        mssim = np.empty(nch)
        S = np.empty(im1.shape)
        for ch in range(nch):
            ch_result = SSIM(im1[..., ch], im2[..., ch], multichannel=False, mask=mask, heatmap=heatmap)
            mssim[..., ch], S[...,ch] = ch_result
        mssim = mssim.mean()
        if heatmap:
            S = np.mean(S, axis=2)                        
            S = (S+1)/2
            S = (S * 255).astype(np.uint8)            
            error_heatmap = cv2.applyColorMap(S, cv2.COLORMAP_JET)*(1-mask)
            return mssim, error_heatmap
        else:
            return mssim

    if im1.dtype != im2.dtype:
        warn("Inputs have mismatched dtype.  Setting data_range based on im1.dtype.", stacklevel=2)
    dmin, dmax = dtype_range[im1.dtype.type]
    data_range = dmax - dmin
    
    K1 = 0.01
    K2 = 0.03
    R = data_range
    C1 = (K1 * R) ** 2
    C2 = (K2 * R) ** 2  
    S = np.ones_like(im1)
    win_size=3

    # ndimage filters need floating point data
    if not with_mask:
        im1 = im1.astype(np.float64)
        im2 = im2.astype(np.float64)
    else:
        mask = mask.squeeze()
        im1 = (im1*(1-mask)).astype(np.float64)
        im2 = (im2*(1-mask)).astype(np.float64)
    
    if heatmap:
        ux = uniform_filter(im1, size=win_size)
        uy = uniform_filter(im2, size=win_size)

        uxx = uniform_filter(im1*im1, size=win_size)
        uyy = uniform_filter(im2*im2, size=win_size)
        uxy = uniform_filter(im1*im2, size=win_size)
        vx = uxx - ux*ux 
        vy = uyy - uy*uy
        vxy = uxy - ux*uy

        A1, A2, B1, B2 = ((2 * ux * uy + C1,
                           2 * vxy + C2,
                           ux ** 2 + uy ** 2 + C1,
                           vx + vy + C2))
        D = B1 * B2
        S = (A1 * A2) / D    
    

    if not with_mask:
        mask = np.zeros_like(im1)
        x,y = np.nonzero(1-mask)
    else:        
        x,y = np.nonzero(1-mask)
    im1_pixel = im1[x,y]
    im2_pixel = im2[x,y]

    ux = np.mean(im1_pixel)
    uy = np.mean(im2_pixel)
    uxy = np.mean(im1_pixel*im2_pixel)
    vx = np.var(im1_pixel)
    vy = np.var(im2_pixel)
    vxy = uxy - ux*uy

    ssim = (2*ux*uy+C1)*(2*vxy+C2)/((ux**2+uy**2+C1)*(vx+vy+C2))
    return ssim, S

def metrics(args, id, output_images, masks, scene_images=None, source=None):   
    if args.blend_type != 'L' and len(output_images)!=len(scene_images):
        raise ValueError('output images should have the same number as scene images')
    if args.save:
        save_file = os.path.join(args.save_root, 'output.txt') 
        print(f"save metrics to {save_file}")       
    result = {'p':[], 's':[], 'ce':[], 'fail':0, 'psnr_heatmap':[], 'ssim_heatmap':[]}    
    if args.save:
        with open(save_file, 'a') as f:
            f.write('id: {}\n'.format(str(id).zfill(4)))
    print('metric')
    for i, output in tqdm(enumerate(output_images), total=len(output_images)):        
        if output is None: 
            result['p'].append(0)
            result['s'].append(-1)
            result['ce'].append(-1)
            result['fail'] = result['fail']+1            
            if args.save:
                with open(save_file, 'a') as f:
                    f.write('{:s}_{:d}\tFailed\n'.format(args.blend_type, i))
            continue

        if args.blend_type == 'L':
            scene = source
        elif args.blend_type == 'mix':
            scene = scene_images[3*int(i/3)]
        else:
            scene = scene_images[i]
        
        mask = masks[i]                    
        if args.heatmap:
            p, psnr_heatmap = PSNR(output, scene, mask=mask, heatmap=args.heatmap, with_mask=args.with_mask)
            s, ssim_heatmap = SSIM(output, scene, multichannel=True, mask=mask, heatmap=args.heatmap, with_mask=args.with_mask)
            result['psnr_heatmap'].append(psnr_heatmap)
            result['ssim_heatmap'].append(ssim_heatmap)
        else:
            p = PSNR(output, scene, mask=mask, heatmap=args.heatmap, with_mask=args.with_mask)
            s = SSIM(output, scene, multichannel=True, mask=mask, heatmap=args.heatmap, with_mask=args.with_mask)                                        
            
        result['p'].append(round(p, 2))
        result['s'].append(round(s, 2))          
                    
        if args.save:
            with open(save_file, 'a') as f:            
                f.write('{:s}_{:d}\t{:.2f}\t{:.2f}\n'.format(args.blend_type, i, p, s))
    return result