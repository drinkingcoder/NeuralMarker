import os
os.environ["OMP_NUM_THREADS"]="1"
os.environ["MKL_NUM_THREADS"]="1"
os.environ["CUDA_VISIBLE_DEVICES"]="4"
os.environ["CUDA_LAUNCH_BLOCKING"]="1"

from skimage import io
from matplotlib import pyplot as plt
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
import warnings
from tqdm import tqdm
import datetime
from PIL import Image

import sys

# our model
sys.path.append('./core')
from flow_estimator import Flow_estimator
from config import get_twins_args, get_eval_args

torch.set_grad_enabled(False)

from colormap import get_colormap
from eval_utils import resize_lighting, resize_viewpoint
from metrics import metrics

def coords_grid(batch, ht, wd):
    coords = torch.meshgrid(torch.arange(ht), torch.arange(wd))    
    coords = torch.stack(coords[::-1], dim=0).float()     
    return coords[None].repeat(batch, 1, 1, 1)        
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
    grid = (flow + coords).permute([0, 2, 3, 1]).contiguous()   # (1, H, W, 2)  
    
    grid[:, :, :, 0] = (grid[:, :, :, 0] * 2 - W + 1) / (W - 1)
    grid[:, :, :, 1] = (grid[:, :, :, 1] * 2 - H + 1) / (H - 1)    
    image = image[None].permute([0, 3, 1, 2]).cuda().float()
    
    outImg = F.grid_sample(image, grid, padding_mode=padding_mode, align_corners=False)[0].cpu().numpy().transpose([1, 2, 0])
    return outImg

def blend(out, source, scene, blend_type, mask=None, use_colormap=False):
    if mask is None:
        intensity = np.linalg.norm(out, axis=2)
        mask = (intensity == 0)[:,:,np.newaxis]   
    else:
        mask = mask
    scene = source if blend_type=='L' or blend_type == 'mix' else scene
    if not use_colormap:
        result = (out * (1 - mask) + scene * mask).astype(np.uint8) 
    else:
        colormap = cv2.addWeighted(out, 0.5, scene, 0.5, 0)
        result = (colormap * (1 - mask) + scene * mask).astype(np.uint8) 
    return result, mask

def blend_pdc(scene_images, marker, flow_estimator, estimate_uncertainty, source=None, blend_type='D', use_colormap=True):    
    print('blend with pdc')
    blends = []
    masks = []
    blends_colormap = []    
    for id, scene in tqdm(enumerate(scene_images), total=len(scene_images)):                 
        marker = cv2.resize(marker, (scene.shape[1], scene.shape[0]))        
        Is_original = np.ascontiguousarray(marker)
        It_original = np.ascontiguousarray(scene)
        Is_tensor = torch.from_numpy(Is_original).permute(2,0,1).unsqueeze(0)
        It_tensor = torch.from_numpy(It_original).permute(2,0,1).unsqueeze(0)
        with torch.no_grad():
            flow, uncertainty_est = flow_estimator.estimate_flow_and_confidence_map(Is_tensor, It_tensor)                         

        out = image_flow_warp(marker, flow[0].permute([1,2,0]))
        mask_origin = np.ones(shape=(marker.shape[0], marker.shape[1], 1)).astype(np.float64) 
        mask_origin = image_flow_warp(mask_origin, flow[0].permute([1,2,0]),padding_mode='zeros')
        mask = (1 - mask_origin)        
             
        if blend_type == 'mix':
            scene_id = 3 * int(id / 3)
            source = scene_images[scene_id]
        blend_i, mask_i = blend(out, source, scene, blend_type, mask=mask)         
        blends.append(blend_i)
        masks.append(mask_i)
        
        if use_colormap:
            colormap = get_colormap(flow, scene.shape[0], scene.shape[1]) 
            out_colormap =  image_flow_warp(colormap, flow[0].permute([1,2,0])) 
            out_colormap = ((out_colormap + 256) / 2)            
            mask_colormap = np.ones_like(colormap).astype(np.float32) / 2
            mask_colormap = image_flow_warp(mask_colormap, flow[0].permute([1,2,0]))
            mask_colormap = (1 - mask_colormap)
            if blend_type == 'L' or blend_type == 'mix':
                scene = source            
            blend_colormap = (out_colormap * (1 - mask_colormap) + scene * mask_colormap).astype(np.uint8) 
            blends_colormap.append(blend_colormap)
    return blends, masks, blends_colormap
        
def blend_life(scene_images, marker, estimator, source=None, blend_type='D', warp='grid_sample', use_colormap=True):        
    print('blend with our model')
    blends = []
    masks = []
    blends_colormap = []    
    for id, scene in tqdm(enumerate(scene_images), total=len(scene_images)):                 
        marker = cv2.resize(marker, (scene.shape[1], scene.shape[0]))        
        if blend_type == "mix":
            source_id = 3 * int(id / 3)
            source = scene_images[source_id]            

        flow = estimator.estimate(scene, marker)           
        
        if use_colormap:
            colormap = get_colormap(flow, scene.shape[0], scene.shape[1])            

        mask = None
        if warp == 'grid_sample':
            out = image_flow_warp(marker, flow[0].permute([1,2,0]))              
            mask_origin = np.ones(shape=(marker.shape[0], marker.shape[1], 1)).astype(np.float64) 
            mask_origin = image_flow_warp(mask_origin, flow[0].permute([1,2,0]),padding_mode='zeros')
            mask = (1 - mask_origin)
            if use_colormap:
                out_colormap =  image_flow_warp(colormap, flow[0].permute([1,2,0])) 
                out_colormap = ((out_colormap + 256) / 2)                   
                mask_colormap = np.ones_like(colormap).astype(np.float32) / 2
                mask_colormap = image_flow_warp(mask_colormap, flow[0].permute([1,2,0]))
                mask_colormap = (1 - mask_colormap) 
                      
        elif warp == 'homography':
        # RANSAC homography
            flow = flow[0].permute([1,2,0])
            image = marker            
            image = torch.from_numpy(image)
            if image.ndim == 2:
                image = image[None].permute([1,2,0])
            H, W, _ = image.shape
            coords = coords_grid(1, H, W).cuda().float().contiguous()
            flow = flow[None].repeat(1, 1, 1, 1).permute([0, 3, 1, 2]).float().contiguous()    
            grid = (flow + coords).permute([0, 2, 3, 1]).contiguous()   # (1, H, W, 2)
            grid = grid[0].cpu()
            src_pts = []
            dst_pts = []
            for y in range(H):
                for x in range(W):
                    if grid[y,x,0]>=0 and grid[y,x,0]<W and grid[y,x,1]>=0 and grid[y,x,1]<H:
                        src_pts.append((grid[y,x,0], grid[y,x,1]))
                        dst_pts.append((x, y))                        
            src_pts = np.float32(src_pts)
            dst_pts = np.float32(dst_pts)

            M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if M is None:
                blends.append(None)
                continue
            out = cv2.warpPerspective(marker, M, (scene.shape[1], scene.shape[0]))        

        blend_i, mask_i = blend(out, source, scene, blend_type, mask=mask)        
        blends.append(blend_i)
        masks.append(mask_i)
        if use_colormap:     
            if blend_type == 'L' or blend_type == 'mix':
                scene = source
            blend_colormap = (out_colormap * (1 - mask_colormap) + scene * mask_colormap).astype(np.uint8) 
            blends_colormap.append(blend_colormap)
    return blends, masks, blends_colormap

def blend_RANSAC(scene_images, marker, coarseModel=None, network=None, source=None, blend_type='D', use_colormap=False):
    print('blend with ransac-flow')
    blends = []
    masks = []   
    blends_colormap = []    
    for idx, scene in tqdm(enumerate(scene_images), total=len(scene_images)):    
        scene_ = Image.fromarray(scene)     
        marker_ = Image.fromarray(marker)     
        marker_ = marker_.resize(scene_.size)        
        coarseModel.setSource(marker_)
        coarseModel.setTarget(scene_)        

        I2w, I2h = coarseModel.It.size
        featt = F.normalize(network['netFeatCoarse'](coarseModel.ItTensor))
                    
        #### -- grid     
        gridY = torch.linspace(-1, 1, steps = I2h).view(1, -1, 1, 1).expand(1, I2h,  I2w, 1)
        gridX = torch.linspace(-1, 1, steps = I2w).view(1, 1, -1, 1).expand(1, I2h,  I2w, 1)
        grid = torch.cat((gridX, gridY), dim=3).cuda() 
        warper = tgm.HomographyWarper(I2h,  I2w)

        bestPara, InlierMask = coarseModel.getCoarse(np.zeros((I2h, I2w)))
        bestPara = torch.from_numpy(bestPara).unsqueeze(0).cuda()

        flowCoarse = warper.warp_grid(bestPara)        
        I1_coarse = F.grid_sample(coarseModel.IsTensor, flowCoarse)        

        featsSample = F.normalize(network['netFeatCoarse'](I1_coarse.cuda()))

        corr12 = network['netCorr'](featt, featsSample)
        flowDown8 = network['netFlowCoarse'](corr12, False) ## output is with dimension B, 2, W, H

        flowUp = F.interpolate(flowDown8, size=(grid.size()[1], grid.size()[2]), mode='bilinear')
        flowUp = flowUp.permute(0, 2, 3, 1)

        flowUp = flowUp + grid

        flow12 = F.grid_sample(flowCoarse.permute(0, 3, 1, 2), flowUp).permute(0, 2, 3, 1).contiguous()        
        
        I1_fine = F.grid_sample(coarseModel.IsTensor, flow12)
        I1_fine_pil = transforms.ToPILImage()(I1_fine.cpu().squeeze())        
                        
        if blend_type == "mix":
            source_id = 3 * int(idx / 3)
            source = scene_images[source_id] 
        blend_i, mask_i = blend(np.array(I1_fine_pil), source, scene, blend_type)        
        blends.append(blend_i)
        masks.append(mask_i)

        if use_colormap:    
            colormap = Image.open('./colormap.jpg')
            coarseModel.setSource(colormap)
            colormap_fine = F.grid_sample(coarseModel.IsTensor, flow12)
            out_colormap = transforms.ToPILImage()(colormap_fine.cpu().squeeze())
            blend_colormap, _ = blend(np.array(out_colormap), source, scene, blend_type, use_colormap=use_colormap)#, mask=mask)
            blends_colormap.append(blend_colormap)
        
    return blends, masks, blends_colormap

def blend_homography(scene_images, marker, source=None, blend_type='D', detector='SIFT', use_colormap=True):
    print('blend with homography')
    if detector == 'ORB':
        detect = cv2.ORB_create()    
    elif detector == 'SIFT':
        detect = cv2.SIFT_create()
    else:
        raise ValueError('detector not implemented')
    blends = []
    masks = []
    blends_colormap = []
    bar = tqdm(enumerate(scene_images), total=len(scene_images))
    for idx, scene in bar:  
        bar.set_description('Editing %d' % idx)          
        marker = cv2.resize(marker, (scene.shape[1], scene.shape[0]))   
        kp1, des1 = detect.detectAndCompute(marker, None)
        kp2, des2 = detect.detectAndCompute(scene, None)           
        if des1 is None or des2 is None:
            blends.append(None)   
            continue
        
        if detector == 'ORB':
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)                
            matches = bf.match(des1, des2)
            
        elif detector == 'SIFT':
            FLANN_INDEX_KDTREE = 0
            indexParams = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            searchParams = dict(checks=50)
            flann = cv2.FlannBasedMatcher(indexParams, searchParams)    
            if len(des1)<2 or len(des2)<2:
                blends.append(None)
                continue
            matches = flann.knnMatch(des1, des2, k=2)
            matches = [m for m,n in matches if m.distance < 0.7*n.distance]
        else:
            raise ValueError('detector {} not implemented'.format(detector))
        if len(matches) < 4:
            blends.append(None)
            continue        

        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches[:50]]).reshape(-1,1,2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches[:50]]).reshape(-1,1,2)        
        M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if M is None:
            blends.append(None)
            continue        
        out = cv2.warpPerspective(marker, M, (scene.shape[1], scene.shape[0])) 

        blend_i, mask_i = blend(out, source, scene, blend_type)

        if use_colormap:
            colormap = np.asarray(Image.open('./colormap.jpg'))            
            colormap = cv2.resize(colormap[:,:,::-1], (scene.shape[1], scene.shape[0])) 
            blend_colormap = cv2.warpPerspective(colormap, M, (scene.shape[1], scene.shape[0])) 
            blend_colormap, _ = blend(blend_colormap, source, scene, blend_type, use_colormap=use_colormap)
            blends_colormap.append(blend_colormap)            
        
        blends.append(blend_i)
        masks.append(mask_i)

    return blends, masks, blends_colormap

def blend_SPSG(scene_images, marker, matching=None, source=None, blend_type='D', use_colormap=True):
    print('blend with SPSG')
    blends = []
    blends_colormap = []
    masks = []
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    for scene in scene_images:
        marker = cv2.resize(marker, (scene.shape[1], scene.shape[0]))
        marker_gray = cv2.cvtColor(marker, cv2.COLOR_BGR2GRAY)
        scene_gray = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)
        inp0 = frame2tensor(marker_gray, device=device)
        inp1 = frame2tensor(scene_gray, device=device)        
        pred = matching({'image0': inp0, 'image1': inp1})
        pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
        kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
        matches, conf = pred['matches0'], pred['matching_scores0']

        valid = matches > -1          
        mkpts0 = kpts0[valid]
        mkpts1 = kpts1[matches[valid]] 
        mconf = conf[valid]
        
        valid = mconf > 0.5
        mkpts0 = mkpts0[valid]
        mkpts1 = mkpts1[valid]
        
        if np.count_nonzero(valid) < 4:
            blends.append(None)
            masks.append(None)
            blends_colormap.append(None)
            continue
        
        src_pts = np.float32(mkpts0).reshape(-1,1,2)
        dst_pts = np.float32(mkpts1).reshape(-1,1,2)
        M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if M is None:
            blends.append(None)
            masks.append(None)
            blends_colormap.append(None)
            continue
        out = cv2.warpPerspective(marker, M, (scene.shape[1], scene.shape[0]))        

        blend_i, mask_i = blend(out, source, scene, blend_type)        
        blends.append(blend_i)
        masks.append(mask_i)    

        if use_colormap:
            colormap = np.asarray(Image.open('./colormap.jpg'))            
            colormap = cv2.resize(colormap[:,:,::-1], (scene.shape[1], scene.shape[0])) 
            blend_colormap = cv2.warpPerspective(colormap, M, (scene.shape[1], scene.shape[0])) 
            blend_colormap, _ = blend(blend_colormap, source, scene, blend_type)
            blends_colormap.append(blend_colormap)            
    return blends, masks, blends_colormap

def eval(args, id, scene_images, marker, source=None,    
         estimator=None, 
         coarseModel=None, network=None, 
         flow_estimator=None, estimate_uncertainty=None, 
         matching=None,                  
):   
    '''
    Args:
        @id: (int) marker image id, for logging
        @scene_images: (list) test scene images
        @marker: (array) marker image
        @source: source image to calculate metrics when blend_type=='lighting'            
        @estimator: twins-onestage
        @flow_estimator & estimate_uncertainty: pdc        
        @coarseModel & network: ransac-flow
        @matching: SPSG                
        @args: other args
    '''
    if args.blend_method == 'twins-onestage':
        if estimator is None:
            raise ValueError('estimator not set')        
        out, mask, out_colormap = blend_life(scene_images=scene_images, marker=marker, source=source, estimator=estimator, warp=args.warp, 
                                            blend_type=args.blend_type, use_colormap=args.use_colormap)
    
    elif args.blend_method == 'homography':        
        out, mask, out_colormap = blend_homography(scene_images=scene_images, marker=marker, source=source, detector=args.detector, 
                                            blend_type=args.blend_type, use_colormap=args.use_colormap)
    elif args.blend_method == 'pdc':
        if flow_estimator is None:
            raise ValueError('estimator not set')
        out, mask, out_colormap = blend_pdc(scene_images=scene_images, marker=marker, source=source, flow_estimator=flow_estimator, estimate_uncertainty=estimate_uncertainty, 
                                            blend_type=args.blend_type, use_colormap=args.use_colormap)
    
    elif args.blend_method == 'ransac-flow':
        if coarseModel is None or network is None:
            raise ValueError('coarseModel not set')
        out, mask, out_colormap = blend_RANSAC(scene_images=scene_images, marker=marker, source=source, coarseModel=coarseModel, network=network, 
                                            blend_type=args.blend_type, use_colormap=args.use_colormap)

    elif args.blend_method == 'SPSG':                
        out, mask, out_colormap = blend_SPSG(scene_images=scene_images, marker=marker, source=source, matching=matching, 
                                            blend_type=args.blend_type, use_colormap=args.use_colormap)    

        
    if args.blend_type == 'L':
        result = metrics(args=args, id=id, output_images=out, masks=mask, source=source)
    else:
        result = metrics(args=args, id=id, output_images=out, masks=mask, scene_images=scene_images)
        
    if args.draw or args.save: 
        # save_root = os.path.join(save_root, str(id).zfill(4))
        # if save and not os.path.exists(save_root):
        #     os.makedirs(save_root)
        for i in range(len(scene_images)): 
            if out[i] is None:
                continue                           
            
            if not args.with_mask:
                if args.blend_type == 'V':
                    scene_images[i] = resize_viewpoint(scene_images[i])
                    out[i] = resize_viewpoint(out[i])
                    mask[i] = resize_viewpoint(mask[i])
                    if args.use_colormap:
                        out_colormap[i] = resize_viewpoint(out_colormap[i])                    
                elif args.blend_type == 'L':
                    scene_images[i] = resize_lighting(scene_images[i])
                    out[i] = resize_lighting(out[i])
                    mask[i] = resize_lighting(mask[i])
                    if args.use_colormap:
                        out_colormap[i] = resize_lighting(out_colormap[i])
            if args.draw: 
                plt.figure(figsize=(16,12),facecolor='white')
                title = 'PSNR: '+str(result['p'][i])+' SSIM: '+str(result['s'][i])
                if args.use_colormap:                    
                    plt.figure(facecolor='white')
                    plt.subplot(1, 3, 1)
                    plt.imshow(scene_images[i]), plt.axis('off'), plt.title('scene '+str(i).zfill(4))        
                    plt.subplot(1, 3, 2) 
                    plt.imshow(mask[i]), plt.axis('off')
                    plt.subplot(1, 3, 3) 
                    plt.imshow(out_colormap[i]), plt.axis('off')
                else:
                    plt.subplot(1, 3, 1)
                    plt.imshow(scene_images[i]), plt.axis('off'), plt.title('scene '+str(i).zfill(4))        
                    plt.subplot(1, 3, 2) 
                    plt.imshow(out[i]), plt.axis('off'), plt.title(title)
                    plt.subplot(1, 3, 3)
                    plt.imshow(mask[i]), plt.axis('off')
                # plt.subplot(1, 2, 2)
                # if with_mask:
                #     plt.imshow(out[i]*(1-mask[i])), plt.axis('off') #, plt.title(title)  
                # else:                    
                #     plt.imshow(out[i]), plt.axis('off')                
              
            if args.save: 
                # plt.savefig(os.path.join(save_root,str(i)+'_'+title+'.png'), dpi=200, bbox_inches='tight')                            
                io.imsave(os.path.join(args.save_root, str(id).zfill(4)+'_'+str(i)+'_out'+'.png'), out[i].astype(np.uint8))                
                io.imsave(os.path.join(args.save_root, str(id).zfill(4)+'_'+str(i)+'_scene'+'.png'), scene_images[i].astype(np.uint8))
                if args.use_colormap:
                    io.imsave(os.path.join(args.save_root, str(id).zfill(4)+'_'+str(i)+'_out_colormap'+'.png'), out_colormap[i])
                if args.heatmap:
                    io.imsave(os.path.join(args.save_root, str(id).zfill(4)+'_'+str(i)+'_psnr_heatmap'+'.png'), resize_lighting(result['psnr_heatmap'][i]))
                    io.imsave(os.path.join(args.save_root, str(id).zfill(4)+'_'+str(i)+'_ssim_heatmap'+'.png'), resize_lighting(result['ssim_heatmap'][i]))
            plt.show()
            
    return result

def run():    
    args = get_eval_args()

    H, W = args.img_shape       
    if args.blend_type == 'L':
        if args.source_id == -1:
            raise FileExistsError('no source image id specified.') 
    
    # ============== Environ Setting ================       
    print('gpu id:', os.environ["CUDA_VISIBLE_DEVICES"]) 

    # ============== Path Setting ===================
    print('\n===> Path Config')
    # marker / scene images root
    marker_root = os.path.join(args.root, 'marker')
    if args.folder != ""   :
        scene_root = os.path.join(args.root, args.folder) 
        args.save_root = os.path.join(args.save_root, args.folder)
    else:
        assert args.blend_type in ['D','V','L']
        scene_root = os.path.join(args.root, args.blend_type)
        args.save_root = os.path.join(args.save_root, args.blend_type)
    assert os.path.exists(scene_root) and os.path.exists(marker_root) 
    print(f"Loading marker from: {marker_root}")
    print(f"Loading scene image from: {scene_root}")
    # save root
    if args.save:
        time = datetime.datetime.now()
        suffix = datetime.datetime.strftime(time, '%m%d%H')
        if args.with_mask:
            args.save_root = os.path.join(args.save_root, args.blend_method+'_mask_'+suffix)            
        else:
            args.save_root = os.path.join(args.save_root, args.blend_method+'_'+suffix)
        os.makedirs(args.save_root, exist_ok=True)
        print('Save image to: '+args.save_root)          

    # =============== Data Loading ===================
    print('\n===> Data Loading')
    blend_type = {'D':'deformation', 'V':'viewpoint', 'L':'light'}
    if H > W:
        H, W = W, H #keep H < W
    scene_images = []
    marker_images = []
    for id in tqdm(range(args.start_img_id, args.start_img_id+args.img_num), desc='Loading Image', total=args.img_num):
        scene_image = []
        for i in range(args.start_scene_id, args.start_scene_id+args.scn_num):            
            scene_path = os.path.join(scene_root, str(id).zfill(4) + "_" + blend_type[args.blend_type] + "_"+str(i)+".jpg")            
            scene = io.imread(scene_path)
            ## numpy shape(height, width); cv size (width, height)
            if scene.shape[0] > scene.shape[1]: # origin H > origin W 
                scene = cv2.resize(scene, (H, W))
            else: # origin H < origin W
                scene = cv2.resize(scene, (W, H))
            scene_image.append(scene)
        scene_images.append(scene_image)

        marker_path = os.path.join(marker_root, str(id).zfill(4) + ".jpg")
        marker_image = io.imread(marker_path)
        if marker_image.shape[0] > marker_image.shape[1]: # H > W
            marker_image = cv2.resize(marker_image, (H, W))
        else:
            marker_image = cv2.resize(marker_image, (W, H))        
        marker_images.append(marker_image)

    total = len(scene_images)*len(scene_images[0])
    print('input {}x{} scene images'.format(len(scene_images), len(scene_images[0])))    
    print('input {} marker images'.format(len(marker_images)))        
    # show_input_images(scene_images, marker_images)

    # ================= Model Loading ====================    
    assert args.blend_method in ['homography', 'pdc', 'twins-onestage', 'ransac-flow', 'SPSG']
    print('\n===> Model Loading')
    print('blend method:', args.blend_method)            
    if args.blend_method == 'pdc':
        model_args = easydic({
            'model': 'PDCNet',
            'pre_trained_model_type': 'megadepth',
            'path_to_pre_trained_models': os.path.abspath('./pretrained/DenseMatching'),
            'local_optim_iter': 7,
            'optim_iter': 3,
            'network_type': None
        })        
        flow_estimator, estimate_uncertainty = select_model(
            model_args.model, 
            model_args.pre_trained_model_type, 
            model_args,
            model_args.optim_iter, 
            model_args.local_optim_iter,
            path_to_pre_trained_models=model_args.path_to_pre_trained_models)

    elif args.blend_method == 'ransac-flow':        
        resumePth = './pre_trained_model/RANSAC-Flow/MegaDepth_Theta1_Eta001_Grad1_0.774.pth' ## model for visualization
        kernelSize = 7
        network = {'netFeatCoarse' : model.FeatureExtractor(), 
                'netCorr'       : model.CorrNeigh(kernelSize),
                'netFlowCoarse' : model.NetFlowCoarse(kernelSize), 
                'netMatch'      : model.NetMatchability(kernelSize),
                }
        for key in list(network.keys()) : 
            network[key].cuda()                
        param = torch.load(resumePth)
        msg = 'Loading pretrained model from {}'.format(resumePth)
        print(msg)
        for key in list(param.keys()) : 
            network[key].load_state_dict( param[key] ) 
            network[key].eval()
        nbScale = 7
        coarseIter = 10000
        coarsetolerance = 0.05
        minSize = 640
        imageNet = True # we can also use MOCO feature here
        scaleR = 1.2 
        coarseModel = CoarseAlign(nbScale, coarseIter, coarsetolerance, 'Homography', minSize, 1, True, imageNet, scaleR)      
    
    elif args.blend_method == 'SPSG':
        opt = easydic({
            'nms_radius': 4,
            'keypoint_threshold': 0.005,
            'max_keypoints': 1024,
            'superglue': 'indoor',
            'sinkhorn_iterations': 20,
            'match_threshold': 0.2,
            'sp_checkpoint': './pre_trained_model/SPSG/superpoint_v1.pth',
            'sg_checkpoint': './pre_trained_model/SPSG/superglue_indoor.pth',
        })
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('Running inference on device \"{}\"'.format(device))
        config = {
            'superpoint': {
                'nms_radius': opt.nms_radius,
                'keypoint_threshold': opt.keypoint_threshold,
                'max_keypoints': opt.max_keypoints,
                'checkpoint': opt.sp_checkpoint,
            },
            'superglue': {
                'weights': opt.superglue,
                'sinkhorn_iterations': opt.sinkhorn_iterations,
                'match_threshold': opt.match_threshold,
                'checkpoint': opt.sg_checkpoint,
            }
        }
        matching = Matching(config).eval().to(device)

    elif args.blend_method == 'homography':
        print('detector: '+ args.detector)

    elif args.blend_method == 'twins-onestage':
        model_args = get_twins_args()                
        print('pretrained model:', model_args.model)
        estimator = Flow_estimator(model_args)        
        print('warp method:', args.warp)
                    
    p = []
    s = []
    fail = 0
    for id in range(args.img_num):
        scene = scene_images[id]
        marker = marker_images[id]        
        source = scene[args.source_id]  
        if args.blend_method == 'twins-onestage':
            result = eval(args=args, id=id, scene_images=scene, marker=marker, source=source, estimator=estimator)

        elif args.blend_method == 'pdc':
            result = eval(args=args, id=id, scene_images=scene, marker=marker, source=source, 
                        flow_estimator=flow_estimator, estimate_uncertainty=estimate_uncertainty)

        elif args.blend_method == 'ransac-flow':
            result = eval(args=args, id=id, scene_images=scene, marker=marker, source=source,
                        coarseModel=coarseModel, network=network)

        elif args.blend_method == 'homography':
            result = eval(args=args, id=id, scene_images=scene, marker=marker, source=source)

        elif args.blend_method == 'SPSG':
            result = eval(args=args, id=id, scene_images=scene, marker=marker, source=source, matching=matching)

        p.append([result['p'][i] for i in range(len(result['p'])) if result['p'][i]!=0])
        s.append([result['s'][i] for i in range(len(result['s'])) if result['s'][i]!=-1])
        fail = fail + result['fail']
        
    p_array = np.asarray([item for sub in p for item in sub])    
    s_array = np.asarray([item for sub in s for item in sub])    
    
    print(args.blend_method+'\t PSNR: {:.2f}/{:.2f}\t SSIM: {:.2f}/{:.2f}\t fail: {:.2f}%({}/{})'
        .format(np.mean(p_array), 
                np.median(p_array),
                np.mean(s_array), 
                np.median(s_array),
                fail*1.0/total*100,
                fail, total))        

if __name__ == '__main__':        
    # run(blend_type='deformation', blend_method='twins-onestage', warp='grid_sample', img_num=10, scn_num=10, source_id=1, with_mask=True, multisample=True, folder="minions", start_img_id=0, start_scene_id = 0, draw=False, save=True, detector="SIFT")
    # run(blend_type='viewpoint', blend_method='twins-onestage', warp='grid_sample', img_num=10, scn_num=10, source_id=1, with_mask=True, multisample=True, folder="minions", start_img_id=0, start_scene_id = 0, draw=False, save=True, detector="SIFT")
    # run(blend_type='light', blend_method='twins-onestage', warp='grid_sample', img_num=10, scn_num=10, source_id=1, with_mask=True, multisample=True, folder="minions", start_img_id=0, start_scene_id = 0, draw=False, save=True, detector="SIFT")    
    run()

