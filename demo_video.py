import os
os.environ["OMP_NUM_THREADS"]="1"
os.environ["MKL_NUM_THREADS"]="1"
os.environ["CUDA_VISIBLE_DEVICES"]="4"

from skimage import io
from matplotlib import pyplot as plt
import cv2
import numpy as np
import torch
from tqdm import tqdm
import torch.nn.functional as F
import sys

# model path
sys.path.append('./core')
from flow_estimator import Flow_estimator
from config import get_demo_video_args, get_life_args

def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    imgs = []
    plt.figure()
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:            
            imgs.append(frame)            
        else:
            break
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    print(f"Load {len(imgs)} frames")
    return imgs, fps

def save_video(imgs, size, video_path, fps=30):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(video_path, fourcc, fps, size)
    for frame in imgs:
        out.write(frame)
    out.release()
    print('video saved')
    
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

def blend(estimator, marker, scene, frame, args, warp = 'homography'):
    H, W = 480, 640
    scene_ori_H, scene_ori_W = scene.shape[:2]
    frame_ori_H, frame_ori_W = frame.shape[:2]
    marker_ori_H, marker_ori_W = marker.shape[:2]
    zero = np.zeros_like(marker)
    
    if frame_ori_H > frame_ori_W:
        ratio = marker_ori_H / frame_ori_H
        frame = cv2.resize(frame, None, fx=ratio, fy=ratio)
        frame_H, frame_W = frame.shape[:2]
        start_x = int(marker_ori_W/2 - frame_W/2)
        zero[0:frame_H, start_x:start_x+frame_W] = frame
    
    else:
        ratio = marker_ori_W / frame_ori_W
        frame = cv2.resize(frame, None, fx=ratio, fy=ratio)
        frame_H, frame_W = frame.shape[:2]
        start_y = int(marker_ori_H/2 - frame_H/2)
        zero[start_y:start_y+frame_H, 0:frame_W] = frame
    frame = zero     
    
    marker = cv2.resize(marker, (W, H))
    scene = cv2.resize(scene, (W, H))
    frame = cv2.resize(frame, (W, H))      
    
    flow = estimator.estimate(scene, marker)
    frame = cv2.GaussianBlur(frame,(5,5),1,borderType=cv2.BORDER_CONSTANT)
    
    if warp == 'grid_sample':    
        out = image_flow_warp(frame, flow[0].permute([1,2,0]),padding_mode='zeros')
        mask_origin = (np.ones(shape=(frame.shape[0], frame.shape[1], 1)) * 255).astype(np.uint8)
        mask_origin = image_flow_warp(mask_origin, flow[0].permute([1,2,0]),padding_mode='zeros')
        mask = mask_origin.astype(np.float64) / 255.0  
    elif warp == 'homography':    
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
        out = cv2.warpPerspective(frame, M, (scene.shape[1], scene.shape[0]))    
        mask_origin = (np.ones(shape=(frame.shape[0], frame.shape[1], 1)) * 255).astype(np.uint8)
        mask = cv2.warpPerspective(mask_origin, M, (mask_origin.shape[1], mask_origin.shape[0]))    
        mask = mask.astype(np.float64)/255.0
    
    mask = cv2.GaussianBlur(mask,(3,3),1, borderType=cv2.BORDER_REPLICATE)
    if len(mask.shape)==2:
        mask = mask[:,:,np.newaxis]
    blend = (out * mask + scene * (1-mask)).astype(np.uint8)
    # blend = cv2.resize(blend, (scene_ori_W, scene_ori_H))
    
    if args.draw:
        plt.figure(figsize=(16,10),facecolor='white')
        plt.subplot(231), plt.imshow(scene[:,:,::-1]), plt.title('scene'), plt.axis('off')
        plt.subplot(232), plt.imshow(marker[:,:,::-1]), plt.title('marker'), plt.axis('off')
        plt.subplot(233), plt.imshow(frame[:,:,::-1]), plt.title('frame'), plt.axis('off')
        plt.subplot(234), plt.imshow(out[:,:,::-1]), plt.title('out'), plt.axis('off')
        plt.subplot(235), plt.imshow(mask), plt.title('mask'), plt.axis('off')
        plt.subplot(236), plt.imshow(blend[:,:,::-1]), plt.title('blend'), plt.axis('off')
        plt.show()
    if args.save:
        cv2.imwrite(os.path.join(args.demo_root, 'scene.png'), scene)
        cv2.imwrite(os.path.join(args.demo_root, 'frame.png'), frame)        
        cv2.imwrite(os.path.join(args.demo_root, 'blend.png'), blend)
    
    return blend  
    
def demo():           
    args = get_demo_video_args()
    scene_path = os.path.join(args.demo_root, args.scene_name)
    marker_path = os.path.join(args.demo_root, args.marker_name)
    movie_path = os.path.join(args.demo_root, args.movie_name)
    save_path = os.path.join(args.demo_root, args.save_name)
    print('===> Path Config')
    print(f"marker: {marker_path}")
    print(f"scene video: {scene_path}")
    print(f"movie: {movie_path}")
    print(f"save to: {save_path}")

    print('\n===> Loading marker image')
    marker = cv2.imread(marker_path)

    print('===> Loading scene video')
    scenes, _ = read_video(scene_path)
    scenes = scenes[args.scene_start_idx:]

    print('===> Loading movie')
    frames, fps = read_video(movie_path)
    frames = frames[args.movie_start_idx:]  

    print('===> Loading Model\n')  
    model_args = get_life_args()        
    estimator = Flow_estimator(model_args)
    
    if args.test:
        print('> Test mode')
        if args.draw:
            plt.figure()        
            plt.imshow(scenes[args.scene_id][:,:,::-1])
        blend(estimator, marker, scenes[args.scene_id], frames[args.frame_id], args)
    else:
        print('> Video saving mode')
        imgs = []
        for i, scene in tqdm(enumerate(scenes), total = len(scenes)):
            frame = frames[i%len(frames)]      
            if i == 0:
                args.save = True
            else:
                args.save = False      
            imgs.append(blend(estimator, marker, scene, frame, args))
        save_video(imgs, fps=fps, size=(imgs[0].shape[1], imgs[0].shape[0]), video_path=save_path)

if __name__=='__main__': 
    demo()