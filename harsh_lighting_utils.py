import cv2
import numpy as np
import os
from third_party.NIID.decompose import decompose_image

# model path
from core.utils.utils import image_flow_warp

DEBUG = False

def srgb_to_rgb(srgb):
    ret = np.zeros_like(srgb)
    idx0 = srgb <= 0.04045
    idx1 = srgb > 0.04045
    ret[idx0] = srgb[idx0] / 12.92
    ret[idx1] = np.power((srgb[idx1] + 0.055) / 1.055, 2.4)
    return ret

def rgb_to_srgb(rgb):
    ret = np.zeros_like(rgb)
    idx0 = rgb <= 0.0031308
    idx1 = rgb > 0.0031308
    ret[idx0] = rgb[idx0] * 12.92
    ret[idx1] = np.power(1.055 * rgb[idx1], 1.0 / 2.4) - 0.055
    return ret

'''
render image with rgb, illumination and reflectance
'''
def render(decompose_result, replace_result, poster_brightness: float, data_root: str, mask = None, save=False):
    replace = replace_result.astype(np.float64)
    illumination = decompose_result['pred_S'][:,:,::-1].astype(np.float64)
    reflectance = decompose_result['pred_R'][:,:,::-1].astype(np.float64)
    rgb = decompose_result['rgb'][:,:,::-1].astype(np.float64)
    if DEBUG:
        render = cv2.multiply(illumination, reflectance)
        cv2.imshow("srgb", rgb_to_srgb(rgb))
        cv2.imshow("s", rgb_to_srgb(illumination))
        cv2.imshow("r", rgb_to_srgb(reflectance))
        cv2.imshow("render", render)
        cv2.imshow("srender", rgb_to_srgb(render))
        cv2.waitKey()
        cv2.destroyAllWindows()
    if save:
        ofs = cv2.FileStorage(os.path.join(data_root, "split.yml"), cv2.FILE_STORAGE_WRITE)
        ofs.write("illumination", illumination)
        ofs.write("reflectance", reflectance)
        ofs.write("rgb", rgb)

    reflectance = reflectance * 255
    rgb = rgb * 255
    if mask is None:        
        assert os.path.exists(os.path.join(data_root, "mask.jpg"))
        mask = cv2.imread(os.path.join(data_root, "mask.jpg"), cv2.IMREAD_UNCHANGED)   
        mask = mask / 255.0     

    r_sum = replace.sum(axis=-1)
    ref_sum = reflectance.sum(axis=-1)
    tot = np.count_nonzero(r_sum)    
    ri = r_sum.sum()/(3.0*tot)
    refi = ref_sum.sum()/(3.0*tot)
    refi = refi * poster_brightness
    
    if len(mask.shape) == 2:
        mask = mask[:,:,np.newaxis]
    dst = mask * rgb + (1.0 - mask) * replace * illumination * refi / ri     
    sdst = dst.astype(np.uint8)

    return sdst


'''
edit single image
'''
def image_editing(data_root: str,
                  marker_path: str, 
                  scene_path:str, 
                  src_path:str,                  
                  niid_net,
                  estimator,
                  poster_brightness=1/2.5,
                  save_decomposed=False
):
    # =========== Decompose Scene Image ===========
    print('===> Decompose Scene Image')
    resized_scene_name = os.path.split(scene_path)[-1]
    if save_decomposed:
        decompose_dir = os.path.join(data_root, 'decompose')
        os.mkdir(decompose_dir, exist_ok=True)
        print(f"save decompose scene images to {decompose_dir}")
    else:        
        decompose_dir = None 

    decompose_result = decompose_image( data_root = data_root, 
                                img_name  = resized_scene_name, 
                                model = niid_net,
                                save = save_decomposed, 
                                decompose_dir = decompose_dir,    
                                **{ 'pretrained_file': './third_party/NIID/pretrained_model/final.pth.tar',
                                    'offline': True,
                                    'gpu_devices': [0],
                                }
                               )
    
    # =========== Load Image ===========
    print('===> Load Image')    
    starget = cv2.imread(marker_path, cv2.IMREAD_UNCHANGED) #bgr        
    src = cv2.imread(src_path, cv2.IMREAD_UNCHANGED) #bgr
    if not src.shape[2] == 3:        
        raise ValueError("replace image should have 3 channels")    

    dst = np.float64(decompose_result['rgb'])    
    dst = dst[:,:,::-1] #rgb to bgr  
    sdst = (dst*255.0).astype(np.uint8)        
  
    # =========== Estimate & Warp =============
    print('===> Estimate Flow & Warp Image')   
    ori_H, ori_W = dst.shape[:2] 
    flow = estimator.estimate(sdst, starget) 
    src = cv2.GaussianBlur(src,(3,3),1,borderType=cv2.BORDER_CONSTANT)
    out = image_flow_warp(src, flow[0].permute([1,2,0]), padding_mode='border')              
    mask_origin = (np.ones(shape=(src.shape[0], src.shape[1], 1)) * 255).astype(np.uint8)
    mask_origin = image_flow_warp(mask_origin, flow[0].permute([1,2,0]),padding_mode='zeros')    
    mask = (255 - mask_origin).astype(np.float64) / 255.0
    mask = cv2.GaussianBlur(mask,(3,3),1, borderType=cv2.BORDER_REPLICATE)
    mask = mask[:,:,np.newaxis]
    
    result = (out*(1-mask) + sdst*mask).astype(np.uint8)
    replace_result = cv2.resize(result, (ori_W, ori_H)) 
    mask = cv2.resize(mask, (ori_W, ori_H))        

    # =========== Render Light ============
    print('===> Render')        
    sdst = render(decompose_result, replace_result, poster_brightness=poster_brightness, data_root=data_root, mask=mask)
    return sdst 
