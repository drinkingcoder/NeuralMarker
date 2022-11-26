import os 
os.environ["OMP_NUM_THREADS"]="1"
os.environ["MKL_NUM_THREADS"]="1"
os.environ["CUDA_VISIBLE_DEVICES"]="4"
import sys

sys.path.append('./core')
from flow_estimator import Flow_estimator
from harsh_lighting_utils import image_editing
from config import get_demo_harsh_lighting_args,get_twins_args

sys.path.append('./third_party/NIID')
sys.path.append('./third_party/NIID/utils')
sys.path.append('./third_party/NIID/models')
sys.path.append('./third_party/NIID/models/Hu_nets')
from third_party.NIID.config import TestOptions
from third_party.NIID.config import TestOptions
from third_party.NIID.models.manager import create_model
from third_party.NIID.utils import pytorch_settings, visualize


import cv2
from PIL import Image, ImageOps
import numpy as np

def resize_rewrite_cv2(img_path):
    frame = cv2.imread(img_path)
    if frame.shape[0] > frame.shape[1]: # H > W
        frame = cv2.resize(frame, (480, 640), cv2.INTER_LANCZOS4)     #size = (W, H)
    else: # H <= W
        frame = cv2.resize(frame, (640, 480), cv2.INTER_LANCZOS4)     #size = (W, H)               
    resize_name = '%s_resize.jpg' % (img_path[:img_path.rfind('.')])
    cv2.imwrite(resize_name, frame)
    return resize_name

def resize_rewrite_PIL(img_path):
    frame = Image.open(img_path)
    frame = ImageOps.exif_transpose(frame)
    if frame.size[1] > frame.size[0]: # H > W Image.size => (W, H)
        frame = frame.resize((480, 640), Image.ANTIALIAS)     #size = (W, H)
    else: # H <= W
        frame = frame.resize((640, 480), Image.ANTIALIAS)     #size = (W, H)               
    resize_name = '%s_resize.jpg' % (img_path[:img_path.rfind('.')])
    quality_val = 90
    frame.save(resize_name, 'JPEG', quality=quality_val)
    return resize_name

def resize_PIL(img_path):
    frame = Image.open(img_path)
    frame = ImageOps.exif_transpose(frame)
    if frame.size[1] > frame.size[0]: # H > W Image.size => (W, H)
        frame = frame.resize((480, 640), Image.ANTIALIAS)     #size = (W, H)
    else: # H <= W
        frame = frame.resize((640, 480), Image.ANTIALIAS)     #size = (W, H)               
    return np.array(frame)

def process_image(args):      
    # =========== Path Config =========
    print('===> Path Config & Image Resize')
    # scene_yml_path = "./data/scene_decomposed.yml"  
    scene_path = os.path.join(args.exp_dir, args.scene_name)    
    marker_path = os.path.join(args.exp_dir, args.marker_name)
    source_path = os.path.join(args.exp_dir, args.source_name)         
    if not (os.path.exists(scene_path) and os.path.exists(marker_path) and os.path.exists(source_path)):
        raise FileNotFoundError() 
    
    scene_path = resize_rewrite_PIL(scene_path)        
    marker_path = resize_rewrite_PIL(marker_path)
    source_path = resize_rewrite_PIL(source_path)
                
    print(f"scene : {scene_path}")
    print(f"marker: {marker_path}")
    print(f"source: {source_path}") 

    # ========== Load Model =========
    print('===> Load NIID Model')    
    kwargs = {'pretrained_file': './pre_trained_model/NIID/final.pth.tar',
              'offline': True,
              'gpu_devices': [0],
             }
    opt = TestOptions()
    opt.parse(kwargs)    
    pytorch_settings.set_(with_random=False, determine=True)    
    niid_net = create_model(opt)
    niid_net.switch_to_eval()
    
    print('===> Load Flow Estimate Model')
    estimator_args = get_twins_args()   
    print(estimator_args) 
    estimator = Flow_estimator(estimator_args)

    # ========== Edit Image =========
    print('===> Edit Image')                
    result = image_editing( data_root=args.exp_dir, 
                            marker_path=marker_path, 
                            scene_path=scene_path, 
                            src_path=source_path, 
                            niid_net=niid_net,
                            estimator=estimator,
                            poster_brightness=args.poster_brightness,
                            save_decomposed=args.save_decomposed)

    prefix=os.path.split(scene_path)[-1][:-4]
    return result, prefix    

if __name__ == '__main__':
    args = get_demo_harsh_lighting_args()    
    result, prefix = process_image(args)

    frame1 = Image.open(os.path.join(args.exp_dir, args.scene_name))
    frame1 = ImageOps.exif_transpose(frame1)
    frame2 = Image.fromarray(result[:,:,::-1])    
    frame2 = frame2.resize(frame1.size, Image.ANTIALIAS)
    frame2.save(os.path.join(args.exp_dir, prefix+'_soutput.jpg'), 'JPEG', quality=90)


    
