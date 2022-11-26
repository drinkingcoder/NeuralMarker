from xml.etree.ElementInclude import default_loader
import matplotlib.pyplot as plt
import cv2
import os

def show_input_images(scene_images, target_images):
    plt.figure(figsize=(15,15))
    num = len(scene_images)
    plt.subplot(221), plt.imshow(scene_images[0]), plt.axis('off')
    plt.subplot(222), plt.imshow(target_images[0]), plt.axis('off')
    plt.subplot(223), plt.imshow(scene_images[num-1]), plt.axis('off')
    plt.subplot(224), plt.imshow(target_images[num-1]), plt.axis('off')
    plt.show()

def resize_viewpoint(img):
    if img.shape[0] > img.shape[1]:
        resize = cv2.flip(cv2.transpose(img), 0)
        return resize
    else:
        return img

def resize_lighting(img):
    if img.shape[0] > img.shape[1]:        
        region = img[140:500, 0:480]
        resize = cv2.resize(region, (640, 480))
        return resize 
    else:
        return img

