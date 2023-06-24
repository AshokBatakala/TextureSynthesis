#imports
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch



# images without background
def remove_background(image_path,seg_mask_path):
    '''
    returns image without background. 
    A numpy array of shape (H,W,3)

    Note: 
    The background should be 0 in the segmentation mask.
    shape of segmentation mask should be (H,W)
    shap of image should be (H,W,C)


    '''
    image = plt.imread(image_path)
    seg_mask = plt.imread(seg_mask_path)
    binary_mask = np.where(seg_mask > 0, 1, 0)
    image_without_background = image * binary_mask[...,None]
    return image_without_background
    

def squared_image(image_path ): 
    '''
    returned image tensor of shape (128,128,3)
    pastes the image in the center of a white square of size 128,128
    '''
    img = Image.open(image_path)
    new_img = Image.new('RGB', (128, 128), (255, 255, 255))
    new_img.paste(img, ((128 - img.size[0]) // 2, (128 - img.size[1]) // 2))
    image = np.array(new_img)/ 255.0
    return torch.tensor(image)
