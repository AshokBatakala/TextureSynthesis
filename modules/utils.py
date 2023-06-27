#imports
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch



def remove_background(image_path,seg_mask_path):
    '''
    returns image without background. 
    A numpy array of shape (H,W,3). normalized between 0 and 1

    Note: 
    The background should be 0 in the segmentation mask.
    shape of segmentation mask should be (H,W)
    shap of image should be (H,W,C)


    '''
    image = plt.imread(image_path)
    seg_mask = plt.imread(seg_mask_path)
    binary_mask = np.where(seg_mask > 0, 1, 0)
    image_without_background = image * binary_mask[...,None]
    return image_without_background / 255.0
    

def squared_image(img=None,image_path = None): 
    '''
    pastes the image in the center of a black square of size 128,128

    args:
    img : numpy array of shape (H,W,C) normalized between 0 and 1
    image_path : path to image

    returns:
    numpy array : (128,128,3) normalized between 0 and 1
    '''
    assert img is None or image_path is None, "only one of img or image_path should be provided"
    
    if image_path is not None:
        img = plt.imread(image_path) / 255.0

    if img.dtype == np.uint8:
        img = img / 255.0 # normalize the image if it is not normalized
        
    H,W,C = img.shape
    h1 = (128 - H)//2
    h2 = h1 + H
    w1 = (128 - W)//2
    w2 = w1 + W

    new_img = np.zeros((128,128,3)) # white background
    new_img[h1:h2,w1:w2,:] = img
    return new_img # normalize the image



def get_texture_map(texture_flow,image,device = torch.device('cpu')):

    """
    texture_flow: [1, H, W, 2]
    image: [1, H_in, W_in, 3] :  ndarray or tensor : range [0, 225]

    return: tensor : [1, H, W, 3]
    """
    # change image to (N,C,H in,W in) format
    image = torch.tensor(image).permute(0,3,1,2)

    image = image.float().to(device)
    texture_flow = texture_flow.float().to(device)
    # print(f"{image.requires_grad = }, {texture_flow.requires_grad = }")

    texture_map = torch.nn.functional.grid_sample(image,texture_flow)
    texture_map = texture_map.permute(0,2,3,1)
    return texture_map