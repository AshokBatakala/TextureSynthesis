# ashok added code 
import cv2

def read_image(image_path):
    """
    Read image from image_path in RGB format
    image: [H, W, 3]
    """
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image



import matplotlib.pyplot as plt
import torch

def lazy_plot(image,gray=False,figsize=(2.5,2.5),centered=False):

    """
    expected usage:
    image is np_array or a tensor

    Plot image
    image: [H, W, 3]
    if tensor: image: [3, H, W]
    """
    # if centered i.e. b/w [-1,+1] make it [0,1]
    if centered:
      image = (image+1)/2
    # 1.check if tensor
    if isinstance(image,torch.Tensor) :
        image = image.detach().cpu().numpy()
    # 2. batch
    if len(image.shape) == 4:
        N = image.shape[0]
        if N != 1:
            print("Plotting only the first image in the batch")
        image = image[0]
    # 3. channel
    first_channel = image.shape[0]
    if first_channel in [3, 1, 4]:
        image = image.transpose(1,2,0)
    # 4. check the range of the image
    if image.max() > 1 + 1e-3:
        image = image / 255.0
    # 5. plot
    plt.figure(figsize=figsize)
    if gray:
        plt.imshow(image,cmap='gray')
    else:
        plt.imshow(image)
    plt.axis('off')
