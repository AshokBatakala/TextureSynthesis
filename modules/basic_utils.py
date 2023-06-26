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
def plot_image(image,gray=False,tensor=False,figsize=(3,3)):
    """
    Plot image
    image: [H, W, 3]
    if tensor: image: [3, H, W]
    """
    if tensor:
        image = image.permute(1,2,0).numpy()
    plt.figure(figsize=figsize)
    if gray:
        plt.imshow(image,cmap='gray')
    else:
        plt.imshow(image)
    plt.show()
