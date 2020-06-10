import torch
from kornia import denormalize
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import math

# ad hoc, ineffecient method of getting two factors closest to sqr root of n
def getClosestFactors(n):
    
    val = int(math.sqrt(n))
    
    while (n % val != 0):
        val -= 1
        
    return (n/val, val)
    

# code for visualizing dataset
def visualize(img, label, task, normalize, n=6):
    
    denormalized_img = denormalize(img[:n], torch.tensor(normalize.mean), torch.tensor(normalize.std))
    
    fig, ax = plt.subplots()
    fig.suptitle("Task " + str(task[0]))
    
    rows, cols = getClosestFactors(n)
    
    # check number of channels
    cmap = 'gray' if denormalized_img.shape[1] == 1 else None
    
    for i in range(n):
        
        im = TF.to_pil_image(denormalized_img[i])
        plt.subplot(rows, cols, i+1)
        plt.tight_layout()
        plt.imshow(im, cmap = cmap)
        plt.title("Class: {}".format(label[i]))
        plt.xticks([])
        plt.yticks([])
    
    
    
    



