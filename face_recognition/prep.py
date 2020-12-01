import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

import matplotlib.pyplot as plt
# %matplotlib inline

def load_and_transform_img(path1, path2):
    #prepare preprocess pipeline
    preprocess_pipelines = [transforms.Resize(224),  
                           transforms.ToTensor(), 
                           transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                std=[0.229, 0.224, 0.225])]

    trfrm = transforms.Compose(preprocess_pipelines)

    # read the image and transform it into tensor then normalize it with our trfrm function pipeline
    img1 = trfrm(Image.open(path1)).unsqueeze(0)
    img2 = trfrm(Image.open(path2)).unsqueeze(0)
   
    return img1, img2

def show_tensor_img(img):
    plt.imshow(img[0].permute(1, 2, 0))