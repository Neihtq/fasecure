import argparse
import os
import torch
import numpy as np
import glob

from tqdm.contrib.concurrent import thread_map
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from PIL import Image

parser = argparse.ArgumentParser(description='Augment dataset.')
parser.add_argument('--path', type=str, help='Path to dataset to be augmented.')
args = parser.parse_args()

def augment(path, normalize=True):
    '''Perform seven augmentation techniques on given image
    input: path to single image
    output: list of seven augmented images
    '''
    composition = [transforms.Resize(224), transforms.ToTensor()]
    if normalize:
        composition.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                std=[0.229, 0.224, 0.225]))

    transform = transforms.Compose(composition) 
    
    augmentations = [
        transforms.Compose([transforms.ToPILImage(),
                        transforms.ColorJitter(brightness=0.8, contrast=0, saturation=0, hue=0),
                        transforms.ToTensor()]),
        transforms.Compose([
            transforms.ToPILImage(),
            transforms.ColorJitter(brightness=0, contrast=0.8, saturation=0, hue=0),
            transforms.ToTensor()]),
        transforms.Compose([
            transforms.ToPILImage(),
            transforms.ColorJitter(brightness=0, contrast=0, saturation=0.8, hue=0),
            transforms.ToTensor()]),
        transforms.Compose([
            transforms.ToPILImage(),
            transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0.1),
            transforms.ToTensor()]), 
        transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(p=1),
            transforms.ToTensor()]),   
        transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomPerspective(distortion_scale=0.1, p=1),
            transforms.ToTensor()])  
    ]
    reg_img = transform(Image.open(path)).unsqueeze(0)    
    out = [reg_img] + [augment(reg_img.squeeze(0)).unsqueeze(0) for augment in augmentations]

    return out


def augment_data(root):
    ''' Augments whole dataset of given path
    Images should be stored in following structure:
    path_to_data/
    |--label/
    |----x.jpg
    |----y.jpg
    ...

    root: path to data 
    '''
    def augment_and_save(img_path):
        trans = transforms.ToPILImage()
        augmented_imgs = augment(img_path, normalize=False)
    
        folder, fname = os.path.split(img_path)
        img_name = fname.split('.')[0]
        
        for i, aug_img in enumerate(augmented_imgs):
            reg_img = trans(aug_img.squeeze(0))
            reg_img.save(os.path.join(folder, f"{img_name}_augmented_{i}.jpg"))

    paths = [img_path for folder in glob.glob(root+"/*") for img_path in glob.glob(folder+"/*")]
    results = thread_map(augment_and_save, paths, max_workers=os.cpu_count())


if __name__ == '__main__':
    root = args.path
    if root:
        augment_data(root)
    else:
        print("No path specified.")
