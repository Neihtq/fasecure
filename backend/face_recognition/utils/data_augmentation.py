import os
import sys
import glob
import numpy as np

from PIL import Image
from torchvision import transforms

from face_recognition.utils.constants import LFW_DIR


def augment_and_normalize(tensor_img, aug_mean=[0.6068, 0.4517, 0.3800], aug_std=[0.2492, 0.2173, 0.2082]):
    '''
    input: tensor 1x3x224x224
    output: tensor 1x3x224x224
    '''
    transformer = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),  
            transforms.ToTensor(), 
            transforms.Normalize(mean=aug_mean,std=aug_std)
        ])

    augmentation_techniques = [
        transforms.Compose([
            transforms.ToPILImage(),
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
    
    aug_img_1 = transformer(tensor_img.squeeze(0)).unsqueeze(0)    
    aug_imgs = [aug_img_1]
    for augmentation in augmentation_techniques:
        aug = augmentation(aug_img_1.squeeze(0)).unsqueeze(0)
        aug_imgs.append(aug)
   

    return aug_imgs
