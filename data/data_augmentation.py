import os
import torch
import numpy as np
import glob
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from PIL import Image

#path to data folder
root = os.path.join('.','data','lfw_crop_augmented_1')

#augmentation function
def load_and_transform_img(path):
    '''Loads images and prepares for data augmentation'''
    trfrm = transforms.Compose([transforms.Resize(224),  
                           transforms.ToTensor()]) 
                           #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    #prepare preprocess pipeline
    augmentation_1 = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ColorJitter(brightness=0.8, contrast=0, saturation=0, hue=0),
        transforms.ToTensor()])   

    augmentation_2 = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ColorJitter(brightness=0, contrast=0.8, saturation=0, hue=0),
        transforms.ToTensor()]) 
    
    augmentation_3 = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ColorJitter(brightness=0, contrast=0, saturation=0.8, hue=0),
        transforms.ToTensor()]) 
    
    augmentation_4 = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0.1),
        transforms.ToTensor()]) 
    
    augmentation_5 = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(p=1),
        transforms.ToTensor()])   

    augmentation_6 = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomPerspective(distortion_scale=0.1, p=1),
        transforms.ToTensor()])  


    # read the image and transform it into tensor then normalize it with our trfrm function pipeline
    reg_img = trfrm(Image.open(path)).unsqueeze(0)

    reg_img_1 = reg_img
    # with newer torchvision version, one can also transform tensor batches (but cannot update torchvision)
    # Thus, I have to convert it to an PIL image first
    reg_img_2 = augmentation_1(reg_img.squeeze(0)).unsqueeze(0)
    reg_img_3 = augmentation_2(reg_img.squeeze(0)).unsqueeze(0)
    reg_img_4 = augmentation_3(reg_img.squeeze(0)).unsqueeze(0)
    reg_img_5 = augmentation_4(reg_img.squeeze(0)).unsqueeze(0)
    reg_img_6 = augmentation_5(reg_img.squeeze(0)).unsqueeze(0)
    reg_img_7 = augmentation_6(reg_img.squeeze(0)).unsqueeze(0)
   

    return reg_img_1, reg_img_2, reg_img_3, reg_img_4, reg_img_5, reg_img_6, reg_img_7


#iteration through our data
for folder in glob.glob(root+"/*"):
    number = 0
    for pics in glob.glob(folder+"/*"):
        reg_img_1, reg_img_2, reg_img_3, reg_img_4, reg_img_5, reg_img_6, reg_img_7 = load_and_transform_img(pics)
        
        number += 1
        split = folder.split("\\")
        name = split[-1]
        trans = transforms.ToPILImage()
        
        reg_img_1_pil = trans(reg_img_1.squeeze(0))
        reg_img_2_pil = trans(reg_img_2.squeeze(0))
        reg_img_3_pil = trans(reg_img_3.squeeze(0))
        reg_img_4_pil = trans(reg_img_4.squeeze(0))
        reg_img_5_pil = trans(reg_img_5.squeeze(0))
        reg_img_6_pil = trans(reg_img_6.squeeze(0))
        reg_img_7_pil = trans(reg_img_7.squeeze(0))
        
        reg_img_1_pil.save(folder + "\{name}_000{number}_augmented_1.jpg".format(name=name, number=number))
        reg_img_2_pil.save(folder + "\{name}_000{number}_augmented_2.jpg".format(name=name, number=number))
        reg_img_3_pil.save(folder + "\{name}_000{number}_augmented_3.jpg".format(name=name, number=number))
        reg_img_4_pil.save(folder + "\{name}_000{number}_augmented_4.jpg".format(name=name, number=number))
        reg_img_5_pil.save(folder + "\{name}_000{number}_augmented_5.jpg".format(name=name, number=number))
        reg_img_6_pil.save(folder + "\{name}_000{number}_augmented_6.jpg".format(name=name, number=number))
        reg_img_7_pil.save(folder + "\{name}_000{number}_augmented_7.jpg".format(name=name, number=number))