import os
import sys
import glob
import argparse

from PIL import Image
from torchvision import transforms

from face_recognition.utils.constants import LFW_DIR

parser = argparse.ArgumentParser(description='Script for agumenting LFW dataset')

# parser.add_argument('--data-dir', default=LFW_DIR, type=str, help='Path to LFW dataset (default: ./data/images/lfw/')

argparse = parser.parse_args()


def augment(tensor_img):
    augmentation_1 = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize([224,224]),  
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])   

    augmentation_2 = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ColorJitter(brightness=0.8, contrast=0, saturation=0, hue=0),
        transforms.ToTensor()])   

    augmentation_3 = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ColorJitter(brightness=0, contrast=0.8, saturation=0, hue=0),
        transforms.ToTensor()]) 
    
    augmentation_4 = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ColorJitter(brightness=0, contrast=0, saturation=0.8, hue=0),
        transforms.ToTensor()]) 
    
    augmentation_5 = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0.1),
        transforms.ToTensor()]) 
    
    augmentation_6 = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(p=1),
        transforms.ToTensor()])   

    augmentation_7 = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomPerspective(distortion_scale=0.1, p=1),
        transforms.ToTensor()])  
    
    
    # with newer torchvision version, one can also transform tensor batches (but cannot update torchvision)
    # Thus, I have to convert it to an PIL image first
    aug_img_1 = augmentation_1(tensor_img.squeeze(0)).unsqueeze(0)
    aug_img_2 = augmentation_2(aug_img_1.squeeze(0)).unsqueeze(0)
    aug_img_3 = augmentation_3(aug_img_1.squeeze(0)).unsqueeze(0)
    aug_img_4 = augmentation_4(aug_img_1.squeeze(0)).unsqueeze(0)
    aug_img_5 = augmentation_5(aug_img_1.squeeze(0)).unsqueeze(0)
    aug_img_6 = augmentation_6(aug_img_1.squeeze(0)).unsqueeze(0)
    aug_img_7 = augmentation_7(aug_img_1.squeeze(0)).unsqueeze(0)
   

    return aug_img_1, aug_img_2, aug_img_3, aug_img_4, aug_img_5, aug_img_6, aug_img_7


def augment_LFW_folder(data_dir):
    for folder in glob.glob(data_dir + "/*"):
        number = 0
        for img in glob.glob(folder+"/*"):
            aug_imgs = augment(img)

            _, fname = os.path.split(img, normalize=False)
            fname = fname.split('.')[0]
            trans = transforms.ToPILImage()
            for i, aug_img in enumerate(aug_imgs):
                aug_img_pil = trans(aug_img.squeeze(0))
                dest = os.path.join(folder, f'{fname}_augmented_{i}.jpg')
                aug_img_pil.save(dest)


if __name__ == '__main__':
    data_dir = args.data_dir
    augment_LFW_folder(data_dir)
    sys.exit(0)