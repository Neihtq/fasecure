import os
import sys
import glob
import argparse

from PIL import Image
from torchvision import transforms

from backend.face_recognition.utils.constants import LFW_DIR

parser = argparse.ArgumentParser(description='Script for agumenting LFW dataset')

parser.add_argument('--data-dir', default=LFW_DIR, type=str, help='Path to LFW dataset (default: ./data/images/lfw/')

argparse = parser.parse_args()


def augment(img_np, normalize=True):
    '''Loads images and prepares for data augmentation
    input:
        - path: path to image to be augmented
    '''
    to_tensor = transforms.ToTensor()
    augmentations = [transforms.Compose([
                        transforms.ColorJitter(brightness=0.8, contrast=0, saturation=0, hue=0),
                        transforms.ToTensor()]),
                    transforms.Compose([
                        transforms.ColorJitter(brightness=0, contrast=0.8, saturation=0, hue=0),
                        transforms.ToTensor()]),
                    transforms.Compose([
                        transforms.ColorJitter(brightness=0, contrast=0, saturation=0.8, hue=0),
                        transforms.ToTensor()]),
                    transforms.Compose([
                        transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0.1),
                        transforms.ToTensor()]),
                    transforms.Compose([
                        transforms.RandomHorizontalFlip(p=1),
                        transforms.ToTensor()]),
                    transforms.Compose([
                        transforms.RandomPerspective(distortion_scale=0.1, p=1),
                        transforms.ToTensor()])]

    img = Image.fromarray(img_np).resize((224, 224))

    aug_imgs = []
    for aug in augmentations:
        aug_img = aug(img)
        aug_imgs.append(aug_img)

    aug_imgs.insert(0, to_tensor(img))

    return aug_imgs


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