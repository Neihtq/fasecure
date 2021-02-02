import os
import sys
import glob
import argparse
import numpy as np

from PIL import Image
from torchvision import transforms

from face_recognition.utils.constants import LFW_DIR

parser = argparse.ArgumentParser(description='Script for agumenting LFW dataset')

parser.add_argument('--data-dir', default=LFW_DIR, type=str, help='Path to LFW dataset (default: ./data/images/lfw/')

args = parser.parse_args()


def augment_and_normalize(img_input, normalize=True):
    '''Loads images and prepares for data augmentation
    input:
        - path: path to image to be augmented
    '''
    transformers = [transforms.Resize((224, 224))]

    if normalize:
        transformers += [transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        img = Image.fromarray(np.uint8(img_input))
    else:
        img = Image.open(img_input)

    transform = transforms.Compose(transformers)
    augmentations = [
        transforms.ColorJitter(brightness=0.8, contrast=0, saturation=0, hue=0),
        transforms.ColorJitter(brightness=0, contrast=0.8, saturation=0, hue=0),
        transforms.ColorJitter(brightness=0, contrast=0, saturation=0.8, hue=0),
        transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0.1),
        transforms.RandomHorizontalFlip(p=1),
        transforms.RandomPerspective(distortion_scale=0.1, p=1),
    ]
    aug_images = [transform(img)] + [transform(aug(img)) for aug in augmentations]

    return aug_images


def augment_LFW_folder(data_dir):
    for folder in glob.glob(data_dir + "/*"):
        for img in glob.glob(folder+"/*"):
            aug_images = augment_and_normalize(img, normalize=False)

            _, fname = os.path.split(img)
            fname = fname.split('.')[0]
            trans = transforms.ToPILImage()
            for i, aug_img in enumerate(aug_images):
                aug_img_pil = trans(aug_img.squeeze(0))
                dest = os.path.join(folder, f'{fname}_augmented_{i}.jpg')
                aug_img_pil.save(dest)


if __name__ == '__main__':
    data_dir = args.data_dir
    augment_LFW_folder(data_dir)
    sys.exit(0)