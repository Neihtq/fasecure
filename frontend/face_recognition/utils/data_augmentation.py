import os
import sys
import glob
import args

from PIL import Image
from torchvision import transforms

from face_recognition.utils.constants import LFW_DIR

parser = args.ArgumentParser(description='Script for agumenting LFW dataset')

parser.add_argument('--data-dir', default=LFW_DIR, type=str, help='Path to LFW dataset (default: ./data/images/lfw/')

args = parser.parse_args()


def augment(path, normalize=True):
    '''Loads images and prepares for data augmentation
    input:
        - path: path to image to be augmented
    '''
    transformers = [transforms.Resize(224), transforms.ToTensor()])
    if normalize: 
        transformers.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
    transform = transforms.Compose(transformers)
    #prepare preprocess pipeline

    augmentations = [transforms.Compose([
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
                        transforms.ToTensor()])]

    aug_imgs = [transform(Image.open(path)).unsqueeze(0)]
    for aug in augmentations:
        aug_img = aug(aug_imgs[0].squeeze(0)).unsqueeze(0)
        aug_imgs.append(aug_img)

    return aug_imgs


def augment_LFW_folder(data_dir):
    for folder in glob.glob(data_dir +"/*"):
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


if __name__ = '__main__':
    data_dir = args.data_dir
    augment_LFW_folder(data_dir)
    sys.exit(0)