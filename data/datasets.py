import os
import random
import torch
import glob
import multiprocessing as mp

from random import shuffle
from os import listdir
from os.path import dirname, abspath
from PIL import Image
from itertools import compress

from torch.utils.data import Dataset
from torchvision import transforms
    

class TupleDataset(Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets
    
    def __getitem__(self, idx):
        return tuple(d[idx] for d in self.datasets)
    
    def __len__(self):
        return min(len(d) for d in self.datasets)
    
    
class ImageDataset(Dataset):
    '''Regular Dataset where images are stored like this:
    path_to_data/
    |--label/
    |----x.jpg
    |----y.jpg

    root: path to images
    transform: functions from torchvisions.transforms to apply on images
    '''
    def __init__(self, root, transform=None):
        self.root = root
        self.label_to_number = {}
        self.data = []
        
        folder_pairs = list(enumerate(listdir(root)))
        for i, label in folder_pairs:
            self.label_to_number[i] = label
        
        with mp.Pool(processes=20) as pool:
            data = pool.map(self.aggregate_data, folder_pairs)
            flattened = [pair for person in data for pair in person]
            self.data = flattened
        
        self.transform = transform
        shuffle(self.data)

    def aggregate_data(self, folder_index_tuple):
        data = []
        i, label = folder_index_tuple
        label_path = os.path.join(self.root, label)
        for img in listdir(label_path):
            img_path = os.path.join(label_path, img)
            if os.path.exists(img_path):
                data.append((i, img_path))
        
        return data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        label, img_path = self.data[idx]
        img = self.get_image(img_path)

        return label, img
        
    def get_image(self, img_path):
        img = Image.open(img_path).convert('RGB')
       
        if self.transform:
            img = self.transform(img)
       
        if not torch.is_tensor(img):        
            img = transforms.ToTensor()(img)
        
        return img.float()
    

class LFWValidationDataset(Dataset):
    def __init__(self, root, pairs_txt, transform=None):
        self.root = root
        self.pairs = []
        self.transform = transform
        self.read_pairs_txt(pairs_txt)
        
    def read_pairs_txt(self, pairs_txt):
        with open(pairs_txt, 'r') as f:
            lines = f.readlines()[1:]
            for line in lines:
                line_split = line.split()
                if len(line_split) == 3:
                    name, img_1, img_2 = line_split
                    img_1, img_2 = int(img_1), int(img_2)
                    fpath_1 = self.add_suffix(os.path.join(self.root, name, name + '_' + f"{img_1:04d}"))
                    fpath_2 = self.add_suffix(os.path.join(self.root, name, name + '_' + f"{img_2:04d}"))
                    same = True
                else:
                    name_1, img_1, name_2, img_2 = line_split
                    img_1, img_2 = int(img_1), int(img_2)
                    fpath_1 = self.add_suffix(os.path.join(self.root, name_1, name_1 + '_' + f"{img_1:04d}"))
                    fpath_2 = self.add_suffix(os.path.join(self.root, name_2, name_2 + '_' + f"{img_2:04d}"))
                    same = False
                
                if os.path.exists(fpath_1) and os.path.exists(fpath_2):
                    self.pairs.append((fpath_1, fpath_2, same))

    def add_suffix(self, path):
        if os.path.exists(path + '.jpg'):
            return path + '.jpg'
        return path + '.png'            

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        fpath_1, fpath_2, same = self.pairs[idx]
        img_1 = self.get_image(fpath_1)
        img_2 = self.get_image(fpath_2)
        
        return img_1, img_2, same
        
    def get_image(self, img_path):
        '''Returns Pytorch.Tensor of image'''
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        if not torch.is_tensor(img):        
            img = transforms.ToTensor()(img)
        
        return img.float()


class LFWDataset(Dataset):
    def __init__(self, root, transform=None, cropped_faces=False):
        self.root = root
        self.cropped_faces = cropped_faces
        self.labels = []
        self.mask = []
        self.transform = transform
        for label in listdir(root):
            img_path = os.path.join(root, label)
            if len(listdir(img_path)) > 5:
                for imgs_per_folder in range(len(listdir(img_path))):
                    self.labels.append(label)
                    self.mask.append(True)
            else:
                for imgs_per_folder in range(len(listdir(img_path))):
                    self.mask.append(False)

        self.image_filenames = glob.glob(os.path.join(root, "**/*.jpg"), recursive=True)
        
        # Use mask to filter classes with less than certain amount of images
        self.image_filenames = list(compress(self.image_filenames, self.mask))

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        label = self.labels[idx]
        img = self.image_filenames[idx]
        
        if self.cropped_faces:
            img = self.get_image(img)

        return label, img
        
    def get_image(self, img_path):
        img = Image.open(img_path)
        img_tensor = transforms.ToTensor()(img)
        
        if self.transform:
            img_tensor = self.transform(img_tensor)
        
        return img_tensor