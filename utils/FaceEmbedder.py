import os
import csv
import torch
import numpy as np
import pandas as pd

from os import listdir
from os.path import dirname, abspath
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from models.FaceNet import FaceNet


ABSOLUTE_DIR = dirname(abspath(__file__))
MODEL_DIR = os.path.join(ABSOLUTE_DIR, '..', 'models', 'FaceNetOnLFW.pth')

class FaceEmbedder():
    def __init__(self, root, transform=None):
        self.root = root
        self.labels = []
        for label in listdir(root):
            img_path = os.path.join(root, label)
            if len(listdir(img_path)) > 1:
                self.labels.append(label)
        
        self.transform = transform
        self.labels = self.labels[0:50]
        self.model = FaceNet(num_classes=len(self.labels))
        
       # if os.path.exists(MODEL_DIR):
       #     self.model.load_state_dict(torch.load(MODEL_DIR))

    def calculate_embedding(self):
        '''Creates CSV of all embeddings from all persons with latest model'''
        device = "cuda" if torch.cuda.is_available() else "cpu"
        embeddings = {} # key: label, value: [ (index, path_to_img ) ]
        np_arrays = []
        index = 0
        self.model = self.model.to(device)

        for label in self.labels:
            folder = os.path.join(self.root, label)
            for i in listdir(folder):
                img_path = os.path.join(folder, i)
                img = self.get_image(img_path)
                img = img.reshape((1,) + tuple(img.shape)).to(device)

                embedding = self.model(img).detach().to("cpu").numpy()

                np_arrays.append(embedding)

                if not label in embeddings:
                    embeddings[label] = []
                embeddings[label].append((index, img_path))
                index += 1

        df = pd.DataFrame(dict([ (k, pd.Series(v)) for k, v in embeddings.items()]))
        df.to_csv('embeddings.csv', index=False)

        np_arrays = np.vstack(np_arrays)
        with open ('embeddings.npy', 'wb') as f:
            np.save(f, np_arrays)
    
    def get_image(self, img_path):
        '''Returns Pytorch.Tensor of image'''
        img = Image.open(img_path)      

        if self.transform:
            img = self.transform(img)
       
        if not torch.is_tensor(img):        
            img = transforms.ToTensor()(img)
        
        return img