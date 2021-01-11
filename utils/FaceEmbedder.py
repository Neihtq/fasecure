import os
import csv
import pathlib
import torch
import numpy as np
import pandas as pd


from os import listdir
from os.path import dirname, abspath
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from multiprocessing import Pool

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
        self.update()
        
    def update(self):
        print("Create embedding space.")
        self.calculate_embedding()
        print("Create triplet selections.")
        self.select_triplets()

    def get_embedding_and_path(self, info):
        infos = info.split(',')
        index = int(infos[0][1:])
        img_path = infos[1][2:-2]
        embedding = self.embeddings_np[index]
        
        return embedding, img_path

    def select_triplets(self):
        self.embeddings_info = pd.read_csv('embeddings.csv')
        with open('embeddings.npy', 'rb') as f:
            self.embeddings_np = np.load(f)

        pool = Pool(os.cpu_count())
        results = pool.map(self.get_triplets, self.labels)

        triplets = {}
        for i, label in enumerate(self.labels):
            triplets[label] = results[i]

        with open("triplets.csv", "w", newline='') as f:
            writer = csv.DictWriter(f, triplets.keys())
            writer.writeheader()
            writer.writerow(triplets)
        
    def get_triplets(self, label):
        anchor, pos, anchor_embedding = self.get_positive_and_anchor(label)
        neg = self.get_negative(label, anchor_embedding)
        return anchor, pos, neg
    
    def get_positive_and_anchor(self, label):
        diff = 0
        curr_anchor = ""
        anchor_embedding = np.zeros(128)
        curr_pos = ""
        for info in self.embeddings_info[label]:
            if pd.notnull(info):
                embedding, anchor_path = self.get_embedding_and_path(info)
                pos_path, dist = self.finder(label, embedding)
                if dist > diff:
                    diff = dist
                    curr_anchor = anchor_path
                    curr_pos = pos_path
                    anchor_embedding = embedding
        
        return curr_anchor, curr_pos, anchor_embedding

    def get_negative(self, label, anchor):
        diff = float('inf')
        neg_path = ""
        for l in self.labels:
            if l != label:
                path, dist = self.finder(l, anchor, find_max=False)
                if dist < diff:
                    neg_path = path
                    diff = dist

        return neg_path

    def finder(self, label, anchor, find_max=True):
        diff = 0 if find_max else float('inf')
        curr_path = ""
        for info in self.embeddings_info[label]:
            if pd.notnull(info):
                embedding, img_path = self.get_embedding_and_path(info)
                dist = np.linalg.norm(anchor - embedding)
                update = dist > diff if find_max else dist < diff
                if update:
                    diff = dist
                    curr_path = img_path
        
        return curr_path, diff

    def calculate_embedding(self):
        '''Creates CSV of all embeddings from all persons with latest model'''
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = FaceNet(num_classes=len(self.labels))
        model = model.to(device)
        if os.path.exists(MODEL_DIR):
            try:
                model.load_state_dict(torch.load(MODEL_DIR))
            except:
                print("Could not load pretrained model. Continue without weights")
        
        embeddings = {}
        np_arrays = []
        index = 0
        for label in self.labels:
            folder = os.path.join(self.root, label)
            for i in listdir(folder):
                img_path = os.path.join(folder, i)
                img = self.get_image(img_path)
                img = img.reshape((1,) + tuple(img.shape)).to(device)

                embedding = model(img).detach().to("cpu").numpy()
                np_arrays.append(embedding)

                if not label in embeddings:
                    embeddings[label] = []

                embeddings[label].append((index, img_path))
                index += 1

        df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in embeddings.items()]))
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