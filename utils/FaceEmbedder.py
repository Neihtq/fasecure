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
from torch.multiprocessing import Pool, set_start_method

from models.FaceNet import FaceNet


ABSOLUTE_DIR = dirname(abspath(__file__))
MODEL_DIR = os.path.join(ABSOLUTE_DIR, '..', 'models', 'FaceNetOnLFW.pth')

class FaceEmbedder():
    def __init__(self, root, transform=None, init_update=True):
        self.root = root
        self.labels = []
        for label in listdir(root):
            img_path = os.path.join(root, label)
            if len(listdir(img_path)) > 1:
                self.labels.append(label)
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.transform = transform
        
        if init_update:
            self.update()
        
    def update(self):
        print("Create embedding space.")
        self.calculate_embedding()
        print("Create triplet selections.")
        self.select_triplets()

    def get_embedding_and_path(self, info):
        index = int(info[0])
        img_path = str(info[1])
        embedding = self.embeddings[index]
        
        return embedding, img_path

    def select_triplets(self):
        #self.embeddings_info = pd.read_csv('embeddings.csv', dtype={label: str for label in self.labels})
        #with open('embeddings.npy', 'rb') as f:
        #    self.embeddings_np = np.load(f)
        
        try:
            set_start_method('spawn')
        except RuntimeError:
            print("Error with set_start_method(spawn")
        
        pool = Pool(processes=None)
        results = pool.map(self.get_triplets, self.labels, 250)
        #pool.close()
        #pool.join()
        
        #results = list(results)
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
        anchor_embedding = torch.zeros(128, device=torch.device(self.device))
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
                dist = torch.norm(anchor - embedding)
                update = dist > diff if find_max else dist < diff
                if update:
                    diff = dist
                    curr_path = img_path
        
        return curr_path, diff

    def calculate_embedding(self):
        '''Creates CSV of all embeddings from all persons with latest model'''
        model = FaceNet(num_classes=len(self.labels))
        model = model.to(self.device)
        if os.path.exists(MODEL_DIR):
            try:
                model.load_state_dict(torch.load(MODEL_DIR))
            except:
                print("Could not load pretrained model. Continue without weights")
        
        embeddings = {}
        self.embeddings = []
        index = 0
        for label in self.labels:
            folder = os.path.join(self.root, label)
            for i in listdir(folder):
                img_path = os.path.join(folder, i)
                img = self.get_image(img_path).to(self.device)
                img = img.unsqueeze(0)

                embedding = model(img).detach()
                self.embeddings.append(embedding)

                if not label in embeddings:
                    embeddings[label] = []

                embeddings[label].append((index, img_path))
                index += 1

        self.embeddings_info = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in embeddings.items()]))
        
        #df.to_csv('embeddings.csv', index=False)

        #np_arrays = np.vstack(np_arrays)
        #with open ('embeddings.npy', 'wb') as f:
        #    np.save(f, np_arrays)
    
    def get_image(self, img_path):
        '''Returns Pytorch.Tensor of image'''
        img = Image.open(img_path)      

        if self.transform:
            img = self.transform(img)
       
        if not torch.is_tensor(img):        
            img = transforms.ToTensor()(img)
        
        return img