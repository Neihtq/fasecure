import os
import torch 
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np

from os.path import dirname, abspath
from torchvision import transforms
from torch.multiprocessing import Pool, set_start_method
from PIL import Image

from models.FaceNet import FaceNet

ABSOLUTE_DIR = dirname(abspath(__file__))
MODEL_DIR = os.path.join(ABSOLUTE_DIR, '..', 'models', 'FaceNetOnLFW.pth')


def pairwise_distance(embeddings):
    dot_product = torch.dot(embeddings, torch.transpose(embeddings))
    square_norm = torch.diagonal(dot_product)
    
    # Compute the pairwise distance matrix as we have:
    # ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
    # shape (batch_size, batch_size)
    distances = square_norm.unsqueeze(0) - 2.0 * dot_product + square_norm.unsqueeze(1)
    # cross out negative distances
    distances = torch.maximum(distances, 0,0)

    mask = torch.eq(distances, 0.0).double()
    
    distances = distances + mask * 1e-16
    distances = torch.sqrt(distances)
    distances = distances * (1.0 - mask)

    return distances

def get_anchor_positive_triplet_mask(labels):
    indices_equal = torch.eye(labels.shape[0]).type(torch.BoolTensor)
    indices_not_equal = torch.logical_not(indices_equal)

    labels_equal = torch.eq(labels.unsqueeze(1))

    mask = torch.logical_and(indices_not_equal, labels_equal)

    return mask

def get_anchor_negative_triplet_mask(labels):
    labels_equal = torch.eq(labels.unsqueeze(0), labels.unsqueeze(1))
    mask = torch.logical_not(labels_equal)

    return mask

def batch_hard_triplet_loss(labels, embeddings, margin):
    pairwise_dist = pairwise_dist(embeddings)

    mask_anchor_positive = get_anchor_positive_triplet_mask(labels).double()
    anchor_positive_dist = mask_anchor_positive * pairwise_dist
    hardest_positive_dist = torch.amax(anchor_positive_dist, 1, keepdim=True)

    mask_anchor_negative = get_anchor_negative_triplet_mask(labels).double()
    max_anchor_negative_dist = torch.amax(pairwise_dist, 1, keepdim=True)
    anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * (1.0 - mask_anchor_negative)




def collate_fn(batch):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    root = os.path.join(ABSOLUTE_DIR, '..', 'data', 'lfw_crop')
    model = FaceNet()
    if os.path.exists(MODEL_DIR):
            try:
                model.load_state_dict(torch.load(MODEL_DIR))
            except:
                print("Could not load pretrained model. Continue without weights")
        
    model.to(device)
    model.eval()
    embeddings = {}
    for label, path in batch:
        for pic in os.listdir(path):
            img_path = os.path.join(path, pic)
            img = get_image(img_path).to(device).unsqueeze(0)
            
            embedding = model(img)
            if label not in embeddings:
                embeddings[label] = []
            embeddings[label] = embedding

    batch = select_triplets(embeddings)
    labels = (x[0] for x in batch)
    anchors = torch.vstack([x[1] for x in batch])
    positive = torch.vstack([x[2] for x in batch])
    negative = torch.vstack([x[3] for x in batch])

    return [labels, anchors, positive, negative]


def select_triplets(embeddings):
    try:
        set_start_method('spawn')
    except RuntimeError:
        print("Error with set_start_method(spawn")

    pool = Pool(os.cpu_count())
    results = pool.map(get_triplets, list(embeddings.keys()))

    batch = []
    for i, label in enumerate(list(embeddings.keys())):
        batch.append((label,) + results[i]) 

    return batch

    
def get_triplets(label, embeddings) :
    anchor, pos, anchor_embedding = get_positive_and_anchor(label, embeddings)
    neg = get_negative(label, anchor_embedding, embeddings)
    return anchor, pos, neg


def get_positive_and_anchor(self, label, embeddings):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    diff = 0
    anchor_embedding = torch.zeros(128, device=torch.device(device))
    curr_pos = torch.zeros(128, device=torch.device(device))
    for l in embeddings:
        anchor = embeddings[l]
        pos_embedding, dist = finder(label, anchor, embeddings)
        if dist > diff:
            diff = dist
            curr_pos = pos_embedding
            anchor_embedding = anchor
    
    return curr_pos, anchor_embedding


def get_negative(self, label, anchor, embeddings):
    diff = float('inf')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    curr_embedding = torch.zeros(128, device=torch.device(device))
    for l in embeddings:
        if l != label:
            embedding, dist = self.finder(l, anchor, find_max=False)
            if dist < diff:
                curr_embedding = embedding
                diff = dist

    return curr_embedding


def finder(self, label, anchor, embeddings, find_max=True):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    diff = 0 if find_max else float('inf')
    curr_embedding = torch.zeros(128, device=torch.device(device))
    for embedding in embeddings[label]:
        dist = np.linalg.norm(anchor - embedding)
        update = dist > diff if find_max else dist < diff
        if update:
            diff = dist
            curr_embedding = embedding

    return curr_embedding, diff
        

def get_image(img_path):
        '''Returns Pytorch.Tensor of image'''
        img = Image.open(img_path)      
        transform = transforms.Compose([
            transforms.Resize((250, 250)),
            transforms.ToTensor(), 
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )
        img = transform(img)
        
        return img