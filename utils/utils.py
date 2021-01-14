import os
import cv2 
import torch 
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np

from os.path import dirname, abspath
from torch.autograd import Variable
from PIL import Image
from deepface import DeepFace
from torchvision import transforms
from torch.multiprocessing import Pool, set_start_method

from models.FaceNet import FaceNet

ABSOLUTE_DIR = dirname(abspath(__file__))
MODEL_DIR = os.path.join(ABSOLUTE_DIR, '..', 'models', 'FaceNetOnLFW.pth')

def face_alignment(imgname):
    '''Rotates the image as much necessary that you could draw a straight line between the two eyes
    input: path to image
    return: 224 x 224 x 3 image
    '''
    detected_face = DeepFace.detectFace(imgname)
    return detected_face


def predict_transform(prediction, in_dim, anchors, num_classes, device="cpu"):
    batch_size = prediction.size(0)
    stride =  in_dim // prediction.size(2)
    grid_size = in_dim // stride
    bbox_attrs = 5 + num_classes
    num_anchors = len(anchors)
    
    prediction = prediction.view(batch_size, bbox_attrs*num_anchors, grid_size*grid_size)
    prediction = prediction.transpose(1,2).contiguous()
    prediction = prediction.view(batch_size, grid_size*grid_size*num_anchors, bbox_attrs)
    anchors = [(a[0]/stride, a[1]/stride) for a in anchors]

    #Sigmoid the  centre_X, centre_Y. and object confidencce
    prediction[:,:,0] = torch.sigmoid(prediction[:,:,0])
    prediction[:,:,1] = torch.sigmoid(prediction[:,:,1])
    prediction[:,:,4] = torch.sigmoid(prediction[:,:,4])
    
    #Add the center offsets
    grid = np.arange(grid_size)
    a,b = np.meshgrid(grid, grid)

    x_offset = torch.FloatTensor(a).view(-1,1)
    y_offset = torch.FloatTensor(b).view(-1,1)

    x_offset = x_offset.to(device)
    y_offset = y_offset.to(device)

    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1,num_anchors).view(-1,2).unsqueeze(0)

    prediction[:,:,:2] += x_y_offset

    #log space transform height and the width
    anchors = torch.FloatTensor(anchors)

    anchors = anchors.to(device)

    anchors = anchors.repeat(grid_size*grid_size, 1).unsqueeze(0)
    prediction[:,:,2:4] = torch.exp(prediction[:,:,2:4])*anchors
    
    prediction[:,:,5: 5 + num_classes] = torch.sigmoid((prediction[:,:, 5 : 5 + num_classes]))

    prediction[:,:,:4] *= stride
    
    return prediction


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
            img = get_image(img_path).to(device)
            
            embedding = model(img)
            if label not in embeddings:
                embeddings[label] = []
            embeddings[label] = embedding

    batch = select_triplets(embedding)
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
