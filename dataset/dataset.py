import os
import torch
import torchvision
import torch.nn
import glob
import random
import numpy as np

from os import listdir, makedirs, listdir, mkdir
from os.path import exists, join

from PIL import Image
from deepface import DeepFace

from torch.utils.data import Dataset, Subset, random_split
from torchvision import transforms


def face_alignment(imgname):
    detected_face = DeepFace.detectFace(imgname)
    return detected_face
    
    
class LFWDataset(Dataset):
    def __init__(self, image_filenames, target_folder):
        self.image_filenames = image_filenames
        self.target_folder = target_folder
        
    def __len__(self):
        return len(self.image_filenames)
    
    def __getitem__(self, idx):
        label, cropped_face = self.load_anchor(idx)
        positive_cropped_face, positive_label = self.load_positive(label, idx)
        negative_cropped_face, negative_label = self.load_negative(label)
        
        return cropped_face, label, positive_cropped_face, positive_label, negative_cropped_face, negative_label
        
    def load_anchor(self, idx):
        image = Image.open(self.image_filenames[idx])      
        label_split = self.image_filenames[idx].split("\\")
        image_label = label_split[0].split("/")[-1]    
        
        cropped_face_numpy = face_alignment(self.image_filenames[idx])
        np.transpose(cropped_face_numpy, (1, 2, 0))
        cropped_face_tensor = torch.from_numpy(cropped_face_numpy.copy())
        
        return image_label, cropped_face_tensor       
        
    def load_positive(self, image_label, idx):   
        #target_folder = './data/lfw-deepfunneled/lfw-deepfunneled/lfw-deepfunneled/'
        target_folder = self.target_folder
        list_person_faces = listdir(join(target_folder, image_label))        
        
        positive_image_filename = join(target_folder, image_label, random.choice(list_person_faces))
        while positive_image_filename == self.image_filenames[idx]:
            positive_image_filename = join(target_folder, image_label, random.choice(list_person_faces))

        positive_image = Image.open(positive_image_filename)      
        positive_label_split = positive_image_filename.split("\\")
        positive_image_label = label_split[0].split("/")[-1]    
        #(print(positive_image_label))
        
        positive_cropped_face_numpy = face_alignment(positive_image_filename)
        np.transpose(positive_cropped_face_numpy, (1, 2, 0))
        positive_cropped_face_tensor = torch.from_numpy(positive_cropped_face_numpy.copy())
        
        return positive_cropped_face_tensor, positive_image_label
        
    def load_negative(self, image_label)
        target_folder = self.target_folder
        list_person_faces = listdir(join(target_folder)
        negative_image_label = random.choice(list_person_faces))
        while negative_image_label == image_label:
            negative_image_label = random.choice(listdir(join(target_folder)))
        negative_image_filename = join(target_folder, negative_image_label, random.choice(listdir(join(target_folder, negative_image_label))))
        
        negative_image = Image.open(negative_image_filename)      
        negative_cropped_face_numpy = face_alignment(negative_image_filename)
        np.transpose(negative_cropped_face_numpy, (1, 2, 0))
        negative_cropped_face_tensor = torch.from_numpy(negative_cropped_face_numpy.copy())
        
        return negative_cropped_face_tensor, negative_image_label