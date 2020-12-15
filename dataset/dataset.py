import os
import random
import requests
import tarfile 

from os import listdir
from PIL import Image
from deepface import DeepFace

from torch.utils.data import Dataset
from torchvision import transforms


def face_alignment(imgname):
    detected_face = DeepFace.detectFace(imgname)
    return detected_face
    

def download_data(url):
    req = requests.get(url, allow_redirects=True)
    open("data.tgz", 'wb').write(req.content)

    if not os.path.exists('./data'):
        os.makedirs('./data/')

    with tarfile.open('data.tgz', 'r') as f:
        f.extractall('data')
        f.close()
    
    
class LFWDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.labels = []
        for label in listdir(root):
            img_path = os.path.join(root, label)
            if len(listdir(img_path)) > 1:
                self.labels.append(label)
        self.transform = transform
        
    def __len__(self):
        # returns amount of classes
        return len(self.labels)
    
    def __getitem__(self, idx):
        label = self.labels[idx]
        folder = os.path.join(self.root, label)
        img_path = os.path.join(folder, listdir(folder)[0])
        anchor = self.get_image(img_path)
        img_path = os.path.join(folder, listdir(folder)[1])
        positive = self.get_image(img_path)
        negative = self.get_negative(idx)
        
        return label, anchor, positive, negative
        
    def get_image(self, img_path):
        img = Image.open(img_path)
        img_tensor = transforms.ToTensor()(img)
        
        if self.transform:
            img_tensor = self.transform(img_tensor)
        
        return img_tensor
    
    def get_negative(self, idx):
        include = [n for n in range(0, len(self.labels)) if n != idx]
        i = random.choice(include)
        label = self.labels[i]
        folder = os.path.join(self.root, label)
        img_path = os.path.join(folder, listdir(folder)[0])
    
        return self.get_image(img_path)
    
    
if __name__ == '__main__':
    url = "http://vis-www.cs.umass.edu/lfw/lfw-deepfunneled.tgz"
    download_data(url)