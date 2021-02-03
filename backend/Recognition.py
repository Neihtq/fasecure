import torch
import torch.nn.functional as F
import numpy as np

from torchvision import transforms

from face_recognition.models.FaceNet import get_model, load_weights
from face_recognition.database.RegistrationDatabase import RegistrationDatabase
from face_recognition.utils.data_augmentation import augment_and_normalize

class Recognition:
    def __init__(self):
        self.model = load_weights()
        self.model.eval()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        self.model.eval()
        self.normalize = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),  
            transforms.ToTensor(), 
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.db = RegistrationDatabase(fixed_initial_threshold=0)

    def embed(self, img):
        '''
        input:
            - img: numpy array
        '''
        tensor = torch.from_numpy(img).to(self.device).float().permute(2, 1, 0).unsqueeze(0)
        tensor = F.interpolate(tensor, size=224)
        tensor = self.normalize(tensor)
        embedding = self.model(tensor)

        return embedding

    def verify(self, img):
        img_tensor = torch.from_numpy(img).permute(2, 1, 0).float()
        img_tensor = self.normalize(img_tensor).unsqueeze(0)
        embedding = self.model(img_tensor.to(self.device))
        label, access = self.db.face_recognition(embedding)

        return label, bool(access)

    def wipe_db(self):
        status = self.db.clean_database()

        return status

    def register(self, name, img):
        '''
        input:
            - img: numpy array
            - name: string
        '''
        img_tensor = torch.from_numpy(img).permute(2, 1, 0).unsqueeze(0).float()
        aug_images = augment_and_normalize(img_tensor)
        for aug_img in aug_images:
            embedding = self.model(aug_img.to(self.device))
            self.db.face_registration(name, embedding)

        return 0

    def deregister(self, name):
        status = self.db.face_deregistration(name)

        return status

    def list_labels(self):
        label_list = self.db.database["label"].unique().tolist()

        return label_list
