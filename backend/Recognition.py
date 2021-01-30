import torch
import torch.nn.functional as F
import numpy as np

from torchvision import transforms

from face_recognition.models.FaceNet import get_model
from face_recognition.database.RegistrationDatabase import RegistrationDatabase
from face_recognition.utils.data_augmentation import augment_and_normalize


class Recognition:
    def __init__(self):
        self.model = get_model()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        self.normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        self.db = RegistrationDatabase(fixed_initial_threshold=98.5)

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
        embedding = self.embed(img)
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
        aug_images = augment_and_normalize(img)
        for aug_img in aug_images:
            embedding = self.model(aug_img.to(self.device).unsqueeze(0))
            self.db.face_registration(name, embedding)

        return 0

    def deregister(self, name):
        status = self.db.face_deregistration(name)

        return status

    def list_registered(self):
        pass
        # TODO: return list of name for registered faces