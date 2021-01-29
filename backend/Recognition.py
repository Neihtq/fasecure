import torch

from PIL import Image
from torchvision import transforms

from face_recognition.models.FaceNet import get_model
from face_recognition.database.RegistrationDatabase import RegistrationDatabase
from face_recognition.utils.data_augmentation import augment_and_normalize


class Recognition:
    def __init__(self):
        self.model = get_model()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.db = RegistrationDatabase(fixed_initial_threshold=98.5)

    def embed(self, img):
        '''
        input:
            - img: numpy array
        '''
        tensor = torch.from_numpy(img)
        tensor = self.transform(tensor)
        embedding = self.model(tensor.unsqueeze(0))

        return embedding

    def get_img_tensor(self, img):
        img = Image.open(img)
        tensor = self.transformers(img).to(self.device)

        return tensor

    def verify(self, img):
        embedding = self.embed(img)
        label, access = self.db.face_recognition(embedding)

        return label, access

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
            embedding = self.model(aug_img)
            self.db.face_registration(name, embedding)

        return 0

    def deregister(self, name):
        status = self.db.face_deregistration(name)

        return status
