import torch

from PIL import Image
from torchvision import transforms

from face_recognition.models.FaceNet import get_model
from face_recognition.database.RegistrationDatabase import RegistrationDatabase


class Recognition:
    def __init__(self):
        self.model = get_model()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        self.transformers = transforms.Compose([transforms.Resize(224), transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.485, 0.456, 0.406])])
        self.db = RegistrationDatabase(fixed_initial_threshold=98.5)

    def embed(self, img):
        tensor = self.get_img_tensor(img)
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
        embedding = self.embed(img)
        status = self.db.face_registration(name, embedding)

        return status

    def deregister(self, name):
        status = self.db.face_deregistration(name)

        return status
