import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import resnet50, resnet18
from torch.hub import download_url_to_file

from face_recognition.utils.constants import TRAINED_WEIGHTS_DIR, FACESECURE_MODEL

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_pretrained(weight_path=TRAINED_WEIGHTS_DIR):
    model = FaceNetResnet()
    model.load_state_dict(torch.load(weight_path, map_location=torch.device(device))['model_state_dict'])
    
    return model


def get_model(pretrained=True, model_path=FACESECURE_MODEL):
    model = FaceNet(pretrained)
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))

    return model


class FaceNet(nn.Module):
    def __init__(self, pretrained=False, embedding_size=128):
        super(FaceNet, self).__init__()
        resnet = resnet50(pretrained)
        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4
        )
        in_features = 2048 * 7 * 7 # input image of backbone is of shape (3, 224, 244)
        self.embedder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, embedding_size)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.embedder(x)
        x = F.normalize(x, p=2, dim=1)
        
        x = x * 10  # alpha = 10

        return x


class FaceNetResnet(nn.Module):
    '''FaceNet with Resnet backbone, inspired by pre_trained model'''
    def __init__(self, embedding_dimension=256, pretrained=False):
        super(FaceNetResnet, self).__init__()
        self.model = resnet18(pretrained=pretrained)

        input_features_fc_layer = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(input_features_fc_layer, embedding_dimension, bias=False),
            nn.BatchNorm1d(embedding_dimension, eps=0.001, momentum=0.1, affine=True)
        )

    def forward(self, x):
        x = self.model(x)
        x = F.normalize(x, p=2, dim=1)
        alpha = 10
        x = x * alpha

        return x

