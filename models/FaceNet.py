import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torchvision.models import resnet50, inception_v3
from torch.hub import download_url_to_file


def load_state():
    path = 'https://github.com/khrlimam/facenet/releases/download/acc-0.92135/model921-af60fb4f.pth'

    model_dir = "./pretrained_model"
    os.makedirs(model_dir, exist_ok=True)

    cached_file = os.path.join(model_dir, os.path.basename(path))
    if not os.path.exists(cached_file):
        download_url_to_file(path, cached_file)

    state_dict = torch.load(cached_file)  
    
    return state_dict

def get_model(pretrained=True):
    model = FaceNet(pretrained)
    if pretrained:
        state = load_state()
        model.load_state_dict(state['state_dict'])
    return model


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

    
class FaceNet(nn.Module):
    def __init__(self, pretrained=False, num_classes=500, embedding_size=128):
        super(FaceNet, self).__init__()
        self.model = resnet50(pretrained)
        self.cnn = nn.Sequential(
            self.model.conv1,
            self.model.bn1,
            self.model.relu,
            self.model.maxpool,
            self.model.layer1,
            self.model.layer2,
            self.model.layer3,
            self.model.layer4
        )

        fc_dim = 2048 * 8 * 8
        if pretrained:
            fc_dim = 100352

        self.model.fc = nn.Sequential(
            Flatten(),
            nn.Linear(fc_dim, embedding_size)
        )
        
        self.model.classifier = nn.Linear(embedding_size, num_classes)

    def l2_norm(self, input):
        input_size = input.size()
        buffer = torch.pow(input, 2)
        normp = torch.sum(buffer, 1).add_(1e-10)
        norm = torch.sqrt(normp)
        _output = torch.div(input, norm.view(-1, 1).expand_as(input))
        output = _output.view(input_size)
        return output

    def forward(self, x):
        x = self.cnn(x)
        x = self.model.fc(x)

        features = self.l2_norm(x)
        alpha = 10
        features = features * alpha
        
        return features

    def forward_classifier(self, x):
        features = self.forward(x)
        res = self.model.classifier(features)
        return res


class FaceNetInceptionV3(nn.Module):
    def __init__(self, embedding_dimension=128, pretrained=False):
        super(FaceNetInceptionV3, self).__init__()
        self.model = inception_v3(pretrained=pretrained)

        input_features_fc_layer = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(input_features_fc_layer, embedding_dimension, bias=False),
            nn.BatchNorm1d(embedding_dimension, eps=0.001, momentum=0.1, affine=True)
        )

    def forward(self, x):
        x = self.model(x)
        x = F.normalize(x, p=2, dim=1)
        
        return x