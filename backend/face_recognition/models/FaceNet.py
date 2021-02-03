import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import resnet50, inception_v3, resnet18
from torch.hub import download_url_to_file

from face_recognition.utils.constants import PRETRAINED_URL, PRETRAINED_MODEL_DIR, MODEL_DIR, TRAINED_WEIGHTS_DIR

# change to TRAINED_WEIGHTS_DIR
def load_weights(weight_path=TRAINED_WEIGHTS_DIR):
    model = FaceNetResnet()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.load_state_dict(torch.load(weight_path, map_location=torch.device(device))['model_state_dict'])
    
    return model



def load_state():
    cached_file = os.path.join(PRETRAINED_MODEL_DIR, os.path.basename(PRETRAINED_URL))
    if not os.path.exists(cached_file):
        download_url_to_file(PRETRAINED_URL, cached_file)

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


class FaceNetResnet(nn.Module):
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

        return x


