import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from torchvision.models import resnet50
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
    model = FaceNet()
    if pretrained:
        state = load_state()
        model.load_state_dict(state['state_dict'])
    return model


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

    
class FaceNet(nn.Module):
    def __init__(self, hparams, pretrained=False, num_classes=1680, embedding_size=128):
        super(FaceNet, self).__init__()

        self.hparams = hparams
        # pretrained is false by default, as I only need the architecture of Resnet50 and not the parameters
        # Parameters are loaded by download from github
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

        # modify fc layer based on https://arxiv.org/abs/1703.07737
        self.model.fc = nn.Sequential(
            Flatten(),
            nn.Linear(2048*8*8, embedding_size)
        )
        
        self.model.classifier = nn.Linear(embedding_size, num_classes)
        self.criterion = nn.TripletMarginLoss(margin=self.hparams["margin"], p=2)

    def l2_norm(self, input):
        input_size = input.size()
        buffer = torch.pow(input, 2)
        normp = torch.sum(buffer, 1).add_(1e-10)
        norm = torch.sqrt(normp)
        _output = torch.div(input, norm.view(-1, 1).expand_as(input))
        output = _output.view(input_size)
        return output

    def freeze_all(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def unfreeze_all(self):
        for param in self.model.parameters():
            param.requires_grad = True

    def freeze_fc(self):
        for param in self.model.fc.parameters():
            param.requires_grad = False

    def unfreeze_fc(self):
        for param in self.model.fc.parameters():
            param.requires_grad = True

    def freeze_only(self, freeze):
        for name, child in self.model.named_children():
            if name in freeze:
                for param in child.parameters():
                    param.requires_grad = False
            else:
                for param in child.parameters():
                    param.requires_grad = True

    def unfreeze_only(self, unfreeze):
        for name, child in self.model.named_children():
            if name in unfreeze:
                for param in child.parameters():
                    param.requires_grad = True
            else:
                for param in child.parameters():
                    param.requires_grad = False

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

    
    
class LightningFaceNet(pl.LightningModule):
    def __init__(self, hparams, pretrained=False):
        super(LightningFaceNet, self).__init__()
        self.model = FaceNet(hparams, pretrained=pretrained)
        
    def forward(self, x):
        return self.model(x)
        
    def general_step(self, batch):
        label, anchor, positive, negative = batch
        
        anchor_enc = self.forward(anchor)
        pos_enc = self.forward(positive)
        neg_enc = self.forward(negative)
        
        loss = self.model.criterion(anchor_enc, pos_enc, neg_enc)

        return loss
    
    def training_step(self, batch, batch_idx):
        loss = self.general_step(batch)
        log = {'train_loss': loss}
        
        return {"loss": loss, "log": log}
        
    def validation_step(self, batch, batch_idx):
        loss = self.general_step(batch)
        log = {'val_loss': loss}
        self.log('val_loss', loss)
        
        return {"val_loss": loss, "log": log}
    
    def test_step(self, batch, batch_idx):
        loss = self.general_step(batch)
        log = {'test_loss': loss}
        
        return {"test_loss": loss, "log": log}
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.model.hparams['lr'], weight_decay=1e-5)
        if self.model.hparams['optimizer'] == 'SGD':
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.model.hparams['lr'], weight_decay=0.0001)
                    
        return optimizer