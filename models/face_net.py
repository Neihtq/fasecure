import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np

from pytorch_lightning.metrics import Metric
from torchvision.models import resnet50


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

    
class FaceNet(nn.Module):
    def __init__(self, hparams, pretrained=False):
        super(FaceNet, self).__init__()

        self.hparams = hparams
        # pretrained is false by default, as I only need the architecture of Resnet50 and not the parameters
        # Parameters are loaded by download from github
        self.model = resnet50(pretrained)
        embedding_size = 128

        # Adapt for our case
        num_classes = 1680

        self.cnn = nn.Sequential(
            self.model.conv1,
            self.model.bn1,
            self.model.relu,
            self.model.maxpool,
            self.model.layer1,
            self.model.layer2,
            self.model.layer3,
            self.model.layer4)

        # modify fc layer based on https://arxiv.org/abs/1703.07737
        self.model.fc = nn.Sequential(
            Flatten(),
            # nn.Linear(100352, 1024),
            # nn.BatchNorm1d(1024),
            # nn.ReLU(),
            nn.Linear(2048*8*8, embedding_size))
        
        self.model.classifier = nn.Linear(embedding_size, num_classes)

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

    # returns face embedding(embedding_size)
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
        self.criterion = nn.TripletMarginLoss(margin=hparams["margin"], p=2)
        self.train_metric = EmbeddingAccuracy()
        self.val_metric = EmbeddingAccuracy()
        self.test_metric = EmbeddingAccuracy()

    def forward(self, x):
        return self.model(x)
        
    def general_step(self, batch, mode):
        label, anchor, positive, negative = batch
        
        anchor_enc = self.forward(anchor)
        pos_enc = self.forward(positive)
        neg_enc = self.forward(negative)
        
        loss = self.criterion(anchor_enc, pos_enc, neg_enc)

        if mode == 'train':
            self.train_metric(anchor_enc, pos_enc, neg_enc)
        elif mode == 'val':
            self.val_metric(anchor_enc, pos_enc, neg_enc)
        else:
            self.test_metric(anchor_enc, pos_enc, neg_enc)

        return loss
    
    def training_step(self, batch, batch_idx):
        loss = self.general_step(batch, "train")
        log = {'train_loss': loss}

        accuracy, precision, recall, f1_score = self.train_metric.compute()
        self.log("train_acc", accuracy, prog_bar=True, logger=True)

        return {"loss": loss, "log": log}

    def training_epoch_end(self, training_step_outputs):
        accuracy, precision, recall, f1_score = self.train_metric.compute()
        self.log("train_epoch_acc", accuracy, prog_bar=True, logger=True)

    def validation_step(self, batch, batch_idx):
        loss = self.general_step(batch, "val")
        log = {'val_loss': loss}
        self.log("val_loss", loss, logger=True)

        return {"val_loss": loss, "log": log}

    def validation_epoch_end(self, validation_step_outputs):
        accuracy, precision, recall, f1_score = self.val_metric.compute()
        self.log("val_acc", accuracy, logger=True)

    def test_step(self, batch, batch_idx):
        loss = self.general_step(batch, "test")
        log = {'test_loss': loss}
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", self.test_metric.compute(), prog_bar=True)
        
        return {"test_loss": loss, "log": log}
        
    def test_epoch_end(self, test_step_outputs):
        accuracy, precision, recall, f1_score = self.test_metric.compute()
        self.log("test_acc", accuracy, prog_bar=True, logger=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.model.hparams['lr'], weight_decay=1e-5)
        if self.model.hparams['optimizer'] == 'SGD':
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.model.hparams['lr'], weight_decay=0.0001)
                    
        return optimizer


class EmbeddingAccuracy(Metric):
    def __init__(self, threshold=0.2):
        super().__init__()
        self.threshold = threshold
        self.add_state("false_positive", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("true_positive", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("true_negative", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("false_negative", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, anchor_enc, pos_enc, neg_enc):
        assert anchor_enc.shape == pos_enc.shape == neg_enc.shape
        l2_pos = torch.dist(anchor_enc, pos_enc, 2)
        l2_neg = torch.dist(anchor_enc, neg_enc, 2)

        FN, TP, FP, TN = 0, 0, 0, 0
        if l2_pos > self.threshold:
            self.false_negative += 1
        else:
            self.true_negative += 1
            
        if l2_neg <= self.threshold:
            self.false_positive += 1
        else:
            self.true_positive += 1
                
    def compute(self):
        TP, TN = self.true_positive, self.true_negative
        FP, FN = self.false_positive, self.false_negative        

        accuracy = TP + TN / (TP + TN + FP + FN)
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        F1_Score = TP / (TP + 0.5 * (FP + FN)) 
        
        return accuracy, precision, recall, F1_Score