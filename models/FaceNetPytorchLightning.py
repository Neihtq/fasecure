import torch
import torch.nn as nn
import pytorch_lightning as pl

from pytorch_lightning.metrics import Metric

from .FaceNet import FaceNet


class LightningFaceNet(pl.LightningModule):
    def __init__(self, hparams, num_classes, embedding_size=128, pretrained=False):
        self.hparams = hparams
        super(LightningFaceNet, self).__init__()
        self.model = FaceNet(pretrained=pretrained, num_classes=num_classes, embedding_size=embedding_size)
        self.criterion = nn.TripletMarginLoss(margin=self.hparams["margin"], p=2)
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
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return {"loss": loss}

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
        self.log("test_loss", loss, prog_bar=True, logger=True)

    def test_epoch_end(self, test_step_outputs):
        accuracy, precision, recall, f1_score = self.test_metric.compute()
        self.log("test_acc", accuracy, prog_bar=True, logger=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams['lr'], weight_decay=1e-5)
        if self.hparams['optimizer'] == 'SGD':
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.hparams['lr'], weight_decay=0.0001)
                    
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

        accuracy = (TP + TN) / (TP + TN + FP + FN)
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        F1_Score = TP / (TP + 0.5 * (FP + FN)) 
        
        return accuracy, precision, recall, F1_Score