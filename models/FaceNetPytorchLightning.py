import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchvision.models as models

from os import listdir
from pytorch_lightning.metrics import Metric
from pytorch_metric_learning import miners, losses

from .FaceNet import FaceNet


class LightningFaceNet(pl.LightningModule):
    def __init__(self, hparams, num_classes, model, embedding_size=128, pretrained=False):
        super(LightningFaceNet, self).__init__()
        self.hparams = hparams
        self.model = model
        self.train_metric = EmbeddingAccuracy()
        self.val_metric = EmbeddingAccuracy()
        self.miner = miners.TripletMarginMiner(type_of_triplets='semihard')
        self.loss_func = losses.TripletMarginLoss(margin=self.hparams['margin'])

    def forward(self, x):
        return self.model(x)
        
    def general_step(self, batch, mode):
        labels, data = batch
        embeddings = self.forward(data)
        triplets = self.miner(embeddings, labels)

        loss = self.loss_func(embeddings, labels, triplets)
        
        if mode == 'train':
            self.train_metric(triplets, embeddings)
        else:
            self.val_metric(triplets, embeddings)

        return loss
    
    def training_step(self, batch, batch_idx):
        label, path = batch
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

    def configure_optimizers(self):
        if self.hparams['optimizer'] == "sgd":
            optimizer = optim.SGD(
                params=self.model.parameters(),
                lr=self.hparams['lr'],
                momentum=0.9,
                dampening=0,
                nesterov=False
            )
        elif self.hparams['optimizer'] == "adagrad":
            optimizer = optim.Adagrad(
                params=self.model.parameters(),
                lr=self.hparams['lr'],
                lr_decay=0,
                initial_accumulator_value=0.1,
                eps=1e-10
            )
        elif self.hparams['optimizer'] == "rmsprop":
            optimizer = optim.RMSprop(
                params=self.model.parameters(),
                lr=self.hparams['lr'],
                alpha=0.99,
                eps=1e-08,
                momentum=0,
                centered=False
            )
        elif self.hparams['optimizer'] == "adam":
            optimizer = optim.Adam(
                params=self.model.parameters(),
                lr=self.hparams['lr'],
                betas=(0.9, 0.999),
                eps=1e-08,
                amsgrad=False
            )
                    
        return optimizer


class EmbeddingAccuracy(Metric):
    def __init__(self, threshold=1.242):
        super().__init__()
        self.threshold = threshold
        self.add_state("false_positive", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("true_positive", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("true_negative", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("false_negative", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, triplets, embeddings):
        anchor_enc = embeddings[triplets[0]]
        pos_enc = embeddings[triplets[1]]
        neg_enc = embeddings[triplets[2]]
        
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

        accuracy = torch.true_divide((TP + TN),(TP + TN + FP + FN))
        precision = torch.true_divide(TP,(TP + FP))
        recall = torch.true_divide(TP,(TP + FN))
        F1_Score = torch.true_divide(TP,(TP + 0.5 * (FP + FN)))
        
        return accuracy, precision, recall, F1_Score