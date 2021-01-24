import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchvision.models as models

from os import listdir
from torch.nn import PairwiseDistance
from pytorch_lightning.metrics import Metric
from pytorch_metric_learning import miners, losses
from sklearn.model_selection import KFold
from sklearn.metrics import auc
from scipy import interpolate

from .FaceNet import FaceNet


class LightningFaceNet(pl.LightningModule):
    def __init__(self, hparams, model, embedding_size=128, pretrained=False):
        super(LightningFaceNet, self).__init__()
        self.hparams = hparams
        self.model = model
        self.val_metric = LFWEvalAccuracy()
        self.test_metric = LFWEvalAccuracy()
        self.miner = miners.TripletMarginMiner(type_of_triplets='semihard')
        self.loss_func = losses.TripletMarginLoss(margin=self.hparams['margin'])

    def forward(self, x):
        return self.model(x)
            
    def training_step(self, batch, batch_idx):
        labels, data = batch
        embeddings = self.forward(data)
        triplets = self.miner(embeddings, labels)
        loss = self.loss_func(embeddings, labels, triplets)
        self.log("train_loss", loss)

        return {"loss": loss}

    def training_epoch_end(self, training_step_outputs):
        accuracy, precision, recall, f1_score = self.train_metric.compute()
        self.log("train_epoch_acc", accuracy, prog_bar=True, logger=True)

    def validation_step(self, batch, batch_idx):      
        img_1, img_2, same = batch
        enc_1, enc_2 = self.forward(img_1), self.forward(img_2) 
                
        self.val_metric(img_1, img_2, same)
        
    def validation_epoch_end(self, validation_step_outputs):
        tp_rate, fp_rate, precision, recall, acc, best_dist, roc_auc = self.val_metric.compute()
        self.log("val_acc", acc, logger=True)
        self.log("tp_rate", tp_rate, logger=True)
        self.log("fp_rate", fp_rate, logger=True)
        self.log("precision", precision, logger=True)
        self.log("recall", recall, logger=True)
        self.log("roc_auc", roc_auc, logger=True)

        return tp_rate, fp_rate, acc

    def test_step(self, batch, batch_idx):
        img_1, img_2, same = batch
        enc_1, enc_2 = self.forward(img_1), self.forward(img_2)
        self.test_metric(img_1, img_2, same)

    def test_epoch_end(self, test_step_outputs):
        tp_rate, fp_rate, precision, recall, acc, best_dist, roc_auc = self.test_metric.compute()
        self.log("test_acc", acc, logger=True)
        self.log("tp_rate", tp_rate, logger=True)
        self.log("fp_rate", fp_rate, logger=True)
        self.log("precision", precision, logger=True)
        self.log("recall", recall, logger=True)
        self.log("roc_auc", roc_auc, logger=True)

        return tp_rate, fp_rate, acc

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


class LFWEvalAccuracy(Metric):
    def __init__(self, num_folds=10, far_target=1e-3, compute_on_step=False):
        super().__init__(compute_on_step=False)
        self.num_folds = num_folds
        self.fa_rate_target = far_target
        self.l2_distance = PairwiseDistance(p=2)
        self.distances = []
        self.labels = []

    def update(self, enc_1, enc_2, same):      
        dist = self.l2_distance(enc_1, enc_2)
        self.distances.append(dist)
        self.labels.append(same)

    def compute(self):
        tp_rate, fp_rate, precision, recall, acc, best_dist, roc_auc = self.evaluate()
        return tp_rate, fp_rate, precision, recall, acc, best_dist, roc_auc

    def evaluate(self):
        thresholds = torch.arange(0, 4, 0.01)
        tp_rate, fp_rate, precision, recall, acc, best_dist = self.calculate_roc(thresholds)
        roc_auc = auc(tp_rate, fp_rate)

        return tp_rate, fp_rate, precision, recall, acc, best_dist, roc_auc

    def calculate_roc(self, thresholds):
        distances_tensor = torch.vstack(self.distances)
        labels_tensor = torch.vstack(self.labels)
        num_pairs = min(len(distances_tensor), len(labels_tensor))
        num_thresholds = len(thresholds)
        k_fold = KFold(n_splits=self.num_folds, shuffle=False)

        tp_rates = torch.zeros((self.num_folds, num_thresholds))
        fp_rates= torch.zeros((self.num_folds, num_thresholds))
        precision = torch.zeros(self.num_folds)
        recall = torch.zeros(self.num_folds)
        acc = torch.zeros(self.num_folds)
        best_dists = torch.zeros(self.num_folds)

        indices = torch.arange(num_pairs)
        for fold_index, (train_set, test_set) in enumerate(k_fold.split(indices)):
            acc_train = torch.zeros(num_thresholds)
            for threshold_index, threshold in enumerate(thresholds):
                _, _, _, _, acc_trainset[threshold_index] = self.calculate_metrics(
                    threshold=threshold, dist=self.distances[train_set], label=self.labels[train_set]
                )
            best_threshold_index = torch.argmax(acc_trainset)

            for threshold_index, threshold in enumerate(thresholds):
                tp_rate[fold_index, threshold_index], fp_rates[fold_index, threshold_index], _, _, _ = self.calculate_metrics(
                    threshold=threshold, dist=self.distances[test_set], label=self.labels[test_set]
                )

            _, _, precision[fold_index], recall[fold_index], acc[fold_index] = self.calculate_metrics(
                threshold=threshold, dist=self.distances[test_set], label=self.labels[test_set]
            )

            tp_rate = torch.mean(tp_rates, 0)
            fp_rate = torch.mean(tp_rates, 0)
            best_dists[fod_index] = thresholds[best_threshold_index]

        return tp_rate, fp_rate, precision, recall, accuracy, best_distances
            
    def calculate_metrics(self, dist, label):
        # distance les than threshold -> prediction = True
        pred = torch.less(dist, threshold)        

        tp = torch.sum(torch.logical_and(pred, label))
        fp = torch.sum(torch.logical_and(pred, torch.logical_not(label)))
        tn = torch.sum(torch.logical_and(torch.logical_not(pred), torch.logical_not(label)))
        fn = torch.sum(torch.logical_and(torch.logical_not(pred), label))

        tp_rate = 0 if (tp + fn == 0 ) else float(tp) / float(tp + fn)
        fp_rate = 0 if (fp + tn == 0 ) else float(fp) / float(fp + tn)
        precision = 0 if (tp + fp) == 0 else float(tp) / float(tp + fp)
        recall = 0 if (tp + fn) == 0 else float(tp) / float(tp + fn)
        accuracy = float(tp + tn) /  torch.numel()

        return tp_rate, fp_rate, precision, recall, accuracy