import torch
import numpy as np
import pytorch_lightning as pl

from torch import optim
from torch.nn import PairwiseDistance, TripletMarginLoss
from torch.nn.modules.distance import PairwiseDistance
from pytorch_lightning.metrics import Metric
from sklearn.model_selection import KFold
from sklearn.metrics import auc
from scipy import interpolate

l2_dist = PairwiseDistance(2)


class LightningFaceNet(pl.LightningModule):
    def __init__(self, hparams, model, embedding_size=128, pretrained=False):
        super(LightningFaceNet, self).__init__()
        self.hparams = hparams
        self.model = model
        self.val_metric = LFWEvalAccuracy()
        self.test_metric = LFWEvalAccuracy()
        self.loss_func = TripletMarginLoss(margin=self.hparams['margin'])

    def forward(self, x):
        return self.model(x)

    def mine_semihard(self, anc, pos, neg):
        dist_pos = l2_dist(anc, pos)
        dist_neg = l2_dist(anc, neg)

        hard_cond = torch.flatten((dist_neg - dist_pos < self.hparams['margin']))
        semihard_cond = torch.flatten((dist_pos < dist_neg))
        all_embeds = torch.logical_and(hard_cond, semihard_cond).cpu().numpy()
        triplets = np.where(all_embeds == 1)
        
        return triplets

    def general_step(self, batch):
        anc, pos, neg = batch
        anc_embed = self.forward(anc)
        pos_embed = self.forward(pos)
        neg_embed = self.forward(neg)
        triplets = self.mine_semihard(anc_embed, pos_embed, neg_embed)
        if len(triplets[0]) == 0:
            return None
        
        anc_embed, pos_embed, neg_embed = anc_embed[triplets], pos_embed[triplets], neg_embed[triplets]
        loss = self.loss_func(anc_embed, pos_embed, neg_embed)
        
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.general_step(batch)
        if loss:
            self.log("train_loss", loss)
            return {"loss": loss}
        
        return None

    def validation_step(self, batch, batch_idx):
        lfw_batch, loss_batch = batch

        img_1, img_2, same = lfw_batch
        enc_1, enc_2 = self.forward(img_1), self.forward(img_2)
        self.val_metric(enc_1, enc_2, same)

        loss = self.general_step(loss_batch)
        if loss:
            self.log("val_loss", loss)
            return {"val_loss": loss}
       
        return None

    def validation_epoch_end(self, validation_step_outputs):
        acc = self.val_metric.compute()
        self.log("val_acc", acc, logger=True)

    def test_step(self, batch, batch_idx):
        img_1, img_2, same = batch
        enc_1, enc_2 = self.forward(img_1), self.forward(img_2)
        self.test_metric(enc_1, enc_2, same)

    def test_epoch_end(self, test_step_outputs):
        acc = self.test_metric.compute()
        self.log("test_acc", acc, logger=True)

    def configure_optimizers(self):
        if self.hparams['optimizer'] == "adagrad":
            optimizer = optim.Adagrad(
                params=self.model.parameters(),
                lr=self.hparams['lr'],
                lr_decay=0,
                initial_accumulator_value=0.1,
                eps=1e-10
            )
        else:
            # self.hparams['optimizer'] == "adam"
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
        distance = self.l2_distance(enc_1, enc_2)
        self.distances.append(distance.cpu().detach().numpy())
        self.labels.append(same.cpu().detach().numpy())

    def compute(self):
        acc = self.evaluate()

        return acc.mean()

    def evaluate(self, far_target=1e-3):
        num_folds = self.num_folds
        distances, labels = self.distances, self.labels
        labels = np.array([sublabel for label in self.labels for sublabel in label])
        distances = np.array([subdist for distance in self.distances for subdist in distance])
        # Calculate ROC metrics
        thresholds_roc = np.arange(0, 5, 0.1)
        accuracy = self.calculate_roc_values(
            thresholds=thresholds_roc, distances=distances, labels=labels, num_folds=num_folds
        )
 
        return accuracy

    def calculate_roc_values(self, thresholds, distances, labels, num_folds=10):
        num_pairs = min(len(labels), len(distances))
        num_thresholds = len(thresholds)
        k_fold = KFold(n_splits=num_folds, shuffle=False)

        true_positive_rates = np.zeros((num_folds, num_thresholds))
        false_positive_rates = np.zeros((num_folds, num_thresholds))
        precision = np.zeros(num_folds)
        recall = np.zeros(num_folds)
        accuracy = np.zeros(num_folds)
        best_distances = np.zeros(num_folds)

        indices = np.arange(num_pairs)

        for fold_index, (train_set, test_set) in enumerate(k_fold.split(indices)):
            # Find the best distance threshold for the k-fold cross validation using the train set
            accuracies_trainset = np.zeros(num_thresholds)
            for threshold_index, threshold in enumerate(thresholds):
                _, _, _, _, accuracies_trainset[threshold_index] = self.calculate_metrics(
                    threshold=threshold, dist=distances[train_set], actual_issame=labels[train_set]
                )
            best_threshold_index = np.argmax(accuracies_trainset)

            # Test on test set using the best distance threshold
            for threshold_index, threshold in enumerate(thresholds):
                true_positive_rates[fold_index, threshold_index], false_positive_rates[
                    fold_index, threshold_index], _, _, _ = self.calculate_metrics(
                    threshold=threshold, dist=distances[test_set], actual_issame=labels[test_set]
                )

            _, _, precision[fold_index], recall[fold_index], accuracy[fold_index] = self.calculate_metrics(
                threshold=thresholds[best_threshold_index], dist=distances[test_set], actual_issame=labels[test_set]
            )
            best_distances[fold_index] = thresholds[best_threshold_index]

        return accuracy

    def calculate_metrics(self, threshold, dist, actual_issame):
        # If distance is less than threshold, then prediction is set to True
        predict_issame = np.less(dist, threshold)

        true_positives = np.sum(np.logical_and(predict_issame, actual_issame))
        false_positives = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
        true_negatives = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
        false_negatives = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

        # For dealing with Divide By Zero exception
        true_positive_rate = 0 if (true_positives + false_negatives == 0) else \
            float(true_positives) / float(true_positives + false_negatives)

        false_positive_rate = 0 if (false_positives + true_negatives == 0) else \
            float(false_positives) / float(false_positives + true_negatives)

        precision = 0 if (true_positives + false_positives) == 0 else \
            float(true_positives) / float(true_positives + false_positives)

        recall = 0 if (true_positives + false_negatives) == 0 else \
            float(true_positives) / float(true_positives + false_negatives)

        accuracy = float(true_positives + true_negatives) / dist.size

        return true_positive_rate, false_positive_rate, precision, recall, accuracy

