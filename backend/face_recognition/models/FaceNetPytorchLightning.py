import torch
import numpy as np
import pytorch_lightning as pl

from torch import optim
from torch.nn import PairwiseDistance, TripletMarginLoss
from torch.nn.modules.distance import PairwiseDistance
from pytorch_lightning.metrics import Metric
from pytorch_metric_learning import miners, losses
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
        triplets = torch.from_numpy(np.where(all_embeds == 1))
        
        return anc[triplets], pos[triplets], neg[triplets]

    def general_step(self, batch):
        anc, pos, neg = batch
        anc_embed = self.forward(anc)
        pos_embed = self.forward(pos)
        neg_embed = self.forward(neg)
        anc_embed, pos_embed, neg_embed = self.mine_semihard(anc_embed, pos_embed, neg_embed)
        loss = loss_func(anc_embed, pos_embed, neg_embed)

        return loss

    def training_step(self, batch, batch_idx):
        loss = self.general_step(batch)
        self.log("train_loss", loss)

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        lfw_batch, loss_batch = batch

        img_1, img_2, same = lfw_batch
        enc_1, enc_2 = self.forward(img_1), self.forward(img_2)
        self.val_metric(enc_1, enc_2, same)

        loss = self.general_step(loss_batch)
        self.log("val_loss", loss)

        return {"val_loss": loss}

    def validation_epoch_end(self, validation_step_outputs):
        tp_rate, fp_rate, precision, recall, acc, roc_auc, best_dist, tar, far = self.val_metric.compute()
        self.log_evaluation(tp_rate, fp_rate, precision, recall, roc_auc, best_dist, tar, far)
        self.log("val_acc", acc, logger=True)

    def test_step(self, batch, batch_idx):
        img_1, img_2, same = batch
        enc_1, enc_2 = self.forward(img_1), self.forward(img_2)
        self.test_metric(enc_1, enc_2, same)

    def test_epoch_end(self, test_step_outputs):
        tp_rate, fp_rate, precision, recall, acc, roc_auc, best_dist, tar, far = self.test_metric.compute()
        self.log_evaluation(tp_rate, fp_rate, precision, recall, roc_auc, best_dist, tar, far)
        self.log("test_acc", acc, logger=True)

    def log_evaluation(self, tp_rate, fp_rate, precision, recall, roc_auc, best_dist, tar, far):
        self.log("tp_rate", tp_rate, logger=True)
        self.log("fp_rate", fp_rate, logger=True)
        self.log("precision", precision, logger=True)
        self.log("recall", recall, logger=True)
        self.log("roc_auc", roc_auc, logger=True)

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
        tp_rate, fp_rate, precision, recall, acc, roc_auc, best_dist, tar, far = self.evaluate()

        return tp_rate.mean(), fp_rate.mean(), precision.mean(), recall.mean(), acc.mean(), roc_auc.mean(), best_dist.mean(), tar.mean(), far.mean()

    def evaluate(self, far_target=1e-3):
        num_folds = self.num_folds
        distances, labels = self.distances, self.labels
        labels = np.array([sublabel for label in self.labels for sublabel in label])
        distances = np.array([subdist for distance in self.distances for subdist in distance])
        # Calculate ROC metrics
        thresholds_roc = np.arange(0, 5, 0.1)
        true_positive_rate, false_positive_rate, precision, recall, accuracy, best_distances = self.calculate_roc_values(
            thresholds=thresholds_roc, distances=distances, labels=labels, num_folds=num_folds
        )

        roc_auc = auc(false_positive_rate, true_positive_rate)

        # Calculate validation rate
        thresholds_val = np.arange(0, 50, 0.001)
        tar, far = self.calculate_val(
            thresholds_val=thresholds_val, distances=distances, labels=labels, far_target=far_target,
            num_folds=num_folds
        )

        return true_positive_rate, false_positive_rate, precision, recall, accuracy, roc_auc, best_distances, tar, far

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

            true_positive_rate = np.mean(true_positive_rates, 0)
            false_positive_rate = np.mean(false_positive_rates, 0)
            best_distances[fold_index] = thresholds[best_threshold_index]

        return true_positive_rate, false_positive_rate, precision, recall, accuracy, best_distances

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

    def calculate_val(self, thresholds_val, distances, labels, far_target=1e-3, num_folds=10):
        num_pairs = min(len(labels), len(distances))
        num_thresholds = len(thresholds_val)
        k_fold = KFold(n_splits=num_folds, shuffle=False)

        tar = np.zeros(num_folds)
        far = np.zeros(num_folds)

        indices = np.arange(num_pairs)

        for fold_index, (train_set, test_set) in enumerate(k_fold.split(indices)):
            # Find the euclidean distance threshold that gives false acceptance rate (far) = far_target
            far_train = np.zeros(num_thresholds)
            for threshold_index, threshold in enumerate(thresholds_val):
                _, far_train[threshold_index] = self.calculate_val_far(
                    threshold=threshold, dist=distances[train_set], actual_issame=labels[train_set]
                )
            if np.max(far_train) >= far_target:
                f = interpolate.interp1d(far_train, thresholds_val, kind='slinear')
                threshold = f(far_target)
            else:
                threshold = 0.0

            tar[fold_index], far[fold_index] = self.calculate_val_far(
                threshold=threshold, dist=distances[test_set], actual_issame=labels[test_set]
            )

        return tar, far

    def calculate_val_far(self, threshold, dist, actual_issame):
        # If distance is less than threshold, then prediction is set to True
        predict_issame = np.less(dist, threshold)

        true_accept = np.sum(np.logical_and(predict_issame, actual_issame))
        false_accept = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))

        num_same = np.sum(actual_issame)
        num_diff = np.sum(np.logical_not(actual_issame))

        if num_diff == 0:
            num_diff = 1
        if num_same == 0:
            return 0, 0

        tar = float(true_accept) / float(num_same)
        far = float(false_accept) / float(num_diff)

        return tar, far