import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import auc
from pytorch_lightning import Callback

from utils.constants import RESULTS_DIR


class MetricsCallback(Callback):
    def __init__(self):
        super().__init__()
        self.val_loss = []
        self.loss = []
        self.val_acc = []
        self.acc = []
        self.fp_rate = []
        self.tp_rate = []
        self.roc_auc = []
        self.epochs = 0
        
    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = trainer.logged_metrics
        self.val_loss.append(float(metrics['val_loss']))
        self.val_acc.append(float(metrics["val_acc"]))
        self.tp_rate.append(float(metrics["tp_rate"]))
        self.fp_rate.append(float(metrics["fp_rate"]))
        self.roc_auc.append(float(metrics["roc_auc"]))
        
    def on_epoch_end(self, trainer, pl_module):
        self.epochs += 1
        metrics = trainer.logged_metrics
        self.loss.append(metrics['train_loss'])

    def on_test_end(self, trainer, pl_module):
        metrics = trainer.logged_metrics
        acc = metrics["test_acc"]
        print(acc)
    
    def on_fit_end(self, trainer, pl_module):
        self.plot_roc()
        self.plot_accuracy()
        self.plot_loss()
    
    def plot_loss(self):
        train_loss, val_loss = self.loss, self.val_loss
        fig = plt.figure()
        epochs = self.epochs
        plt.plot(range(epochs), train_loss, color='red', label='Train Loss on VGG')
        plt.plot(range(epochs), train_loss, color='blue', label='Val Loss on VGG')
        plt.ylim(bottom=0)
        plt.title('Loss curve on VGG')
        plt.legend(loc='upper right')
        fig.savefig(os.path.join(RESULTS_DIR, "losses.png"), dpi=fig.dpi)
        plt.show()

    def plot_accuracy(self):
        accuracy_list = self.val_acc
        epochs = len(self.val_acc)
        fig = plt.figure()
        plt.plot(range(epochs), accuracy_list, color='red', label='LFW Accuracy')
        plt.ylim([0.0, 1.05])
        plt.xlim([0, epochs + 1])
        plt.xlabel('Epoch')
        plt.ylabel('LFW Accuracy')
        plt.title('Evaluation Accuracy on LFW')
        plt.legend(loc='lower right')
        fig.savefig(os.path.join(RESULTS_DIR, "lfw_eval_accuracies.png"), dpi=fig.dpi)
        plt.show()
            
    def plot_roc(self):
        false_positive_rate, true_positive_rate = self.tp_rate, self.fp_rate
        roc_auc = np.array(self.roc_auc).mean()
        fig = plt.figure()
        plt.plot(
            false_positive_rate, true_positive_rate, color='red', lw=2, label="ROC Curve (area = {:.4f})".format(roc_auc)
        )
        plt.plot([0, 1], [0, 1], color="blue", lw=2, linestyle="--", label="Random")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        fig.savefig(os.path.join(RESULTS_DIR, "ROC_LFW_evaluation.png"), dpi=fig.dpi)
        plt.show()