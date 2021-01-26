import matplotlib.pyplot as plt

from sklearn.metrics import auc
from pytorch_lightning import Callback


class MetricsCallback(Callback):
    def __init__(self):
        super().__init__()
        self.val_loss = []
        self.val_acc = []
        self.loss = []
        self.acc = []
        self.fp_rate = []
        self.tp_rate = []
        
    def on_validation_end(self, trainer, pl_module):
        metrics = trainer.logged_metrics
        self.plot_roc(metrics)
        
    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = trainer.logged_metrics
        self.val_acc.append(float(metrics["val_acc"]))
        self.tp_rate.append(float(metrics["tp_rate"]))
        self.fp_rate.append(float(metrics["fp_rate"]))
        
    def on_epoch_end(self, trainer, pl_module):
        metrics = trainer.logged_metrics
        self.loss.append(metrics['train_loss_epoch'])
        if "train_epoch_acc" in metrics:
            self.acc.append(metrics["train_epoch_acc"])

    def on_test_end(self, trainer, pl_module):
        metrics = trainer.logged_metrics
        acc = metrics["test_acc"]
            
    def on_train_end(self, trainer, pl_module):
        self.plot_accuracy()
        
    def plot_accuracy(self):
        accuracy_list = self.val_acc
        epochs = len(self.val_acc)

        fig = plt.figure()
        plt.plot(range(epochs), accuracy_list, color='red', label='LFW Accuracy')
        plt.ylim([0.0, 1.05])
        plt.xlim([0, epochs + 1])
        plt.xlabel('Epoch')
        plt.ylabel('LFW Accuracy')
        plt.title('Evluation Accuracy on LFW')
        plt.legend(loc='lower right')
        fig.savefig("lfw_eval_accuracies.png", dpi=fig.dpi)
        plt.show()
            
    def plot_roc(self, metrics):
        false_positive_rate, true_positive_rate = self.tp_rate, self.fp_rate
        roc_auc = auc(false_positive_rate, true_positive_rate)
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
        fig.savefig("ROC_LFW_evaluation.png", dpi=fig.dpi)
        plt.show()