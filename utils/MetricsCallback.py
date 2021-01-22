import os
import matplotlib.pyplot as plt

from pytorch_lightning import Callback


class MetricsCallback(Callback):
    def __init__(self):
        super().__init__()
        self.val_loss = []
        self.val_acc = []
        self.loss = []
        self.acc = []
        
    def on_validation_end(self, trainer, pl_module):
        self.val_loss.append(trainer.logged_metrics['val_loss'])
        self.val_acc.append(trainer.logged_metrics['val_acc'])
        
    def on_epoch_end(self, trainer, pl_module):
        metrics = trainer.logged_metrics
        self.loss.append(metrics['train_loss_epoch'])
        if "train_epoch_acc" in metrics:
            self.acc.append(metrics["train_epoch_acc"])
        
    def on_train_end(self, trainer, pl_module):
        plt.style.use('seaborn-darkgrid')
        self.create_figures(self.loss, self.val_loss, 'loss')
        self.create_figures(self.acc, self.val_acc, 'accuracy')
        
    def create_figures(self, metric_1, metric_2, metric_name):
        train_metric, = plt.plot(range(len(metric_1)), metric_1)
        val_metric, = plt.plot(range(len(metric_2)), metric_2)
        plt.xticks(range(len(metric_2) if len(metric_2) >= len(metric_2) else len(metric_2)))
        plt.legend([train_metric, val_metric], [f"training {metric_name}", f"validation {metric_name}"])
        
        save_path = os.path.join('.', 'results', 'metrics')
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        save_path = os.path.join(save_path, f"{metric_name}_curves.png")

        plt.savefig(save_path)
        plt.show()