import argparse
import os
import sys
import timeit
import torch
import pytorch_lightning as pl

from datetime import datetime
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_metric_learning.samplers import MPerClassSampler

from utils.MetricsCallback import MetricsCallback
from data.dataset import ImageDataset
from face_detection.face_detection import face_detection
from models.FaceNetPytorchLightning import LightningFaceNet, FaceNetInceptionV3
from models.FaceNet import FaceNet
from registration_database.RegistrationDatabase import RegistrationDatabase

parser = argparse.ArgumentParser(description='Face Recognition using Triplet Loss')

parser.add_argument('--num-epochs', default=200, type=int, metavar='NE',
                    help='number of epochs to train (default: 200)')

parser.add_argument('--batch-size', default=16, type=int, metavar='BS',
                    help='batch size (default: 16)')

parser.add_argument('--num-workers', default=os.cpu_count(), type=int, metavar='NW',
                    help='number of workers (default: os.cpu_count() - Your max. amount of cpus)')

parser.add_argument('--learning-rate', default=0.05, type=float, metavar='LR',
                    help='learning rate (default: 0.05)')

parser.add_argument('--margin', default=0.02, type=float, metavar='MG',
                    help='margin for TripletLoss (default: 0.02)')

parser.add_argument('--train-data-dir', default='./data/images/lfw_crop', type=str,
                    help='path to train root dir')

parser.add_argument('--val-data-dir', default=None, type=str,
                    help='path to train root dir (if not specified, 10% of training set will be used instead')                    

parser.add_argument('--model-dir', default='./models/results', type=str,
                    help='path to train root dir')

parser.add_argument('--optimizer', default='adagrad', type=str,
                    help='Optimizer Algorithm for learning (default: adagrad')

parser.add_argument('--weight-decay', default=1e-5, type=float, metavar='SZ',
                    help='Decay learning rate (default: 1e-5)')

parser.add_argument('--pretrained', action='store_true')

parser.add_argument('--load-last', action='store_true')

parser.add_argument('--backbone', default='InceptionV3', type=str,
                    help='Deep Neural Network architecture used backbone:\n- Resnet50\n- InceptionV3 (default)')

args = parser.parse_args()

architectures = {'InceptionV3': FaceNetInceptionV3 }


def get_dataloader(dataset):
    batch_size = args.batch_size
    num_workers = args.num_workers

    transform = transforms.Compose([
        transforms.Resize((250, 250)),
        transforms.ToTensor(), 
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    
    print("Initialize DataLoader.")
    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True
    )

    return train_loader


def train():
    """Train model with PyTorchLightning""" 
    hparams = {
        'margin': args.margin,
        'lr': args.learning_rate,
        'weight_decay': args.weight_decay,
        "optimizer": args.optimizer
    }
    pretrained = args.pretrained
    num_epochs = args.num_epochs
    load_last = args.load_last
    backbone = args.backbone
    if backbone not in architectures:
        print("Specified architecture not support.")

    model_dir = os.path.expanduser(args.model_dir)
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    time_stamp = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
    subdir = os.path.join(model_dir, time_stamp)
    if not os.path.exists(subdir):
        os.mkdir(subdir)

    train_dir = os.path.expanduser(args.train_data_dir)
    train_set = ImageDataset(train_dir, transform=transform)
    labels = list(train_set.label_to_number.keys())
    sampler = MPerClassSampler(labels, 10)

    if args.val_data_dir:
        val_dir = os.path.expanduser(args.val_data_dir)
        val_set = ImageDataset(val_dir, transform=transform)
        val_loader = get_dataloader(val_dir)        
    else:
        train_length = int(0.9 * len(train_set))
        val_length = len(train_set) - train_length)
        train_set, val_set = random_split(dataset, [train_length, val_length])

    train_loader = get_dataloader(train_set)

    checkpoint_dir = './checkpoints/last_checkpoint'
    checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_dir,
        verbose=True,
        monitor='val_loss',
        mode='min',
        save_top_k=1
    )
    callback = [MetricsCallback()]
    logger = TensorBoardLogger('tb_logs', name='Training')

    model = LightningFaceNet(hparams, num_classes, architectures[backbone], pretrained=pretrained)
    trainer = pl.Trainer(
        gpus=1 if torch.cuda.is_available() else 0,
        max_epochs=num_epochs,
        logger=logger,
        checkpoint_callback=checkpoint_callback,;
        callbacks=callback
        resume_from_checkpoint=checkpoint_dir if load_last else None
    )

    print("Begin Training.")
    start = timeit.default_timer()
    trainer.fit(model, train_loader, val_loader)
    stop = timeit.default_timer()
    print("Finished Training in", stop - start, "seconds")
    
    print("Save trained weights.")
    model_name = os.path.join(subdir, time_stamp + '.pth')
    torch.save(model.model.state_dict(), model_name)


if __name__ == '__main__':
    train()
    sys.exit(0)