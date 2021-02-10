import os
import sys
import timeit
import argparse
import torch
import pytorch_lightning as pl

from datetime import datetime
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from face_recognition.models.MetricsCallback import MetricsCallback
from face_recognition.models.FaceNetPytorchLightning import LightningFaceNet
from face_recognition.models.FaceNet import FaceNetResnet
from face_recognition.data.datasets import LFWValidationDataset, TupleDataset, VGGTripletDataset
from face_recognition.utils.constants import MODEL_DIR, CHECKPOINTS_DIR

parser = argparse.ArgumentParser(description='Face Recognition using Triplet Loss')

parser.add_argument('--num-epochs', default=200, type=int, metavar='NE',
                    help='Number of epochs to train (default: 200)')

parser.add_argument('--num-triplets', default=100000, type=int, metavar='NTT',
                    help='Number of triplets for training (default: 10000)')

parser.add_argument('--batch-size', default=256, type=int, metavar='BS',
                    help='Batch size (default: 256)')

parser.add_argument('--learning-rate', default=0.001, type=float, metavar='LR',
                    help='Learning rate (default: 0.05)')

parser.add_argument('--margin', default=0.2, type=float, metavar='MG',
                    help='Margin for TripletLoss (default: 0.02)')

parser.add_argument('--train-data-dir', default=None, type=str,
                    help='Path to training data')

parser.add_argument('--val-data-dir', default=None, type=str,
                    help='Path to validation data')

parser.add_argument('--val-labels-dir', default=None, type=str,
                    help='Path to pairs.txt of validation data.')

parser.add_argument('--model-dir', default=MODEL_DIR, type=str,
                    help='Path where model will be saved')

parser.add_argument('--optimizer', default='adam', type=str,
                    help='Optimizer Algorithm for learning (default: adam')

parser.add_argument('--weight-decay', default=1e-5, type=float, metavar='SZ',
                    help='Decay learning rate (default: 1e-5)')

parser.add_argument('--load-checkpoint', default=None, type=str,
                    help='Path to checkpoint.')

parser.add_argument('--num-workers', default=os.cpu_count(), type=int, metavar='NM',
                    help='Number of workers for dataloader (default: amount of cpu core')

args = parser.parse_args()


def get_dataloader(dataset, train=False):
    batch_size = args.batch_size
    num_workers = args.num_workers
    phase = "training" if train else 'validation'
    print(f"Initialize {phase} dataloader.")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return dataloader


def init_datasets():
    print("Initialize datasets...")
    train_dir = os.path.expanduser(args.train_data_dir)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_set = VGGTripletDataset(train_dir, args.num_triplets, transform=transform)

    val_loader = None
    if args.val_data_dir and args.val_labels_dir:
        lfw_set = LFWValidationDataset(args.val_data_dir, args.val_labels_dir, transform=transform)
        len_lfw_set = len(lfw_set)

        len_train_set = len(train_set) - len_lfw_set
        train_set, val_set = random_split(train_set, [len_train_set, len_lfw_set])

        tuple_set = TupleDataset(lfw_set, val_set)
        val_loader = get_dataloader(tuple_set)

    train_loader = get_dataloader(train_set, train=True)

    return train_loader, val_loader


def train():
    """Train model with PyTorchLightning"""
    hparams = {
        'margin': args.margin,
        'lr': args.learning_rate,
        'weight_decay': args.weight_decay,
        "optimizer": args.optimizer
    }
    load_checkpoint = args.load_checkpoint
    num_epochs = args.num_epochs

    model_dir = os.path.expanduser(args.model_dir)
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    time_stamp = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
    subdir = os.path.join(model_dir, time_stamp)
    if not os.path.exists(subdir):
        os.mkdir(subdir)

    train_dir = os.path.expanduser(args.train_data_dir)
    if not train_dir:
        raise ValueError('No training data specified.')

    train_loader, val_loader = init_datasets()

    checkpoint_callback = ModelCheckpoint(
        filepath=CHECKPOINTS_DIR,
        verbose=True,
        monitor='val_acc',
        mode='max',
        save_top_k=1
    )
    logger = TensorBoardLogger('tb_logs', name='facesecure_training')
    print("Initialize FaceNet + Resnet")
    backbone = FaceNetResnet(pretrained=True)
    model = LightningFaceNet(hparams, backbone)

    if load_checkpoint:
        model = LightningFaceNet.load_from_checkpoint(load_checkpoint, hparams=hparams, model=model)

    trainer = pl.Trainer(
        gpus=1 if torch.cuda.is_available() else 0,
        max_epochs=num_epochs,
        logger=logger,
        checkpoint_callback=checkpoint_callback,
        callbacks=[MetricsCallback()]
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