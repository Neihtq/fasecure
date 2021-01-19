"""Training a face recognizer with PyTorch and PyTorchLightning 
based on the FaceNet paper (http://arxiv.org/abs/1503.03832)"""

import os

from datetime import datetime
from torchvision import transforms

from data.LFWDataset import LFWDataset
from models.FaceNetPytorchLightning import LightningFaceNet






def main(args):
    subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')

    # log directory
    log_dir = os.path.join(os.path.expanduser(args.logs_base_dir))
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)

    # model directory
    model_dir = os.path.join(os.path.expanduser(args.models_base_dir), subdir)
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

    data_dir = os.path.join(os.path.expanduser(args.data_dir))
    transform = transforms.Compose([
        transforms.Resize((250, 250)),
        transforms.ToTensor(), 
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    dataset = LFWDataset(root=data_dir, transform=transform)

    batch_size = args.batch_size
