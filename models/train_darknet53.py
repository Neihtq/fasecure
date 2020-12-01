import torch

from torch import nn, optim
from torchvision import transforms

from darknet53 import DarkNet53

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = DarkNet53(2) # two classes: face and no-face
    mode.to(device)

    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)

    train_dir = "./../data/train/"
    val_dir = "./../data/val/"

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    