import torch
from torch import nn

def conv_batch(in_num, out_num, kernel_size=3, padding=1, stride=1):
    conv_batch_norm = nn.Sequential(
        nn.Conv2d(in_num, out_num, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_num),
        nn.LeakyReLU()
    )

    return conv_batch_norm


class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        reduced_channels = int(in_channels / 2)

        self.block = nn.Sequential(
            conv_batch(in_channels, reduced_channels, kernel_size=1, padding=0),
            conv_batch(reduced_channels, in_channels)
        )

    
    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        
        return out

    
class DarkNet53(nn.Module):
    def __init__(self, num_classes):
        super(DarkNet53, self).__init__()

        block = ResidualBlock

        self.model = nn.Sequential(
            conv_batch(3, 32),
            conv_batch(32, 64, stride=2),
            self.make_blocklayer(block, in_channels=64, num_blocks=1),
            conv_batch(64, 128, stride=2),
            self.make_blocklayer(block, in_channels=128, num_blocks=2),
            conv_batch(128, 256, stride=2),
            self.make_blocklayer(block, in_channels=256, num_blocks=8),
            conv_batch(256, 512, stride=2),
            self.make_blocklayer(block, in_channels=512, num_blocks=8),
            conv_batch(512, 1024, stride=2),
            self.make_blocklayer(block, in_channels=1024, num_blocks=4),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Linear(1024, self.num_classes)
        )


    def forward(self, x):
        out = self.model(x)

        return out

    
    def make_blocklayer(self, block, in_channels, num_blocks):
        layers = []
        for i in range(0, num_blocks):
            layers.append(block(in_channels))
        
        return nn.Sequential(*layers)