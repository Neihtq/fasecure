import torch.nn as nn

# Reference model for lower bound of accuracy
class refFaceEmbeddingModel(nn.Module):
    def __init__(self):
        super(refFaceEmbeddingModel, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)