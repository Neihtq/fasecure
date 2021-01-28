import torch
import torch.nn as nn

from sklearn.decomposition import PCA
from data.datasets import LFWDataset


class RefFaceEmbeddingModel(nn.Module):
    ''' Reference model for lower bound of accuracy'''
    def __init__(self, dataset_path):
        super(RefFaceEmbeddingModel, self).__init__()
        eval_dataset = LFWDataset(dataset_path, cropped_faces=True)
        batch_size = int(eval_dataset.__len__() / 1.5)
        eval_loader = torch.utils.data.DataLoader(dataset=eval_dataset,
                                                    batch_size=batch_size,
                                                    num_workers=0,
                                                    shuffle=False, 
                                                    sampler=None,
                                                    collate_fn=None)
        for label, image in eval_loader:
            x = image.view(image.size(0), -1)
            x = x.numpy()
            self.pca = PCA(n_components=128)
            self.pca.fit(x)
            break


    def l2_norm(self, input):
        input_size = input.size()
        buffer = torch.pow(input, 2)
        normp = torch.sum(buffer, 1).add_(1e-10)
        norm = torch.sqrt(normp)
        _output = torch.div(input, norm.view(-1, 1).expand_as(input))
        output = _output.view(input_size)
        return output

    def forward(self, x):
        x = x.view(x.size(0), -1)
        features = self.pca.transform(x.numpy())
        features = torch.from_numpy(features)

        return features