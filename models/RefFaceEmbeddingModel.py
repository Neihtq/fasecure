import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from data.LFWDataset import LFWEvaluationDataset
class RefFaceEmbeddingModel(nn.Module):
    ''' Reference model for lower bound of accuracy'''
    def __init__(self, dataset_path):
        super(RefFaceEmbeddingModel, self).__init__()

        # Load almost all the evaluation data and calculate the principal components
        # More data isn´t possible, since RAM is limited
        eval_dataset = LFWEvaluationDataset(dataset_path, cropped_faces=True)
        batch_size = int(eval_dataset.__len__()/1.5)
        eval_loader = torch.utils.data.DataLoader(dataset=eval_dataset,
                                                    batch_size=batch_size,
                                                    num_workers=0,
                                                    shuffle=False, 
                                                    sampler=None,
                                                    collate_fn=None)
        for label, image in eval_loader:
            #print(image.shape)
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
        
        # Flatten pixel values
        x = x.view(x.size(0), -1)

        # Transform from torch tensor to numpy array
        x = x.numpy()

        # Perform pca
        features = self.pca.transform(x)

        # Transform from numpy array to torch tensor
        features = torch.from_numpy(features)

        return features