import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from torchvision.models import resnet50


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

    
class FaceNet(nn.Module):
    def __init__(self, hparams, pretrained=False):
        super(FaceNet, self).__init__()

        self.hparams = hparams
        # pretrained is false by default, as I only need the architecture of Resnet50 and not the parameters
        # Parameters are loaded by download from github
        self.model = resnet50(pretrained)
        embedding_size = 128

        # Adapt for our case
        num_classes = 1680

        self.cnn = nn.Sequential(
            self.model.conv1,
            self.model.bn1,
            self.model.relu,
            self.model.maxpool,
            self.model.layer1,
            self.model.layer2,
            self.model.layer3,
            self.model.layer4)

        # modify fc layer based on https://arxiv.org/abs/1703.07737
        self.model.fc = nn.Sequential(
            Flatten(),
            # nn.Linear(100352, 1024),
            # nn.BatchNorm1d(1024),
            # nn.ReLU(),
            nn.Linear(2048*8*8, embedding_size))
        
        self.model.classifier = nn.Linear(embedding_size, num_classes)
        self.criterion = nn.TripletMarginLoss(margin=self.hparams["margin"],p=2)

    def l2_norm(self, input):
        input_size = input.size()
        buffer = torch.pow(input, 2)
        normp = torch.sum(buffer, 1).add_(1e-10)
        norm = torch.sqrt(normp)
        _output = torch.div(input, norm.view(-1, 1).expand_as(input))
        output = _output.view(input_size)
        return output

    def freeze_all(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def unfreeze_all(self):
        for param in self.model.parameters():
            param.requires_grad = True

    def freeze_fc(self):
        for param in self.model.fc.parameters():
            param.requires_grad = False

    def unfreeze_fc(self):
        for param in self.model.fc.parameters():
            param.requires_grad = True

    def freeze_only(self, freeze):
        for name, child in self.model.named_children():
            if name in freeze:
                for param in child.parameters():
                    param.requires_grad = False
            else:
                for param in child.parameters():
                    param.requires_grad = True

    def unfreeze_only(self, unfreeze):
        for name, child in self.model.named_children():
            if name in unfreeze:
                for param in child.parameters():
                    param.requires_grad = True
            else:
                for param in child.parameters():
                    param.requires_grad = False

    # returns face embedding(embedding_size)
    def forward(self, x):
        x = self.cnn(x)
        x = self.model.fc(x)

        features = self.l2_norm(x)
        alpha = 10
        features = features * alpha
        
        return features

    def forward_classifier(self, x):
        features = self.forward(x)
        res = self.model.classifier(features)
        return res

    

    
    
class LightningFaceNet(pl.LightningModule):
    def __init__(self, hparams, pretrained=False):
        super(LightningFaceNet, self).__init__()
        self.model = FaceNet(hparams, pretrained=pretrained)
        
    def forward(self, x):
        return self.model(x)
        
    def general_step(self, batch):
        label, anchor, positive, negative = batch
        
        anchor_enc = self.forward(anchor)
        pos_enc = self.forward(positive)
        neg_enc = self.forward(negative)
        
        loss = self.model.criterion(anchor_enc, pos_enc, neg_enc)

        return loss
    
    def training_step(self, batch, batch_idx):
        loss = self.general_step(batch)
        log = {'train_loss': loss}
        
        return {"loss": loss, "log": log}
        
    def validation_step(self, batch, batch_idx):
        loss = self.general_step(batch)
        log = {'val_loss': loss}
        self.log('val_loss', loss)
        
        accuracy, precision, recall, F1_Score = self.evaluation(batch)
        
        print("Validation Accuracy: " + accuracy)
        print("Validation Precision: " + precision)
        print("Validation Recall: " + recall)
        print("Validation F1_Score: " + F1_Score)
        
        return {"val_loss": loss, "log": log, "val_acc": accuracy, "val_precision": precision, "val_recall": recall, "val_F1": F1_Score} 
        
    def evaluation(self, batch):
        for x in batch:
            #img_1 = load_and_transform_img(x)
            label, anchor, positive, negative = batch
            anchor_enc = self.forward(anchor)
            pos_enc = self.forward(positive)
            neg_enc = self.forward(negative)  
            
            anchor_enc_np = anchor_enc.numpy()
            pos_enc_np = pos_enc.numpy()
            neg_enc_np = neg_enc.numpy()
            
            d_a_p = numpy.linalg.norm(anchor_enc_np - pos_enc_np)
            d_a_n = numpy.linalg.norm(anchor_enc_np - neg_enc_np)
            
            threshold = 0.02
            
            FN, TP, FP, TN = 0, 0, 0, 0
            if d_a_p > threshold:
                FN += 1
            else:
                TP += 1
                
            if d_a_n <= threshold:
                FP += 1
            else:
                TN += 1
            
        accuracy = TP + TN / (TP + TN + FP + FN)
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        F1_Score = TP / (TP + 0.5 * (FP + FN)) 
        
        return accuracy, precision, recall, F1_Score
    
    def test_step(self, batch, batch_idx):
        loss = self.general_step(batch)
        log = {'test_loss': loss}
        
        print("Test Accuracy: " + accuracy)
        print("Test Precision: " + precision)
        print("Test Recall: " + recall)
        print("Test F1_Score: " + F1_Score)
        
        return {"test_loss": loss, "log": log, "test_acc": accuracy, "test_precision": precision, "test_recall": recall, "test_F1": F1_Score}
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.model.hparams['lr'], weight_decay=1e-5)
        if self.model.hparams['optimizer'] == 'SGD':
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.model.hparams['lr'], weight_decay=0.0001)
                    
        return optimizer