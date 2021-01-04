import torch
import numpy as np

from pytorch_lightning.metrics import tensor_metric

@tensor_metric
def embedding_accuaracy(anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor, threshold=0.2) -> torch.Tensor:
    d_a_p = np.linalg.norm(anchor - positive)
    d_a_n = np.linalg.norm(anchor - negative)    
    
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