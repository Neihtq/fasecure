# open questions:
# - how exactly did tbmoon stored it´s images, in .csv? (first convert into .csv files in order to train?)
# - faces are appearently already aligned with David Sandberg´s MTCNN?
# - centercrop after resize with exact same size has no effect?

# DOTO:
# - calculate accuracy with current model
#   1. Parse image info into .csv
#   2. 
# - train (finetune) model on sample images (different ways) and calculate accuracy again
    # - what means "model.module.forward_classifier"

# - model training/finetuning
# - model evaluation
# - face recognition: implement KNN with fixed threshold for face recognition
# - face registration: one-shot learning (add to database)
# - implement adaptive threshold (for every person own threshold which can change if new person is added)

# Naming convention for images used for registration: name_#.(jpg,png,ppm,...)

# imports
import torch.nn.functional as F
import torch
from faceRecognitionModel import faceRecognitionModel
from prep import load_and_transform_img, show_tensor_img
from reg_database import RegistrationDatabase
from reg_dataset import RegistrationDataset
import sys


#### TESTS #####

# Create dataset
reg_dataset = RegistrationDataset("./faces_db", "ppm")

# Create dataloader with batch_size = 1
reg_loader = torch.utils.data.DataLoader(dataset=reg_dataset,
                                           batch_size=1,
                                           num_workers=0,
                                           shuffle=False, sampler=None,
                                           collate_fn=None)

# sys.exit()
#### TESTS END #####


model = faceRecognitionModel()

# If new dataset: pass dataloader to RegistrationDatabase, then it will rewrite Database
# Otherwise, it trys to return existing database
database = RegistrationDatabase(model, reg_loader)
# print(database.database)







# # adapt folders
# path1 = './faces_db/Lleyton_Hewitt_0003.ppm'
# path2 = './faces_db/Lleyton_Hewitt_0004.ppm'

# img1, img2 = load_and_transform_img(path1, path2)

# # do forward pass (128 dimensional embedding)
# embed1, embed2 = model(img1), model(img2)

# # compute the distance using euclidean distance of image embeddings (0 if the same)
# euclidean_distance = F.pairwise_distance(embed1, embed2)


# # we use 1.5 threshold to decide whether images are genuine or impostor
# threshold = 1.5

# genuine = euclidean_distance <= threshold

# print(genuine)

# print(euclidean_distance)