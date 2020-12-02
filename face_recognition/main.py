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

# - model training/finetuning (use data augmentation (random flix, rotate?, ...))
# - model evaluation
# - face recognition: implement KNN with fixed threshold for face recognition
# - face registration: one-shot learning (add to database)
# - implement adaptive threshold (for every person own threshold which can change if new person is added)

# Naming convention for images used for registration: name_#.(jpg,png,ppm,...)

# imports
import torch.nn.functional as F
import torch
from faceEmbeddingModel import faceEmbeddingModel
from prep import load_and_transform_img, show_tensor_img
from reg_database import RegistrationDatabase
from reg_dataset import RegistrationDataset
import sys
import numpy as np

# Questions to Omar:
# - Do we get like 20 images (10 persons, each 2 images) for the registered people?
# -> our idea: load pretrained model and finetune it with few images we get from you?
# - In order to compare a new person with the registered people, we have to compare the embeddings:
#       - Therefore, we have to store the embeddings of the registered people?
#       - As we store the embeddings, do we store just one embedding per person, or if we have multiple images, then also multiple embeddings?
#           When we have multiple embeddings, we could also store the center of all the embeddings for one person
#           (Results in a KNN classifier in the end to find the nearest embedding. However, if above threshold, then unknown)




# sys.exit()

# Create dataset
reg_dataset = RegistrationDataset("./registered_images", "ppm")

# Create dataloader with batch_size = 1
reg_loader = torch.utils.data.DataLoader(dataset=reg_dataset,
                                           batch_size=1,
                                           num_workers=0,
                                           shuffle=False, sampler=None,
                                           collate_fn=None)




embedding_model = faceEmbeddingModel()

# If new dataset: pass dataloader to RegistrationDatabase, then it will rewrite Database
# Otherwise, it trys to return existing database
database = RegistrationDatabase(embedding_model)
# print(database.database.iloc[5,1])

#print(database.name_list[[1,5,9,22]])

database.face_recognition(path='./test_images/Aaron_04.ppm')





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