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

# Questions:
# - Face registration dann das gleiche wie One-Shot learning?
# - one-shot learning: Just with one example? But could also create with one examples e.g. 4 other examples (and store them into DB)
# - Number per class (registered name) should be equal (balanced dataset) when we use KNN
# -> either per name just one embedding (but can not just calculate mean of embeddings, as embeddings are then no longer on hypersqhere)
# -> or e. g. create 4 samples per registered name



# pytorch lightning anschauen
# error handling: what happens, if reg_database is empty and I try to recognize a face? -> directy unknown
# Modify code so that can also start with just registering one person and then add person by person
# Look how adaptive threshold behaves when just a few registered people

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

#print(database.name_list[[1,5,9,22]])


database.face_recognition(path='./test_recognition_images/Lleyton_04.ppm')

# path = './test_registration_images/John_01.ppm'
# img = database.load_and_transform_img(path)
# database.face_registration('John',img)


# database.face_deregistration('John')

# print(database.database)