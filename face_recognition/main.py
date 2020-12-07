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

# - transformation of input images has to be handled by faceEmbeddingModel and not by database (in face registration and face recognition)

# imports
import torch.nn.functional as F
import torch
from faceEmbeddingModel import faceEmbeddingModel
# from prep import load_and_transform_img, show_tensor_img
from reg_database import RegistrationDatabase
from reg_dataset import RegistrationDataset
import sys
import numpy as np

# face recognition = face embedding (Model trained with Siamese Network) + embedding comparison (not trainable, analytic) 



# - pytorch lightning anschauen
# - Look how adaptive threshold behaves when just a few registered people
#  --> as expected: e. g. if 2 stored embeddings and if embeddings far away, then is tolerance also big -> everyone gets accepted!
#   but also according to paper it´s quite bad for less images (-> store e.g. 5000 pseudo embeddings?)
#   why are fixed threshold so low? around 0.3
#   run their implementation according to github and print database?
#   Update matplotlib, numpy and so on afterwards again
#  --> add more embeddings per person (e.g. 12 or so)

# - implement adaptive threshold also for euclidean distance
# - create very simple model to have reference (instead of embeddings, reshape image itself and calculate inner product)
# - create evaluation of model performance (according to paper of adaptive threshold)

# sys.exit()

# Create dataset
# Naming convention for images used for registration with dataloader: name_#.(jpg,png,ppm,...)
reg_dataset = RegistrationDataset("./registered_images", "ppm")

# Create dataloader with batch_size = 1
reg_loader = torch.utils.data.DataLoader(dataset=reg_dataset, batch_size=1, num_workers=0, shuffle=False, sampler=None, collate_fn=None)


embedding_model = faceEmbeddingModel()

# If new dataset: pass dataloader to RegistrationDatabase, then it will rewrite Database
# Otherwise, it trys to return existing database
database = RegistrationDatabase(embedding_model)


# path = './test_registration_images/John_01.ppm'
# img = database.load_and_transform_img(path)
# database.face_registration('John',img)

# path = './test_registration_images/John_02.ppm'
# img = database.load_and_transform_img(path)
# database.face_registration('John',img)

# path = './test_registration_images/John_03.ppm'
# img = database.load_and_transform_img(path)
# database.face_registration('John',img)


database.face_recognition(path='./test_recognition_images/Kofi_04.ppm')

# database.face_deregistration('Kofi')

# print(database.database)