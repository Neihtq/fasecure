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

# - when final pipeline is ready, then compare all settings with performance metrics (with augmentation, without augmentation,...)

# - transformation of input images has to be handled by faceEmbeddingModel and not by database (in face registration and face recognition)

# imports
import torch.nn.functional as F
import torch
from torchvision import transforms
from faceEmbeddingModel import faceEmbeddingModel
# from prep import load_and_transform_img, show_tensor_img
from reg_database import RegistrationDatabase
from reg_dataset import RegistrationDataset
from prep import load_and_transform_img
import sys
import numpy as np

# face recognition = face embedding (Model trained with Siamese Network) + embedding comparison (not trainable, analytic) 

# to Cao, Simon and Thien: model.eval()!!!!!!!!!!!!!!!!!!!!!!!!!!
# main results:
# - euclidean distance or inner product no difference
# - data augmentation so far no difference
# - works bad for less persons
# - works bad for one shot?

# What I have done since last time:
# - implemented adaptive threshold
# - added three images per person and per image 4 data augmentations -> 12 images per registered person
# - implemented similarity calculation with inner product and euclidean distance (both same results)


# - pytorch lightning anschauen
# - Look how adaptive threshold behaves when just a few registered people
#  --> as expected: e. g. if 2 stored embeddings and if embeddings far away, then is tolerance also big -> everyone gets accepted!
#   but also according to paper it´s quite bad for less images (-> store e.g. 5000 pseudo embeddings?)
#   why are fixed threshold so low? around 0.3
#   run their implementation according to github and print database?
#   Update matplotlib, numpy and so on afterwards again
#  --> add more embeddings per person (e.g. 12 or so)

# - create very simple model to have reference (instead of embeddings, reshape image itself and calculate inner product)
# - create evaluation of model performance (according to paper of adaptive threshold)

# Create dataset
# Naming convention for images used for registration with dataloader: name_#.(jpg,png,ppm,...)
###reg_dataset = RegistrationDataset("./registered_images", "ppm")

# Create dataloader with batch_size = 1
###reg_loader = torch.utils.data.DataLoader(dataset=reg_dataset, batch_size=1, num_workers=0, shuffle=False, sampler=None, collate_fn=None)


embedding_model = faceEmbeddingModel().eval()

# mode='euclidean_distance'
database = RegistrationDatabase()

# ----------------------------------------------------------------------------------
def register_people():

    paths = []
    paths.append('./test_registration_images/Aaron_01.ppm')
    paths.append('./test_registration_images/Aaron_02.ppm')
    paths.append('./test_registration_images/Aaron_03.ppm')
    paths.append('./test_registration_images/Abdoulaye_01.ppm')
    paths.append('./test_registration_images/Abdoulaye_02.ppm')
    paths.append('./test_registration_images/Abdoulaye_03.ppm')
    paths.append('./test_registration_images/George_01.ppm')
    paths.append('./test_registration_images/George_02.ppm')
    paths.append('./test_registration_images/George_03.ppm')
    paths.append('./test_registration_images/Hugo_01.ppm')
    paths.append('./test_registration_images/Hugo_02.ppm')
    paths.append('./test_registration_images/Hugo_03.ppm')
    paths.append('./test_registration_images/Ian_01.ppm')
    paths.append('./test_registration_images/Ian_02.ppm')
    paths.append('./test_registration_images/Ian_03.ppm')
    paths.append('./test_registration_images/Jennifer_01.ppm')
    paths.append('./test_registration_images/Jennifer_02.ppm')
    paths.append('./test_registration_images/Jennifer_03.ppm')
    paths.append('./test_registration_images/Kofi_01.ppm')
    paths.append('./test_registration_images/Kofi_02.ppm')
    paths.append('./test_registration_images/Kofi_03.ppm')
    paths.append('./test_registration_images/Lleyton_01.ppm')
    paths.append('./test_registration_images/Lleyton_02.ppm')
    paths.append('./test_registration_images/Lleyton_03.ppm')
    paths.append('./test_registration_images/Vladimir_01.ppm')
    paths.append('./test_registration_images/Vladimir_02.ppm')
    paths.append('./test_registration_images/Vladimir_03.ppm')
    paths.append('./test_registration_images/Yashwant_01.ppm')
    paths.append('./test_registration_images/Yashwant_02.ppm')
    paths.append('./test_registration_images/Yashwant_03.ppm')


    names = []
    names.append('Aaron')
    names.append('Aaron')
    names.append('Aaron')
    names.append('Abdoulaye')
    names.append('Abdoulaye')
    names.append('Abdoulaye')
    names.append('George')
    names.append('George')
    names.append('George')
    names.append('Hugo')
    names.append('Hugo')
    names.append('Hugo')
    names.append('Ian')
    names.append('Ian')
    names.append('Ian')
    names.append('Jennifer')
    names.append('Jennifer')
    names.append('Jennifer')
    names.append('Kofi')
    names.append('Kofi')
    names.append('Kofi')
    names.append('Lleyton')
    names.append('Lleyton')
    names.append('Lleyton')
    names.append('Vladimir')
    names.append('Vladimir')
    names.append('Vladimir')
    names.append('Yashwant')
    names.append('Yashwant')
    names.append('Yashwant')


    for i in range(len(names)):
        # data augmentation
        reg_img = load_and_transform_img(paths[i])
        img_embedding_tensor = embedding_model(reg_img)
        database.face_registration(names[i],img_embedding_tensor)

register_people()
# ----------------------------------------------------------------------------------

print(database.database)

# Face Recognition with data augmentation
# path = './test_recognition_images/Hugo_04.ppm'
# img = load_and_transform_img(path)
# img_embedding_tensor = embedding_model(img)
# closest_label, check = database.face_recognition(img_embedding_tensor)
# print("Closest person: ", closest_label, " --- ", check)

# database.face_deregistration('Aaron')

# print(database.database)


# input: 128 dim embedding as tensor (convert it internally to numpy array)
#               - registration: embedding + name
#               - deregistration: name
#               - recognition: embedding
# ---------------------------------------------------------------------------
# functions:    - registration
#               - deregistration
#               - recognition
# ---------------------------------------------------------------------------
# output:       - registration: "registered successfully"
#               - deregistration: "deregistered successfully"
#               - recognition: name + access/intruder