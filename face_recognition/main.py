# imports
import torch.nn.functional as F
import torch
from faceEmbeddingModel import faceEmbeddingModel
from refFaceEmbeddingModel import refFaceEmbeddingModel
from reg_database import RegistrationDatabase
from prep import load_and_transform_img
from pipeline_evaluation import PipelineEvaluation
import sys
import numpy as np


# to Cao, Simon and Thien: model.eval()!
# main results:
# - euclidean distance or inner product no difference
# - data augmentation so far no difference (-> look at images)
# - works bad for less persons (-> paper used fixed threshold as initial value of adaptive threshold procedure)
# - works a bit better for few-shot learning instead of one-shot learning (currently 3 images per person)

# Other similarities:
# - Manhatten Distance
# - Cosine Similarity

# main results 2:
# - Created evaluation for whole pipeline
# - only use 1/10 of lfw for evaluation so far
# - log the evaluation and plot results
# - can choose wheter to use detection model or directly cropped images
#   (only in evaluation, if face detected. Otherwise ignored)
# - Implemented reference face embedding model which we have to beat (lower bound)

# embedding_model = faceEmbeddingModel().eval()

# mode='euclidean_distance'
# database = RegistrationDatabase()


# database.clean_database()
# register_people()

#from deepface import DeepFace

dataset_path = "./evaluation_data/lfw_crop"
eval_log_path = "./evaluation_results/ref_model_cropped.txt"
#face_detection_model = DeepFace
ref_face_embedding_model = refFaceEmbeddingModel()
face_embedding_model = faceEmbeddingModel().eval()
registration_database = RegistrationDatabase()

pipeline_evaluation = PipelineEvaluation(dataset_path, eval_log_path,
                                          ref_face_embedding_model, registration_database)

pipeline_evaluation.run()
#pipeline_evaluation.plot_results()

# Face Recognition with data augmentation
# path = './test_recognition_images/Vladimir_04.ppm'
# img_1, img_2, img_3, img_4, img_5, img_6, img_7 = load_and_transform_img(path)
# img_embedding_tensor = embedding_model(img_1)
# closest_label, check = database.face_recognition(img_embedding_tensor)
# if check == 'Access':
#    print("--- Access --- Recognized person: ", closest_label)
# elif check == 'Decline':
#     print("--- Decline ---")

# database.face_deregistration('Aaron')

# print(database.database)