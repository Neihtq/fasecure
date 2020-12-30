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
# - only use 1/10 of lfw for evaluation so far and only if at least 6 images per folder 
#       (->540 images for overal evaluation)
#       -> face recognition is more often tested with already registered user, so can test one-shot learning more often

# - log the evaluation and plot results
# - can choose wheter to use detection model or directly cropped images
#   (only in evaluation, if face detected. Otherwise ignored)
# - Implemented reference face embedding model which we have to beat (lower bound)

# todo
# - have to find optimal fixed_threshold as lower bound (and initial threshold) to fix problem with too less registered people
# - run with that threshold reference model (or with other threshold?)
# - pickle file gets very quick very large

# from deepface import DeepFace
# -------------------


#face_detection_model = DeepFace
ref_face_embedding_model = refFaceEmbeddingModel()
face_embedding_model = faceEmbeddingModel().eval()
dataset_path = "./evaluation_data/lfw_crop"
eval_log_path = "./evaluation_results/normal_model_with_fixed_threshold_"

# loop over different fixed thresholds to find the one resulting in the highest accuracy
for fixed_threshold in range(95,100,1):
    eval_log_path_fix = eval_log_path + str(fixed_threshold) + ".txt"
    registration_database = RegistrationDatabase(fixed_threshold=fixed_threshold)
    pipeline_evaluation = PipelineEvaluation(dataset_path, eval_log_path_fix,
                                          face_embedding_model, registration_database)
    pipeline_evaluation.run()
    #pipeline_evaluation.plot_results()
