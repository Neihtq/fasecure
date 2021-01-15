import sys
import os
import numpy as np
import torch
import torch.nn.functional as F

from os.path import join, dirname, abspath

from models.FaceNet import get_model
from models.antiSpoofingModel import AntiSpoofingModel
from models.RefFaceEmbeddingModel import RefFaceEmbeddingModel
from registration_database.RegistrationDatabase import RegistrationDatabase
from utils.prep import load_and_transform_img
from evaluation.PipelineEvaluation import PipelineEvaluation

# to Cao, Simon and Thien: model.eval()!

# todo
# - mit neuen detection und alignment croppen und in ordner packen
#

from deepface import DeepFace
# -------------------


face_detection_model = DeepFace
absolute_dir = dirname(abspath(__file__))

# subset size wieder zu 10 ändern

def evaluate_pipeline():
    anti_spoofing_model = AntiSpoofingModel()
    ref_face_embedding_model = RefFaceEmbeddingModel()
    face_embedding_model = get_model().eval()
    dataset_path = join(absolute_dir, "data", "fake")
    eval_log_path = join(absolute_dir, "evaluation", "evaluation_results", "spoof_normal_model_with_fixed_threshold_")

    if not os.path.exists(join(absolute_dir, "evaluation", "evaluation_results")):
        os.makedirs(join(absolute_dir, "evaluation", "evaluation_results"))

    # loop over different fixed thresholds to find the one resulting in the highest accuracy
    thresholds = [98.5]
    # best threshold 98.5 (compare_num 530: acc: 0.6509)
    for fixed_threshold in thresholds:
        eval_log_path_fix = eval_log_path + str(fixed_threshold) + ".txt"
        registration_database = RegistrationDatabase(fixed_threshold=fixed_threshold)
        pipeline_evaluation = PipelineEvaluation(dataset_path, eval_log_path_fix, anti_spoofing_model,
                                            face_embedding_model, registration_database, face_detection_model)
        pipeline_evaluation.run()
        pipeline_evaluation.plot_results()
        break


if __name__ == '__main__':
    evaluate_pipeline()
