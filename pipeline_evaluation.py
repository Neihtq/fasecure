
import sys
import os
import numpy as np
import torch
import torch.nn.functional as F

from os.path import join, dirname, abspath

from models.FaceNet import get_model
from models.RefFaceEmbeddingModel import RefFaceEmbeddingModel
from registration_database.RegistrationDatabase import RegistrationDatabase
from utils.prep import load_and_transform_img
from evaluation.PipelineEvaluation import PipelineEvaluation

# to Cao, Simon and Thien: model.eval()!

# todo
# - compare to fixed_threshold = 0 (normal case? and print also the first cases, as there very bad?)
# - implement code, that if cuda available, if moves everything on the gpu
# - evalation normally on which dataset? new one?
# - run with that threshold reference model (or with other threshold?)
# - others should train embedding model with same augmentation techniques as I use for registration!!
# - compare accuracy with paper

# from deepface import DeepFace
# -------------------


#face_detection_model = DeepFace
absolute_dir = dirname(abspath(__file__))

def evaluate_pipeline():
    ref_face_embedding_model = RefFaceEmbeddingModel()
    face_embedding_model = get_model().eval()
    dataset_path = join(absolute_dir, "data", "lfw_crop")
    eval_log_path = join(absolute_dir, "evaluation", "evaluation_results", "normal_model_with_fixed_threshold_")

    if not os.path.exists(join(absolute_dir, "evaluation", "evaluation_results")):
        os.makedirs(join(absolute_dir, "evaluation", "evaluation_results"))

    # loop over different fixed thresholds to find the one resulting in the highest accuracy
    thresholds = [98.5]
    # best threshold 98.5 (compare_num 530: acc: 0.6509)
    for fixed_threshold in thresholds:
        eval_log_path_fix = eval_log_path + str(fixed_threshold) + ".txt"
        registration_database = RegistrationDatabase(fixed_threshold=fixed_threshold)
        pipeline_evaluation = PipelineEvaluation(dataset_path, eval_log_path_fix,
                                            face_embedding_model, registration_database)
        pipeline_evaluation.run()
        pipeline_evaluation.plot_results()
        break


if __name__ == '__main__':
    evaluate_pipeline()
