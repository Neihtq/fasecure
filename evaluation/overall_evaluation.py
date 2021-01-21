
import sys
import os
import numpy as np
import torch
import torch.nn.functional as F

from os.path import join

from models.FaceNet import get_model
from models.RefFaceEmbeddingModel import RefFaceEmbeddingModel
from registration_database.RegistrationDatabase import RegistrationDatabase
from evaluation.PipelineEvaluation import PipelineEvaluation



def evaluate_pipeline(absolute_dir):

    dataset_path = join(absolute_dir, "data", "lfw_crop")
    eval_log_path = join(absolute_dir, "evaluation", "results", "ref_model_")
    ref_face_embedding_model = RefFaceEmbeddingModel(dataset_path)
    face_embedding_model = get_model().eval()

    if not os.path.exists(join(absolute_dir, "evaluation", "results")):
        os.makedirs(join(absolute_dir, "evaluation", "results"))

    # loop over different fixed thresholds to find the one resulting in the highest accuracy
    thresholds = [98.5]
    # best threshold 98.5 (compare_num 530: acc: 0.6509)
    for fixed_threshold in thresholds:
        eval_log_path_fix = eval_log_path + str(fixed_threshold) + ".txt"
        registration_database = RegistrationDatabase(fixed_initial_threshold=fixed_threshold)
        pipeline_evaluation = PipelineEvaluation(dataset_path, eval_log_path_fix, ref_face_embedding_model, registration_database)
        pipeline_evaluation.run()
        pipeline_evaluation.plot_results()