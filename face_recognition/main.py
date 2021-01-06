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
# - problem gefixed mit zu wenig Personen am Anfang

# todo
# - compare to fixed_threshold = 0 (normal case? and print also the first cases, as there very bad?)
# - implement code, that if cuda available, if moves everything on the gpu
# - evalation normally on which dataset? new one?
# - run with that threshold reference model (or with other threshold?)
# - others should train embedding model with same augmentation techniques as I use for registration!!
# - compare accuracy with paper



# from deepface import DeepFace


#face_detection_model = DeepFace
ref_face_embedding_model = refFaceEmbeddingModel()
face_embedding_model = faceEmbeddingModel().eval()
dataset_path = "./evaluation_data/lfw_crop"
eval_log_path = "./evaluation_results/normal_model_with_fixed_threshold_"

# loop over different fixed thresholds to find the one resulting in the highest accuracy
thresholds = [0, 98.5]
# best threshold 98.5 (compare_num 530: acc: 0.6509)
for fixed_threshold in thresholds:
    eval_log_path_fix = eval_log_path + str(fixed_threshold) + ".txt"
    registration_database = RegistrationDatabase(fixed_threshold=fixed_threshold)
    pipeline_evaluation = PipelineEvaluation(dataset_path, eval_log_path_fix,
                                          face_embedding_model, registration_database)
    #pipeline_evaluation.run()
    #pipeline_evaluation.plot_results()
    pipeline_evaluation.compare_evaluations()
    sys.exit()