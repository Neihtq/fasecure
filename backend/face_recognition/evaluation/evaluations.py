
import os
from os.path import join
import numpy as np

from face_recognition.models.FaceNet import get_model_old, get_model
from face_recognition.models.RefFaceEmbeddingModel import RefFaceEmbeddingModel
from face_recognition.evaluation.EvaluationPipeline import EvaluationPipeline
from face_recognition.database.RegistrationDatabase import RegistrationDatabase
from face_recognition.models.FaceNet import load_pretrained

# Evaluate the face embedding model and the overall pipeline
def evaluate_overall_pipeline(dataset_path, eval_log_path, pretrained_face_embedding_model_path):

    # Load all three face embedding models
    own_face_embedding_model = get_model().eval()
    pretrained_face_embedding_model = load_pretrained(pretrained_face_embedding_model_path).eval() 
    ref_face_embedding_model = RefFaceEmbeddingModel(dataset_path)
    
    eval_log_path = eval_log_path + "_evaluations_"


    # perform overall evaluation for own trained face embedding model
    eval_log_path_fix = eval_log_path + "own_trained" + ".txt"
    registration_database = RegistrationDatabase(fixed_initial_threshold=98.9)
    pipeline_evaluation = EvaluationPipeline(dataset_path, eval_log_path_fix, own_face_embedding_model, registration_database, 
                                            aug_mean = [0.485, 0.456, 0.406], aug_std = [0.229, 0.224, 0.225])
    pipeline_evaluation.run()    

    # perform overall evaluation for pretrained face embedding model
    eval_log_path_fix = eval_log_path + "pretrained" + ".txt"
    registration_database = RegistrationDatabase(fixed_initial_threshold=35)
    pipeline_evaluation = EvaluationPipeline(dataset_path, eval_log_path_fix, pretrained_face_embedding_model, registration_database)
    pipeline_evaluation.run()   

    # perform overall evaluation for reference face embedding model
    eval_log_path_fix = eval_log_path + "reference" + ".txt"
    registration_database = RegistrationDatabase(fixed_initial_threshold=0)
    pipeline_evaluation = EvaluationPipeline(dataset_path, eval_log_path_fix, ref_face_embedding_model, registration_database,
                                            aug_mean = [0.485, 0.456, 0.406], aug_std = [0.229, 0.224, 0.225])
    pipeline_evaluation.run()  

    # plot results
    pipeline_evaluation.compare_evaluations()


def evaluate_pipeline(dataset_path, eval_log_path, face_embedding_model_path=None):
    '''Overall evaluation of complete pipeline'''

    # Load pretrained model, if no path specified
    if face_embedding_model_path is None:
        face_embedding_model = get_model().eval()
    else:
        # --- Load model from specified path ---
        face_embedding_model = load_pretrained(face_embedding_model_path).eval()
    
    # Load also reference model (PCA)
    #ref_face_embedding_model = RefFaceEmbeddingModel(dataset_path)


    eval_log_path = eval_log_path + "_evaluations_"

    # loop over different fixed thresholds to find the one resulting in the highest accuracy
    thresholds = [98.9]
    #thresholds = list(np.arange(98.5,99,0.1))
    #thresholds = list(range(96,100))

    for fixed_threshold in thresholds:
        print("current threshold: ", fixed_threshold)
        eval_log_path_fix = eval_log_path + str(fixed_threshold) + ".txt"
        registration_database = RegistrationDatabase(fixed_initial_threshold=fixed_threshold)
        pipeline_evaluation = EvaluationPipeline(dataset_path, eval_log_path_fix, face_embedding_model, registration_database, 
                                                    aug_mean = [0.485, 0.456, 0.406], aug_std = [0.229, 0.224, 0.225])
        pipeline_evaluation.run()
        #pipeline_evaluation.plot_results()
        #pipeline_evaluation.compare_evaluations()
        # os.sys.exit()

if __name__ == '__main__':
    evaluate_results(absolute_dir)