
import os
from os.path import join

from face_recognition.models.FaceNet import get_model
from face_recognition.models.RefFaceEmbeddingModel import RefFaceEmbeddingModel
from face_recognition.evaluation.EvaluationPipeline import EvaluationPipeline
from face_recognition.database.RegistrationDatabase import RegistrationDatabase


# Evaluate the face embedding model and the overall pipeline
def evaluate_results(absolute_dir, face_embedding_model_path=None, fixed_initial_registration_threshold=98.5):
    # FACE EMBEDDING EVALUATION

    pass


    # # OVERALL EVALUATION

    # # Load pretrained model, if no path specified
    # if face_embedding_model_path is None:
    #     face_embedding_model = get_model().eval()
    # else:
    #     # --- Load model from specified path ---
    #     pass

    # # Load evaluation data for overall evaluation from default path
    # overall_eval_dataset_path = join(absolute_dir, "data", "lfw_overall_eval_all")

    # # Define the evaluation log path for the overall evaluation with the default path
    # eval_log_path = join(absolute_dir, "evaluation", "results", "overall_evaluation_logs_")
    # if not os.path.exists(join(absolute_dir, "evaluation", "results")):
    #     os.makedirs(join(absolute_dir, "evaluation", "results"))

    # # Run the overall evaluation
    # eval_log_path_fix = eval_log_path + str(fixed_initial_registration_threshold) + ".txt"
    # registration_database = RegistrationDatabase(fixed_initial_threshold=fixed_initial_registration_threshold)
    # pipeline_evaluation = EvaluationPipeline(overall_eval_dataset_path, eval_log_path_fix, face_embedding_model, registration_database)
    # pipeline_evaluation.run()
    # pipeline_evaluation.plot_results()

def evaluate_pipeline(dataset_path, eval_log_path, face_embedding_model_path=None):
    '''Overall evaluation of complete pipeline'''

    # Load pretrained model, if no path specified
    if face_embedding_model_path is None:
        face_embedding_model = get_model().eval()
    else:
        # --- Load model from specified path ---
        pass
    
    # Load also reference model (PCA)
    ref_face_embedding_model = RefFaceEmbeddingModel(dataset_path)


    eval_log_path = eval_log_path + "final_evaluations_"

    # loop over different fixed thresholds to find the one resulting in the highest accuracy
    thresholds = [98.5]
    # best threshold 98.5 (compare_num 530: acc: 0.6509)
    for fixed_threshold in thresholds:
        eval_log_path_fix = eval_log_path + str(fixed_threshold) + ".txt"
        registration_database = RegistrationDatabase(fixed_initial_threshold=fixed_threshold)
        pipeline_evaluation = EvaluationPipeline(dataset_path, eval_log_path_fix, face_embedding_model, registration_database)
        pipeline_evaluation.run()
        pipeline_evaluation.plot_results()

if __name__ == '__main__':
    evaluate_results(absolute_dir)