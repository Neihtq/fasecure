
import os
from os.path import join

from models.FaceNet import get_model
from registration_database.RegistrationDatabase import RegistrationDatabase
from evaluation.PipelineEvaluation import PipelineEvaluation



# Evaluate the face embedding model and the overall pipeline
def evaluate_results(absolute_dir, face_embedding_eval_data_path=None, overall_eval_data_path=None, face_embedding_model_path=None, fixed_initial_registration_threshold=98.5):

    # FACE EMBEDDING EVALUATION


    # OVERALL EVALUATION

    # Load pretrained model, if no path specified
    if face_embedding_model_path is None:
        face_embedding_model = get_model().eval()
    else:
        # --- Load model from specified path ---
        pass


    # Load evaluation data for overall evaluation from default path, if no path specified
    if overall_eval_data_path is None:
        dataset_path = join(absolute_dir, "data", "lfw_crop")
    else:
        dataset_path = overall_eval_data_path
    
    # Define the evaluation log path for the overall evaluation with the default path
    eval_log_path = join(absolute_dir, "evaluation", "results", "overall_evaluation_logs_")
    if not os.path.exists(join(absolute_dir, "evaluation", "results")):
        os.makedirs(join(absolute_dir, "evaluation", "results"))

   
    # Run the overall evaluation
    eval_log_path_fix = eval_log_path + str(fixed_initial_registration_threshold) + ".txt"
    registration_database = RegistrationDatabase(fixed_initial_threshold=fixed_initial_registration_threshold)
    pipeline_evaluation = PipelineEvaluation(dataset_path, eval_log_path_fix, face_embedding_model, registration_database)
    pipeline_evaluation.run()
    pipeline_evaluation.plot_results()
        


if __name__ == '__main__':
    evaluate_results(absolute_dir)