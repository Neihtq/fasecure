from face_recognition.evaluation.evaluations import evaluate_pipeline, evaluate_overall_pipeline
from face_recognition.utils.constants import LFW_CROP_DIR, EVAL_RESULTS_DIR, OVERALL_EVAL_LFW_DIR_ALL, OVERALL_EVAL_LFW_DIR_MALE, OVERALL_EVAL_LFW_DIR_FEMALE, OVERALL_EVAL_RESULTS_DIR, TRAINED_WEIGHTS_DIR
import os

if __name__ == '__main__':

    face_embedding_model_path=TRAINED_WEIGHTS_DIR
    evaluate_overall_pipeline(OVERALL_EVAL_LFW_DIR_ALL, OVERALL_EVAL_RESULTS_DIR, face_embedding_model_path)

def evaluate():
    # specify path to model, can load his own when changes path in constant
    pretrained_face_embedding_model_path = TRAINED_WEIGHTS_DIR

    #evaluate_pipeline(OVERALL_EVAL_LFW_DIR_ALL, OVERALL_EVAL_RESULTS_DIR)
    evaluate_overall_pipeline(OVERALL_EVAL_LFW_DIR_ALL, OVERALL_EVAL_RESULTS_DIR, pretrained_face_embedding_model_path)