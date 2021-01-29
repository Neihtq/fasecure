from face_recognition.evaluation.evaluations import evaluate_pipeline
from face_recognition.utils.constants import LFW_CROP_DIR, EVAL_RESULTS_DIR, OVERALL_EVAL_LFW_DIR, OVERALL_EVAL_RESULTS_DIR
import os

if __name__ == '__main__':

    evaluate_pipeline(LFW_CROP_DIR, EVAL_RESULTS_DIR)

def evaluate():
    evaluate_pipeline(OVERALL_EVAL_LFW_DIR, OVERALL_EVAL_RESULTS_DIR)