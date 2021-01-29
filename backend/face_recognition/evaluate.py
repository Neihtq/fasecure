from face_recognition.evaluation.evaluations import evaluate_pipeline
from face_recognition.utils.constants import LFW_CROP_DIR, EVAL_RESULTS_DIR, OVERALL_EVAL_LFW_DIR, OVERALL_EVAL_RESULTS_DIR
import os

if __name__ == '__main__':

    evaluate_pipeline(LFW_CROP_DIR, EVAL_RESULTS_DIR)

def evaluate():
    # print("Test---", OVERALL_EVAL_LFW_DIR)
    # print(OVERALL_EVAL_RESULTS_DIR)
    # os.sys.exit()
    evaluate_pipeline(OVERALL_EVAL_LFW_DIR, OVERALL_EVAL_RESULTS_DIR)