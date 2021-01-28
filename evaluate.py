from evaluation.evaluations import evaluate_pipeline

from constants import LFW_CROP_DIR, EVAL_RESULTS_DIR


if __name__ == '__main__':
    evaluate_pipeline(LFW_CROP_DIR, EVAL_RESULTS_DIR)