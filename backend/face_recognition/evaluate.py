import os
import sys
import argparse

from face_recognition.evaluation.evaluations import evaluate_pipeline, evaluate_overall_pipeline
from face_recognition.utils.constants import LFW_CROP_DIR, EVAL_RESULTS_DIR, OVERALL_EVAL_LFW_DIR_ALL, OVERALL_EVAL_LFW_DIR_MALE, OVERALL_EVAL_LFW_DIR_FEMALE, OVERALL_EVAL_RESULTS_DIR, TRAINED_WEIGHTS_DIR

parser = argparse.ArgumentParser(description='Evaluation of Facesecure')

parser.add_argument('--model-dir', default=TRAINED_WEIGHTS_DIR, type=str,
                    help=f'Path to model (default: {TRAINED_WEIGHTS_DIR})')

parser.add_argument('--eval-data', default=OVERALL_EVAL_LFW_DIR_ALL, type=str,
                    help=f'Path to evaluation data (default: {OVERALL_EVAL_LFW_DIR_ALL})')

args = parser.parse_args()


if __name__ == '__main__':
    model_dir = args.model_dir
    eval_data = args.eval_data
    evaluate_overall_pipeline(eval_data, OVERALL_EVAL_RESULTS_DIR, model_dir)
    print(f"Results are stored in {OVERALL_EVAL_RESULTS_DIR}")
    sys.exit()