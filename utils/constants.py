import os
import pathlib

abs_path = os.path.dirname(os.path.abspath(__file__))
root = os.path.join(abs_path, '..')

RESULTS_DIR = os.path.join(root, 'results')
MODEL_DIR = os.path.join(RESULTS_DIR, 'models')
PRETRAINED_MODEL_DIR = os.path.join(root, 'pretrained_model')
CHECKPOINTS_DIR = os.path.join(root, 'checkpoints', 'last_checkpoint')

LFW_ALIGNED_DIR = os.path.join(root, 'data', 'images', 'lfw_aligned')

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PRETRAINED_MODEL_DIR, exist_ok=True)

PRETRAINED_URL = 'https://github.com/khrlimam/facenet/releases/download/acc-0.92135/model921-af60fb4f.pth'