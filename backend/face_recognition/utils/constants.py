import os
import pathlib

# Directory
abs_path = os.path.dirname(os.path.abspath(__file__))
<<<<<<< HEAD
root = os.path.join(abs_path, "../")
=======
root = os.path.join(abs_path, "..")
>>>>>>> overall_evaluations_tobias

RESULTS_DIR = os.path.join(root, 'results')
MODEL_DIR = os.path.join(RESULTS_DIR, 'models')
PRETRAINED_MODEL_DIR = os.path.join(root, 'pretrained_model')
CHECKPOINTS_DIR = os.path.join(root, 'checkpoints', 'last_checkpoint')
TRAINED_WEIGHTS_DIR = os.path.join(root, 'results', 'models', 'presentation_model.pth')


DATA_DIR = os.path.join(root, "data", "images")
LFW_ALIGNED_DIR = os.path.join(root, 'data', 'images', 'lfw_aligned')
LFW_DIR = os.path.join(DATA_DIR,"lfw")
LFW_CROP_DIR = os.path.join(DATA_DIR, "lfw_crop")

RESULTS_DIR = os.path.join(root, "results")
EVAL_RESULTS_DIR = os.path.join(RESULTS_DIR, "evaluation", "refmodel")

OVERALL_EVAL_LFW_DIR_ALL = os.path.join(DATA_DIR, "lfw_overall_eval_all")
OVERALL_EVAL_LFW_DIR_MALE = os.path.join(DATA_DIR, "lfw_overall_eval_male")
OVERALL_EVAL_LFW_DIR_FEMALE = os.path.join(DATA_DIR, "lfw_overall_eval_female")
OVERALL_EVAL_RESULTS_DIR = os.path.join(RESULTS_DIR, "evaluation", "overall_evaluation")

DATABASE_DIR = os.path.join(root, "database", "reg_database", "database.pkl")

pathlib.Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
pathlib.Path(DATABASE_DIR).parent.mkdir(parents=True, exist_ok=True)
pathlib.Path(EVAL_RESULTS_DIR).parent.mkdir(parents=True, exist_ok=True)
pathlib.Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)
pathlib.Path(PRETRAINED_MODEL_DIR).mkdir(parents=True, exist_ok=True)


# Stdoutputs & stderrs
DATABASE_EXIST = "Database already exists. Pickle file will be loaded..."
CREATE_NEW_DATABASE = "No database availabe. Empty database will be created..."
WIPE_DATABASE = "database.pkl exists and will be wiped..."
UNDEFINED_THRESHOLD = "----------- Fixed threshold not defined so far! --------------"
USER_NOT_REGISTERED = "Specified name not in database registered. User can not be deregistered!"
CANNOT_WIPE_DATBABASE = "No database.pkl file exists. Hence, it cannot be cleaned..."
UNKNOWN_PERSON = "Person is unkown"


# Strings
EUCLIDEAN_DISTANCE = "euclidean_distance"
INNER_PRODUCT = "inner_product"

# URLs
PRETRAINED_URL = 'https://github.com/khrlimam/facenet/releases/download/acc-0.92135/model921-af60fb4f.pth'