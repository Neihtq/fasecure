import os
import pathlib

# Directory
abs_path = os.path.dirname(os.path.abspath(__file__))
root = os.path.join(abs_path, "..")
DATA_DIR = os.path.join(root, "images")
LFW_DIR = os.path.join(DATA_DIR,"lfw")
RESULTS_DIR = os.path.join(root, "results")
LFW_CROP_DIR = os.path.join(DATA_DIR, "lfw_crop")
EVAL_RESULTS_DIR = os.path.join(RESULTS_DIR, "evaluation", "refmodel")
DATABASE_DIR = os.path.join(root, "database", "reg_database", "database.pkl")

pathlib.Path(DATABASE_DIR).parent.mkdir(parents=True, exist_ok=True)
pathlib.Path(EVAL_RESULTS_DIR).parent.mkdir(parents=True, exist_ok=True)

if not os.path.exists(RESULTS_DIR):
    os.mkdir(RESULTS_DIR)

# Stdoutputs & stderrs
DATABASE_EXIST = "Database already exists. Pickle file will be loaded..."
CREATE_NEW_DATABASE = "No database availabe. Empty database will be created..."
WIPE_DATABASE = "database.pkl exists and will be wiped..."
UNDEFINED_THRESHOLD = "----------- Fixed threshold not defined so far! --------------"
USER_NOT_REGISTERED = "Specified name not in database registered. User can not be deregistered!"
CANNOT_WIPE_DATBABASE = "No database.pkl file exists. Hence, it cannot be cleaned..."


# Strings
EUCLIDEAN_DISTANCE = "euclidean_distance"
INNER_PRODUCT = "inner_product"
ACCESS = "Access"
DECLINE = "Decline"