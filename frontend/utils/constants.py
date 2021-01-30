from os.path import join, dirname, abspath

# BACKEND REST API
URL = "http://127.0.0.1:5000/"
VERIFY_ENDPOINT = URL + 'verify'
REGISTER_ENDPOINT = URL + 'register'
WIPE_ENDPOINT = URL + 'wipe'

# DIRECTORIES
root = join(dirname(abspath(__file__)), '..')

FACE_DETECTION_PROTOTXT = join(root, "face_detection", "model", "deploy.prototxt")
FACE_DETECTION_MODEL = join(root, "face_detection", "model", "res10_300x300_ssd_iter_140000.caffemodel")
SHAPE_PREDICTOR = join(root,'face_detection', 'model', 'shape_predictor_5_face_landmarks.dat')

DETECTION_THRESHOLD = 0.5
