from os.path import join, dirname, abspath

# BACKEND REST API
URL = "http://127.0.0.1:5000/"
VERIFY_ENDPOINT = URL + 'verify'
REGISTER_ENDPOINT = URL + 'register'
WIPE_ENDPOINT = URL + 'wipe'
LIST_ALL_ENDPOINT = URL + 'listAll'

# DIRECTORIES
root = join(dirname(abspath(__file__)), '..')

FACE_DETECTION_PROTOTXT = join(root, "face_detection", "model", "deploy.prototxt")
FACE_DETECTION_MODEL = join(root, "face_detection", "model", "res10_300x300_ssd_iter_140000.caffemodel")
SHAPE_PREDICTOR = join(root,'face_detection', 'model', 'shape_predictor_5_face_landmarks.dat')
LOGO = join(root, '..', 'images', 'logo.png')

DETECTION_THRESHOLD = 0.5

# strings
TITLE = 'Fasecure'


# messages: stdout & std errs
ACCESS_DENIED = "User not recognized - Access Denied!"
ACCESS_GRANTED = "User recognized Access Granted for:"
BACKEND_UNREACHABLE = "Could not reach backend."
DB_ACCESS_DENIED = "Password incorrect - Access to data base denied"
DB_ACCESS_GRANTED = "Password correct - Access to database granted"
SUCCESS = 'Success!'
FAIL = 'Process failed!'
NO_FACE = "No Face detected. Please try again!"
FACE_ALIGNMENT_ERROR = "Error during Face Alignment. Please try again!"
TRY_AGAIN = 'An error has occurred. Please try again!'
KEEP_IN_FRAME = 'Out of frame'#
