from os.path import join, dirname, abspath

# BACKEND REST API
URL = "http://127.0.0.1:5000/"
VERIFY_ENDPOINT = URL + 'verify'
REGISTER_ENDPOINT = URL + 'register'
WIPE_ENDPOINT = URL + 'wipe'
LIST_ALL_ENDPOINT = URL + 'listAll'

'''Openshift Endpoints
VERIFY_ENDPOINT = "http://facesecure-verify-default.dte-ocp44-kngf35-915b3b336cabec458a7c7ec2aa7c625f-0000.us-east.containers.appdomain.cloud/verify"
REGISTER_ENDPOINT = "http://facesecure-register-default.dte-ocp44-kngf35-915b3b336cabec458a7c7ec2aa7c625f-0000.us-east.containers.appdomain.cloud/register"
WIPE_ENDPOINT = "http://facesecure-wipe-default.dte-ocp44-kngf35-915b3b336cabec458a7c7ec2aa7c625f-0000.us-east.containers.appdomain.cloud/wipe"
LIST_ALL_ENDPOINT = "http://facesecure-listall-default.dte-ocp44-kngf35-915b3b336cabec458a7c7ec2aa7c625f-0000.us-east.containers.appdomain.cloud/listAll"
'''

# DIRECTORIES
root = join(dirname(abspath(__file__)), '..')
FACE_DETECTION_PROTOTXT = join(root, "face_detection", "model", "deploy.prototxt")
FACE_DETECTION_MODEL = join(root, "face_detection", "model", "res10_300x300_ssd_iter_140000.caffemodel")
SHAPE_PREDICTOR = join(root,'face_detection', 'model', 'shape_predictor_5_face_landmarks.dat')
LOGO = join(root, '..', 'images', 'logo.png')

# Parameters
DETECTION_THRESHOLD = 0.5

# strings
TITLE = 'Fasecure'

# stdout & std errs
ACCESS_DENIED = "User not recognized - Access Denied!"
ACCESS_GRANTED = "User recognized Access Granted for: "
BACKEND_UNREACHABLE = "Could not reach backend."
DB_ACCESS_DENIED = "Password incorrect - Access to data base denied"
DB_ACCESS_GRANTED = "Password correct - Access to database granted"
SUCCESS = 'Success!'
FAIL = 'Process failed!'
NO_FACE = "No Face detected. Please try again!"
FACE_ALIGNMENT_ERROR = "Error during Face Alignment. Please try again!"
TRY_AGAIN = 'An error has occurred. Please try again!'
KEEP_IN_FRAME = 'Out of frame'#
WAIT_CLOSE_APP = "Prepare closing application"
CLOSE_APP = "Closing application..."