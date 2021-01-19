import numpy as np
import cv2
from sklearn.externals import joblib

from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
#import argparse
#import imutils
import pickle
import time
import cv2
import os
import dlib
import sys
from scipy.spatial import distance as dist
x = 0

# loading the liveness detecting module that was trained in the training python script
print("loading the liveness detector")
model = load_model("./pretrained_model/liveness.model")
le = pickle.loads(open("./pretrained_model/le.pickle", "rb").read())

def calc_hist(img):
    histogram = [0] * 3
    for j in range(3):
        histr = cv2.calcHist([img], [j], None, [256], [0, 256])
        histr *= 255.0 / histr.max()
        histogram[j] = histr
    return np.array(histogram)

modelFile = "./pretrained_model/res10_300x300_ssd_iter_140000.caffemodel"
configFile = "./pretrained_model/deploy.prototxt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
clf = joblib.load('./pretrained_model/face_spoofing_01.pkl')
cap = cv2.VideoCapture(1)
#cap.open(0)
# width = 320
# height = 240
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)

sample_number = 10
count = 0
measures = np.zeros(sample_number, dtype=np.float)

while True:
    ret, img = cap.read()
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0,(300, 300), (104.0, 177.0, 123.0))
    
    net.setInput(blob)
    faces3 = net.forward()

    measures[count%sample_number]=0
    height, width = img.shape[:2]
    for i in range(faces3.shape[2]):
        confidence = faces3[0, 0, i, 2]
        if confidence > 0.5:
            box = faces3[0, 0, i, 3:7] * np.array([width, height, width, height])
            (x, y, x1, y1) = box.astype("int")
            # cv2.rectangle(img, (x, y), (x1, y1), (0, 0, 255), 5)
            roi = img[y:y1, x:x1]

            point = (0,0)
            
#             img_ycrcb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCR_CB)
#             img_luv = cv2.cvtColor(roi, cv2.COLOR_BGR2LUV)
    
#             ycrcb_hist = calc_hist(img_ycrcb)
#             luv_hist = calc_hist(img_luv)
    
#             feature_vector = np.append(ycrcb_hist.ravel(), luv_hist.ravel())
#             feature_vector = feature_vector.reshape(1, len(feature_vector))
    
#             prediction = clf.predict_proba(feature_vector)
#             prob = prediction[0][1]
    
#             measures[count % sample_number] = prob

    # # extract the face ROI and then preproces it in the exact
    # # same manner as our training data
    #         face = frame[startY:endY, startX:endX]
            face = cv2.resize(roi, (32, 32))
            face = face.astype("float") / 255.0
            face = img_to_array(face)
            face = np.expand_dims(face, axis=0)

    #pass the model to determine the liveness
            preds = model.predict(face)[0]
            j = np.argmax(preds)
            label = le.classes_[j]
    
            cv2.rectangle(img, (x, y), (x1, y1), (255, 0, 0), 2)
    
            point = (x, y-5)
    
            print (label, preds[j])
            if preds[j] != 0:
                text = "True"
                if preds[j] >= 0.85:
                    text = "False"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(img=img, text=text, org=point, fontFace=font, fontScale=0.9, color=(0, 0, 255),
                                thickness=2, lineType=cv2.LINE_AA)
                else:
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(img=img, text=text, org=point, fontFace=font, fontScale=0.9,
                                color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
        
    count+=1
    cv2.imshow('img_rgb', img)
    
    key = cv2.waitKey(1)
    if key & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

# ---------------------------------------------------------------------------

# from keras.preprocessing.image import img_to_array
# from keras.models import load_model
# import numpy as np
# #import argparse
# #import imutils
# import pickle
# import time
# import cv2
# import os
# import dlib
# import sys
# from scipy.spatial import distance as dist
# x = 0


# # construct argument parse and parse the arguments
# # ap = argparse.ArgumentParser()
# # ap.add_argument("-m", "--model", type=str, required=True,
# # 	help="path to trained model")
# # ap.add_argument("-l", "--le", type=str, required=True,
# # 	help="path to label encoder")
# # ap.add_argument("-d", "--detector", type=str, required=True,
# # 	help="path to OpenCV's deep learning face detector")
# # ap.add_argument("-c", "--confidence", type=float, default=0.5,
# # 	help="minimum probability to filter weak detections")
# # ap.add_argument("-p", "--shape-predictor", required=True,
# # 	     help="path to facial landmark predictor")
# # args = vars(ap.parse_args())


# # loading face detector from the place where we stored it
# print("loading face detector")
# protoPath = os.path.sep.join(["./pretrained_model", "deploy.prototxt"])
# #Loading the caffe model 
# modelPath = os.path.sep.join(["./pretrained_model",
# 	"res10_300x300_ssd_iter_140000.caffemodel"])
# #reading data from the model.
# net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# # loading the liveness detecting module that was trained in the training python script
# print("loading the liveness detector")
# model = load_model("./pretrained_model/liveness.model")
# le = pickle.loads(open("./pretrained_model/le.pickle", "rb").read())


# #determining the facial points that are plotted by dlib
# FULL_POINTS = list(range(0, 68))  
# FACE_POINTS = list(range(17, 68))  
   
# JAWLINE_POINTS = list(range(0, 17))  
# RIGHT_EYEBROW_POINTS = list(range(17, 22))  
# LEFT_EYEBROW_POINTS = list(range(22, 27))  
# NOSE_POINTS = list(range(27, 36))  
# RIGHT_EYE_POINTS = list(range(36, 42))  
# LEFT_EYE_POINTS = list(range(42, 48))  
# MOUTH_OUTLINE_POINTS = list(range(48, 61))  
# MOUTH_INNER_POINTS = list(range(61, 68))  
   
# EYE_AR_THRESH = 0.30 
# EYE_AR_CONSEC_FRAMES = 2  

# #initializing the parameters
# COUNTER_LEFT = 0  
# TOTAL_LEFT = 0  
   
# COUNTER_RIGHT = 0  
# TOTAL_RIGHT = 0 
# #defining a function for calculating ear and then comparing with the confidence parametrs

# def eye_aspect_ratio(eye):
    
#     A = dist.euclidean(eye[1], eye[5])  
#     B = dist.euclidean(eye[2], eye[4])
#     C = dist.euclidean(eye[0], eye[3])  
#     ear = (A + B) / (2.0 * C)  
#     return ear 

# #loading the predictor for predicting
# #detector = dlib.get_frontal_face_detector()  

# #accessing the shape predictor
# #predictor = dlib.shape_predictor("pretrained_model/shapepredictor")
# #starting the stream


# video_capture = cv2.VideoCapture(1)  
# #looping over frames
# while True:
#     #checkpoint 1
#     ret, frame = video_capture.read()
#     # if ret:
        
#     #     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  
#     #     rects = detector(gray, 0)
#     #     frame = imutils.resize(frame, width=600)
#     #     for rect in rects:
            
#     #         x = rect.left()  
#     #         y = rect.top()  
#     #         x1 = rect.right()  
#     #         y1 = rect.bottom()
#     #         landmarks = np.matrix([[p.x, p.y] for p in predictor(frame, rect).parts()])  
#     #         left_eye = landmarks[LEFT_EYE_POINTS]  
#     #         right_eye = landmarks[RIGHT_EYE_POINTS]  
#     #         left_eye_hull = cv2.convexHull(left_eye)  
#     #         right_eye_hull = cv2.convexHull(right_eye)  
#     #         ear_left = eye_aspect_ratio(left_eye)  
#     #         ear_right = eye_aspect_ratio(right_eye)
		
#     #         #calculating blink wheneer the ear value drops down below the threshold
	
#     #         if ear_left < EYE_AR_THRESH:
                
#     #             COUNTER_LEFT += 1
            
#     #         else:
                
                
#     #             if COUNTER_LEFT >= EYE_AR_CONSEC_FRAMES:
                    
                    
#     #                 TOTAL_LEFT += 1  
#     #                 print("Left eye winked") 
                
#     #                 COUNTER_LEFT = 0
#     #         if ear_right < EYE_AR_THRESH:  
                
                
#     #             COUNTER_RIGHT += 1  

#     #         else:
                
#     #             if COUNTER_RIGHT >= EYE_AR_CONSEC_FRAMES: 
                    
                    
#     #                 TOTAL_RIGHT += 1  
#     #                 print("Right eye winked")  
#     #                 COUNTER_RIGHT = 0


#     #         x = TOTAL_LEFT + TOTAL_RIGHT

#     (h, w) = frame.shape[:2]
#     temp = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
# 		(300, 300), (104.0, 177.0, 123.0))
#     net.setInput(temp)
#     detections = net.forward()
#     for i in range(0, detections.shape[2]):
        
        
#         confidence = detections[0, 0, i, 2]
            
#           #staisfying the union need of veryfying through ROI and blink detection.  
#         if confidence > 0.5 and x>10:
            
            
             
#             #detect a bounding box
# 	    #take dimensions
#             box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
# 	    #get the dimensions
#             (startX, startY, endX, endY) = box.astype("int")

			
#             startX = max(0, startX)
#             startY = max(0, startY)
#             endX = min(w, endX)
#             endY = min(h, endY)

# 	# extract the face ROI and then preproces it in the exact
# 	# same manner as our training data
#             face = frame[startY:endY, startX:endX]
#             face = cv2.resize(face, (32, 32))
#             face = face.astype("float") / 255.0
#             face = img_to_array(face)
#             face = np.expand_dims(face, axis=0)

# 	#pass the model to determine the liveness
#             preds = model.predict(face)[0]
#             j = np.argmax(preds)
#             label = le.classes_[j]

# 		# tag with the label
# 		#tag with the bounding box
#             label = "{}: {:.4f}".format(label, preds[j])
#             cv2.putText(frame, label, (startX, startY - 10),
# 				 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
#             cv2.rectangle(frame, (startX, startY), (endX, endY),
# 				  (0, 0, 255), 2)


#     #showing the frames and waiting for the key to be pressed
#     cv2.imshow("Frame", frame)
#     key = cv2.waitKey(1) & 0xFF
#     if key == ord("q"):
#         break

    
# cv2.destroyAllWindows()
# #vs.stop()
