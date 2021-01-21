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



def calc_hist(img):
    histogram = [0] * 3
    for j in range(3):
        histr = cv2.calcHist([img], [j], None, [256], [0, 256])
        histr *= 255.0 / histr.max()
        histogram[j] = histr
    return np.array(histogram)


# Define model number:
# 0 (Default): Face spoofing model from:
# 1: Face spoofing model from:
def face_spoofing_live(model_number=0):

    # loading the liveness detecting module that was trained in the training python script
    print("loading the liveness detector")
    model = load_model("pretrained_model/liveness.model")
    le = pickle.loads(open("pretrained_model/le.pickle", "rb").read())



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

if __name__ == '__main__':
    face_spoofing_live(model_number=0)