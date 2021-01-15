
# Using pip
# pip install opencv-python
# pip install scikit-learn==0.19.1 (0.19.2 funktioniert)

# old pipfile
# numpy = "==1.19.3"

# Code Reference: https://medium.com/visionwizard/face-spoofing-detection-in-python-e46761fe5947

# Description:
# - input: type (W,H,C) W and H doesn´t matter, C=3
# - output: fake or real

import numpy as np
import cv2
from sklearn.externals import joblib
import torch.nn as nn

# Problems: 
# - I cannot open webcam with the code I have
# - not sure if it works with scikit-learn 0.19.2
# todo:
# - build in live webcam procedure from Cao. Adapt init(delete face detection) and forward function(add input and delete face detection) and add text in live video
# - delete my test function and pretrained_models from face detection


class AntiSpoofingModel(nn.Module):
    def __init__(self):
        super(AntiSpoofingModel, self).__init__()
        self.clf = joblib.load('./pretrained_model/face_spoofing.pkl')

        #----------------
        modelFile = "./pretrained_model/res10_300x300_ssd_iter_140000.caffemodel"
        configFile = "./pretrained_model/deploy.prototxt"
        self.net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

    def calc_hist(self, img):
        histogram = [0] * 3
        for j in range(3):
            histr = cv2.calcHist([img], [j], None, [256], [0, 256])
            histr *= 255.0 / histr.max()
            histogram[j] = histr
        return np.array(histogram)

    def forward(self):
        #img_input = cv2.imread('C:/Users/Marco/Desktop/Master/3_semester/practical_course/skript/github_01/IBM/data/fake/celebrity_eins/WIN_20210114_15_59_04_Pro.jpg', cv2.IMREAD_COLOR)
        img_input = cv2.imread('C:/Users/Marco/Desktop/Master/3_semester/practical_course/skript/github_01/IBM/data/lfw/Aaron_Eckhart/Aaron_Eckhart_0001.jpg', cv2.IMREAD_COLOR)
        
        # cv2.imshow('image', x)   

        # k = cv2.waitKey(0) & 0xFF
        # # wait for ESC key to exit
        # if k == 27:
        #     cv2.destroyAllWindows()    

        blob = cv2.dnn.blobFromImage(cv2.resize(img_input, (300, 300)), 1.0,(300, 300), (104.0, 177.0, 123.0))
        self.net.setInput(blob)
        faces3 = self.net.forward()      


        height, width = img_input.shape[:2]
        box = faces3[0, 0, 0, 3:7] * np.array([width, height, width, height])
        x, y, x1, y1 = box.astype("int")
        roi = img_input[y:y1, x:x1]

        cv2.imshow('image', roi)
        k = cv2.waitKey(0) & 0xFF
        # wait for ESC key to exit
        if k == 27:
            cv2.destroyAllWindows()

        # roi is input (H,W,3(BGR)), numpy array


        img_ycrcb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCR_CB)
        img_luv = cv2.cvtColor(roi, cv2.COLOR_BGR2LUV)

        ycrcb_hist = self.calc_hist(img_ycrcb)
        luv_hist = self.calc_hist(img_luv)

        feature_vector = np.append(ycrcb_hist.ravel(), luv_hist.ravel())
        feature_vector = feature_vector.reshape(1, len(feature_vector))

        prediction = self.clf.predict_proba(feature_vector)
        prob = prediction[0][1]

        print(prob)

        if prob >= 0.7:
            check = "fake"
        else:
            check = "real"

        return check