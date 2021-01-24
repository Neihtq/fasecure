import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from face_detection.face_alignment import FaceAlignment

face_cascade = cv2.CascadeClassifier('./face_detection/model/haarcascade_frontalface_default.xml')

output = "./data/images/lfw_aligned/"

def detect_and_align(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    detections = detect(img)
    try:
        x, y, w, h = select_closest_face(detections,  img.shape[:2])
        cropped_img = crop_img(img, x-20, y-20, w+20, h+20)
    except:
        return

    fa = FaceAlignment()
    try:
        aligned_img = fa.align(cropped_img)
    except:
        aligned_img = cropped_img

    head, fpath = os.path.split(img_path)
    _, folder = os.path.split(head)
    dest = os.path.join(output, folder)
    if not os.path.exists(dest):
        try:
            os.mkdir(dest)
        except:
            print(dest, "already exists")

    save_path = os.path.join(dest, fpath)
    try:
        cv2.imwrite(save_path, aligned_img)
    except:
        pass

def select_closest_face(detections, shape):
    face_dict = {}
    areas = []
    for (x, y, w, h) in detections:        
        width, height = w + 40, h + 40
        area = width * height
        face_dict[area] = (x, y, w, h)
        areas.append(area)

    return face_dict[np.array(areas).max()]

def crop_img(img, x, y, w, h):
    crop_img = img[x-20:x+w+20, y-20:y+h+20]
    crop_img = cv2.resize(crop_img, (224, 224))
    return crop_img

def detect(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    return faces

