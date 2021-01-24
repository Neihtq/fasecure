import os
import cv2
import numpy as np
import multiprocessing as mp

from face_detection.face_detection import detect


def detect_and_align(img):
    detections = detect(img)
    start_x, start_y, end_x, end_y = select_closest_face(detections)
    cropped_img = crop_img(img, start_x, start_y, end_x, end_y)
    


def select_closest_face(detections):
    face_dict = {}
    areas = []
    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        
        if confidence < THRESHOLD:
            continue
        
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        start_x, start_y, end_x, end_y = box.astype("int")
        height, width = end_y - start_y, end_x - start_x
        area = height * width
        face_dict[area] = (start_x, start_y, end_x, end_y)
        areas.append(area)
    return face_dict[np.array(areas).max()]

def crop_img(img, start_x, start_y, end_x, end_y):
    height, width = end_y - start_y, end_x - start_x
    crop_img = img[start_y:start_y+height, start_x:start_x+width]
    crop_img = cv2.resize(crop_img, (400, 400))
    return crop_img