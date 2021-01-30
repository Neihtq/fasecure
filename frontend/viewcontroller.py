import os
import cv2
import requests
import numpy as np

from face_detection.face_alignment import FaceAlignment
from utils.constants import VERIFY_ENDPOINT, REGISTER_ENDPOINT, WIPE_ENDPOINT, DETECTION_THRESHOLD, FACE_DETECTION_MODEL, FACE_DETECTION_PROTOTXT

face_detection_model = cv2.dnn.readNetFromCaffe(FACE_DETECTION_PROTOTXT, FACE_DETECTION_MODEL)
face_alignment = FaceAlignment()


def wipe_database():
    response = requests.post(WIPE_ENDPOINT)

    return response


def register(frame, start_x, start_y, end_x, end_y, name):
    aligned_img = align(frame, start_x, start_y, end_x, end_y)
    if aligned_img is not None:
        data = {'image': aligned_img.tolist(), 'name': name}
        response = requests.post(REGISTER_ENDPOINT, json=data)
        print("TESTEST", response)
        return int(response)

    return None


def verify(frame, start_x, start_y, end_x, end_y):
    aligned_img = align(frame, start_x, start_y, end_x, end_y)
    if aligned_img is not None:
        data = {'image': aligned_img.tolist()}
        res = requests.post(VERIFY_ENDPOINT, json=data)
        res_json = res.json()
        closest_label, check = res_json['name'], res_json['access']

        return closest_label, check

    return None, None


def face_detection(frame, h, w):
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    face_detection_model.setInput(blob)
    detections = face_detection_model.forward()
    start_x, start_y, end_x, end_y = None, None, None, None
    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence < DETECTION_THRESHOLD:
            continue

        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        start_x, start_y, end_x, end_y = box.astype("int")

    return start_x, start_y, end_x, end_y


def crop_img(img, start_x, start_y, end_x, end_y):
    height, width = end_y - start_y, end_x - start_x
    cropped_img = img[start_y:start_y + height, start_x:start_x + width]
    cropped_img = cv2.resize(cropped_img, (250, 250))
    return cropped_img


def align(frame, start_x, start_y, end_x, end_y):
    cropped_img = crop_img(frame, start_x - 20, start_y - 20, end_x + 20, end_y + 20)
    aligned_img = face_alignment.align(cropped_img, start_x, start_y, end_x, end_y)

    if aligned_img is None:
        print("Error during Face Detection. Please try again!")
        return

    return aligned_img


def take_shot(directory, filename, frame, start_x, start_y, end_x, end_y):
    cropped_img = crop_img(frame, start_x - 20, start_y - 20, end_x + 20, end_y + 20)
    cropped_aligned_img = face_alignment.align(cropped_img, start_x, start_y, end_x, end_y)
    write_root = os.path.join(directory, filename)
    cv2.imwrite(write_root, cropped_aligned_img)
