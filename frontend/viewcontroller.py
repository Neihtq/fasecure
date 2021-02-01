import os
import cv2
import requests
import numpy as np

from face_detection.face_detection import align, crop_img, detect
from utils.constants import VERIFY_ENDPOINT, REGISTER_ENDPOINT, WIPE_ENDPOINT, DETECTION_THRESHOLD, FACE_DETECTION_MODEL, FACE_DETECTION_PROTOTXT, LIST_ALL_ENDPOINT, BACKEND_UNREACHABLE, TRY_AGAIN, KEEP_IN_FRAME

face_detection_model = cv2.dnn.readNetFromCaffe(FACE_DETECTION_PROTOTXT, FACE_DETECTION_MODEL)


def make_request(url, method, data):
    try:
        res = requests.request(method, url, json=data)
    except:
        print(BACKEND_UNREACHABLE)
        res = None
    return res


def get_registered():
    res = make_request(LIST_ALL_ENDPOINT, 'GET')
    if res:
        list_names = res.json()["names"]
        return list_names
    return None
    

def wipe_database():
    res = make_request(WIPE_ENDPOINT, 'POST')
    if res:
        status = res.json()["status"]
        return status

    return -1


def register(frame, start_x, start_y, end_x, end_y, name):
    try:
        aligned_img = align(frame, start_x, start_y, end_x, end_y)
    except:
        print(KEEP_IN_FRAME)

    try:
        data = {'image': aligned_img.tolist(), 'name': name}
        res = make_request(REGISTER_ENDPOINT, 'POST', data)
        status = res.json()["status"]
        return status
    except:
        print(TRY_AGAIN)

    return -1


def verify(frame, start_x, start_y, end_x, end_y):
    try:
        aligned_img = align(frame, start_x, start_y, end_x, end_y)
    except:
        return None, "out of frame"

    try:
        data = {'image': aligned_img.tolist()}
        res = make_request(VERIFY_ENDPOINT, 'POST', data)
        res_json = res.json()
        closest_label, check = res_json['name'], res_json['access']
        return closest_label, check
    except:
        print(TRY_AGAIN)

    return None, None


def face_detection(frame, h, w, color):
    detections = detect(frame)
    start_x, start_y, end_x, end_y = None, None, None, None
    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence < DETECTION_THRESHOLD:
            continue

        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        start_x, start_y, end_x, end_y = box.astype("int")
        cv2.rectangle(frame, (start_x - 30, start_y - 30), (end_x + 30, end_y + 30), color, 3)

    return start_x, start_y, end_x, end_y, frame


def take_shot(directory, filename, frame, start_x, start_y, end_x, end_y):
    cropped_img = crop_img(frame, start_x - 20, start_y - 20, end_x + 20, end_y + 20)
    cropped_aligned_img = align(cropped_img, start_x, start_y, end_x, end_y)
    write_root = os.path.join(directory, filename)
    cv2.imwrite(write_root, cropped_aligned_img)
