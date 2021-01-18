import cv2
import torch
import imutils
import time
import numpy as np

from os.path import join, dirname, abspath

absolute_dir = dirname(abspath(__file__))
PROTO_TXT = join(absolute_dir, "model", "deploy.prototxt")
MODEL = join(absolute_dir, "model", "res10_300x300_ssd_iter_140000.caffemodel")
THRESHOLD = 0.5

def face_detection(callback=None):
    net = cv2.dnn.readNetFromCaffe(PROTO_TXT, MODEL)

    prev_frame_time = 0
    new_frame_time = 0

    cam = cv2.VideoCapture(0)
    while True:
        _, frame = cam.read()
        
        h, w = frame.shape[:2]
        
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()
        for i in np.arange(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            if confidence < THRESHOLD:
                continue
            
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            start_x , start_y, end_x, end_y = box.astype("int")
                

            #cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (0, 0, 255), 2)
            frame = crop_img(frame, start_x-10, start_y-10, end_x+10, end_y+10)

            if callback:
                tensor = callback(frame)
                print(tensor.shape)
                print(tensor)
                cam.release()
                cv2.destroyAllWindows()
                return


        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        fps = str(int(fps))
        cv2.putText(frame, fps, (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 0), 3, cv2.LINE_AA)

        cv2.imshow('Webcam', frame)

        if cv2.waitKey(1) == 27:
            break

    cam.release()
    cv2.destroyAllWindows()


def detect(image, net):
    blob = cv2.dnn.blobFromImage(image, 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()
    return detections


def crop_img(img, start_x, start_y, end_x, end_y):
    height, width = end_y - start_y, end_x - start_x
    crop_img = img[start_y:start_y+height, start_x:start_x+width]
    crop_img = cv2.resize(crop_img, (400, 400))
    return crop_img


if __name__ == '__main__':
    face_detection()
    