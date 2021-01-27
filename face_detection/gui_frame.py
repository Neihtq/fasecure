import cv2
import torch
#import imutils
import time
import numpy as np
import sys
from face_detection.face_alignment import FaceAlignment

from os.path import join, dirname, abspath
from models.FaceNet import get_model
from registration_database.RegistrationDatabase import RegistrationDatabase
from utils.prep import img_augmentation


absolute_dir = dirname(abspath(__file__))
face_detection_prototxt = join(absolute_dir, "model", "deploy.prototxt")
face_detection_path = join(absolute_dir, "model", "res10_300x300_ssd_iter_140000.caffemodel")
detection_threshold = 0.5

# Face detection model
face_detection_model = cv2.dnn.readNetFromCaffe(face_detection_prototxt, face_detection_path)

# Face alignment model
face_alignment = FaceAlignment()

# Face embedding model
face_embedding_model = get_model()
face_embedding_model.eval()

# Face registration & recognition
fixed_initial_registration_threshold = 98.5
registration_database = RegistrationDatabase(fixed_initial_registration_threshold)


def detect(image):
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    face_detection_model.setInput(blob)
    detections = face_detection_model.forward()
    return detections

def crop_img(img, start_x, start_y, end_x, end_y):
    height, width = end_y - start_y, end_x - start_x
    crop_img = img[start_y:start_y+height, start_x:start_x+width]
    crop_img = cv2.resize(crop_img, (250, 250))
    return crop_img

def align_embed(frame, start_x, start_y, end_x, end_y):
    #crop image
    cropped_img = crop_img(frame, start_x-20, start_y-20, end_x+20, end_y+20)
    
    #align image
    detected_face_numpy = face_alignment.align(cropped_img, start_x, start_y, end_x, end_y)
    
    if detected_face_numpy is None:
        print("Error during Face Detection. Please try again!")
        return None, None

    detected_face = torch.from_numpy(detected_face_numpy).permute(2, 1, 0).unsqueeze(0).float()

    # to do: Swap color channels from opencv (BGR) to pytorch (RGB) implementation

    # perform augmentations
    augmented_imgs = img_augmentation(detected_face)
    
    # embedding model   
    embedding = face_embedding_model(augmented_imgs[0]) 

    return embedding, augmented_imgs

def register(augmented_imgs, label):
    for aug_img in augmented_imgs:
        img_embedding_tensor = face_embedding_model(aug_img)
        registration_database.face_registration(label, img_embedding_tensor)
    print("registration for ", label, " successful")

def take_shot(directory, filename, frame, start_x, start_y, end_x, end_y):        
    print("Takeshot in function frame after crop before crop: "+ str(start_x))
    cropped_img = crop_img(frame, start_x-20, start_y-20, end_x+20, end_y+20)
    print("Takeshot in function frame after crop: "+ str(start_x))
    cropped_aligned_img = face_alignment.align(cropped_img, start_x, start_y, end_x, end_y)
    print("Takeshot in function frame after align: "+ str(start_x))
    write_root = join(directory, filename)
    cv2.imwrite(write_root, cropped_aligned_img)

    print("Snapshot taken and saved as: " + filename)

if __name__ == '__main__':
    gui_frame()
    sys.exit(0)