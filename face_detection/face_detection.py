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
PROTO_TXT = join(absolute_dir, "model", "deploy.prototxt")
MODEL = join(absolute_dir, "model", "res10_300x300_ssd_iter_140000.caffemodel")
THRESHOLD = 0.5

# Face detection model
net = cv2.dnn.readNetFromCaffe(PROTO_TXT, MODEL)

# Face alignment model
face_alignment_model = FaceAlignment()

# Face embedding model
face_embedding_model = get_model().eval()

# Face registration & recognition
registration_database = RegistrationDatabase(fixed_threshold=98.5)

def face_detection(callback=None):
    
    prev_frame_time = 0
    new_frame_time = 0
    access = 0

    cam = cv2.VideoCapture(1)

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
                
            color =(255, 0, 0)
            stroke = 3
            cv2.rectangle(frame, (start_x-30, start_y-30), (end_x+30, end_y+30), color, stroke)
            

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
        
            
        # --- FACE REGISTRATION ---

        # if key "6" pressed, then take current frame and register user as "tobias"      
        if cv2.waitKey(20) & 0xFF == ord('6'):
            embedding, augmented_imgs = align_embed(frame, start_x, start_y, end_x, end_y)
            label = "tobias"
            register(augmented_imgs, label)

        # if key "7" pressed, then take current frame and register user as "cao"    
        if cv2.waitKey(20) & 0xFF == ord('7'):
            embedding, augmented_imgs = align_embed(frame, start_x, start_y, end_x, end_y)
            label = "cao"
            register(augmented_imgs, label)
            
        # if key "8" pressed, then take current frame and register user as "thien"
        if cv2.waitKey(20) & 0xFF == ord('8'):
            embedding, augmented_imgs = align_embed(frame, start_x, start_y, end_x, end_y)
            label = "thien"
            register(augmented_imgs, label)

        # if key "9" pressed, then take current frame and register user as "simon"
        if cv2.waitKey(20) & 0xFF == ord('9'):           
            embedding, augmented_imgs = align_embed(frame, start_x, start_y, end_x, end_y)
            label = "simon"
            register(augmented_imgs, label)


        # if key "r" pressed, then reset the database
        if cv2.waitKey(20) & 0xFF == ord('c'):           
            registration_database.clean_database()
            print("database was cleaned...")
        

        cv2.imshow('Webcam', frame)

        # --- FACE RECOGNITION ---
        # if key "1" is pressed, then take current frame and recognize user
        if cv2.waitKey(20) & 0xFF == ord('1'):

            embedding, augmented_imgs = align_embed(frame, start_x, start_y, end_x, end_y)
            closest_label, check = registration_database.face_recognition(embedding)

            if check == 'Access':
                cv2.putText(frame, "User recognized - " + closest_label + " - Access Granted!", (10, 1000), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 3, cv2.LINE_AA)
            elif check == 'Decline':
                cv2.putText(frame, "User not recognized - Access Denied!", (10, 1000), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 3, cv2.LINE_AA)
            
        
        if cv2.waitKey(20) & 0xFF == ord('q'):
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
    crop_img = cv2.resize(crop_img, (250, 250))
    return crop_img

def align_embed(frame, start_x, start_y, end_x, end_y):
    #crop image
    cropped_inference = crop_img(frame, start_x-20, start_y-20, end_x+20, end_y+20)
    #align image
    
    detected_face_numpy = face_alignment_model.align(cropped_inference)
    # img_item = "aligned-image.png"
    # cv2.imwrite(img_item, cropped_aligned_inference)


    detected_face = torch.from_numpy(detected_face_numpy).permute(2, 1, 0).unsqueeze(0).float()

    # Swap color channels from opencv (BGR) to pytorch (RGB) implementation

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

if __name__ == '__main__':
    face_detection()
    sys.exit(0)
    