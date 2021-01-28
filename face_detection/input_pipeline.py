import cv2
import torch
#import imutils
import time
import numpy as np
import sys
from face_detection.face_alignment import FaceAlignment

from os.path import join, dirname, abspath
from models.FaceNet import get_model
from database.RegistrationDatabase import RegistrationDatabase
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


def input_pipeline(callback=None):
    
    prev_frame_time = 0
    new_frame_time = 0
    access = 0

    # Choose camera input (maybe have to adapt input parameter to "1")
    cam = cv2.VideoCapture(0)

    while True:
        _, frame = cam.read()
        

        #start_x , start_y, end_x, end_y = face_detection(frame)

        h, w = frame.shape[:2]
    
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        face_detection_model.setInput(blob)
        detections = face_detection_model.forward()
        
        # --- FACE DETECTION ---
        detections = detect(frame)

        for i in np.arange(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            if confidence < detection_threshold:
                continue
            
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            start_x , start_y, end_x, end_y = box.astype("int")
                
            color =(255, 0, 0)
            stroke = 3
            cv2.rectangle(frame, (start_x-30, start_y-30), (end_x+30, end_y+30), color, stroke)
            #h, w = frame.shape[:2]   


        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        fps = str(int(fps))
        cv2.putText(frame, fps, (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 0), 3, cv2.LINE_AA)
        
        # SHOW WEBCAM INPUT
        #frame = cv2.resize(frame, (300, 300))
        cv2.imshow('Webcam', frame)
        
        # CLOSE APPLICATION        
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break
    

        # --- FACE REGISTRATION ---

        # if key "6" pressed, then take current frame and register user as "Tobias"      
        
        if cv2.waitKey(20) & 0xFF == ord('6'):
            if start_x:
                embedding, augmented_imgs = align_embed(frame, start_x, start_y, end_x, end_y)
                if embedding == None:
                    continue
                label = "Tobias"
                register(augmented_imgs, label)
            else:
                print("No Face detected. Please try again!")
        
        # if key "7" pressed, then take current frame and register user as "Cao"    
        if cv2.waitKey(20) & 0xFF == ord('7'):
            if start_x:
                embedding, augmented_imgs = align_embed(frame, start_x, start_y, end_x, end_y)
                if embedding == None:
                    continue
                label = "Cao"
                register(augmented_imgs, label)
            else:
                print("No Face detected. Please try again!")
        """   
        # if key "8" pressed, then take current frame and register user as "Thien"
        if cv2.waitKey(20) & 0xFF == ord('8'):
            embedding, augmented_imgs = align_embed(frame, start_x, start_y, end_x, end_y)
            label = "Thien"
            register(augmented_imgs, label)

        # if key "9" pressed, then take current frame and register user as "Simon"
        if cv2.waitKey(20) & 0xFF == ord('9'):           
            embedding, augmented_imgs = align_embed(frame, start_x, start_y, end_x, end_y)
            label = "Simon"
            register(Augmented_imgs, label)
        """

        # if key "c" pressed, then reset the database
        
        if cv2.waitKey(20) & 0xFF == ord('c'):           
            registration_database.clean_database()
            print("database was cleaned...")
        

        # --- FACE RECOGNITION ---
        
        # if key "1" is pressed, then take current frame and recognize user
        if cv2.waitKey(20) & 0xFF == ord('1'):
            if start_x:
                embedding, augmented_imgs = align_embed(frame, start_x, start_y, end_x, end_y)
                if embedding == None:
                    continue
                closest_label, check = registration_database.face_recognition(embedding)

                if check == 'Access':
                    print("User recognized - " + closest_label + " - Access Granted!")
                    
                    cv2.putText(frame, "User recognized - " + closest_label + " - Access Granted!", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3, cv2.LINE_AA)
                elif check == 'Decline':
                    print("User not recognized - Access Denied!")
                    cv2.putText(frame, "User not recognized - Access Denied!", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_AA)
            else:
                print("No Face detected. Please try again!")
        
        # --- TAKE SHOT ---
         
        if cv2.waitKey(20) & 0xFF == ord('s'):
            if start_x:
                print(type(blob))
                directory = '.\images\snap_shot'
                filename = "testshot_input_pipeline.png"
                
                take_shot(directory, filename, frame, start_x, start_y, end_x, end_y)
            else:
                print("No Face detected. Please try again!")
        
        start_x = None
    cam.release()
    cv2.destroyAllWindows()
    return



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
    cropped_img = crop_img(frame, start_x-20, start_y-20, end_x+20, end_y+20)
    cropped_aligned_img = face_alignment.align(cropped_img, start_x, start_y, end_x, end_y)

    write_root = join(directory, filename)
    cv2.imwrite(write_root, cropped_aligned_img)

    print("Snapshot taken and saved as: " + filename)

if __name__ == '__main__':
    input_pipeline()
    sys.exit(0)