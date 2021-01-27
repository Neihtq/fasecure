#!/usr/bin/env python
import PySimpleGUI as sg
import cv2
import numpy as np
from face_detection.gui_frame import detect, crop_img, align_embed, take_shot#, register
import time
import torch
import sys

#from face_detection.face_alignment import FaceAlignment
from os.path import join, dirname, abspath
from models.FaceNet import get_model
from registration_database.RegistrationDatabase import RegistrationDatabase
from utils.prep import img_augmentation

absolute_dir = dirname(abspath(__file__))
face_detection_prototxt = join(absolute_dir, "model", "deploy.prototxt")
face_detection_path = join(absolute_dir, "model", "res10_300x300_ssd_iter_140000.caffemodel")
detection_threshold = 0.5

# Face detection model
#face_detection_model = cv2.dnn.readNetFromCaffe(face_detection_prototxt, face_detection_path)

# Face alignment model
#face_alignment = FaceAlignment()

# Face embedding model
face_embedding_model = get_model()
face_embedding_model.eval()

fixed_initial_registration_threshold = 98.5
registration_database = RegistrationDatabase(fixed_initial_registration_threshold)

def register(augmented_imgs, label):
    for aug_img in augmented_imgs:
        img_embedding_tensor = face_embedding_model(aug_img)
        registration_database.face_registration(label, img_embedding_tensor)
    print("registration for ", label, " successful")

def main():

    sg.theme('Reddit')

    print(sg.Window.get_screen_size())
    w, h = sg.Window.get_screen_size()

    # define the window layout
    layout = [[sg.Text('fasecure', size=(50, 1), justification='center', font='OpenSans-ExtraBold 30')],
                [sg.Button('Face Recognition', size=(25, 2), font='OpenSans-Regular 15'),
                sg.Button('Register New Person', size=(25, 2), font='OpenSans-Regular 15'),
                sg.Button('Database', size=(25, 2), font='OpenSans-Regular 15'),
                sg.Button('Clear Database', size=(25, 2), font='OpenSans-Regular 15')
                #sg.Button('Take Shot', size=(25, 2), font='OpenSans-Regular 15')
                ],
              [sg.Text("No face detected", size=(40, 1), justification='center', font='Helvetica 20')],
              [sg.Image(filename='', key='image')],
              [sg.Output(key='-OUT-', size=(163, 10))]
              ]

    # create the window and show it without the plot
    window = sg.Window('Face Recognition System', layout, location=(0, 0))

    cap = cv2.VideoCapture(0)

    prev_frame_time = 0
    new_frame_time = 0
    access = 0

    password = "1234"
    database_list = []
    label = "abc"
    

    while True:
        event, values = window.read(timeout=20)

        _, frame = cap.read()
        h, w = frame.shape[:2]
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
            #print("Rectangleframe: "+ str(start_x))
            #h, w = frame.shape[:2]  
            
        # SHOW FPS
        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        fps = str(int(fps))
        cv2.putText(frame, fps, (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 0), 3, cv2.LINE_AA)
        
        # SHOW WEBCAM 
        frame_rezised = cv2.resize(frame, (1161, 653))
        imgbytes = cv2.imencode('.png', frame_rezised)[1].tobytes()  # ditto
        window['image'].update(data=imgbytes)

        # --- FUNCTION BUTTONS ---
        if event == 'Face Recognition':
            if start_x:
                embedding, augmented_imgs = align_embed(frame, start_x, start_y, end_x, end_y)
                closest_label, check = registration_database.face_recognition(embedding)

                if check == 'Access':
                    print("User recognized - " + closest_label + " - Access Granted!")
                    #cv2.putText(frame, "User recognized - " + closest_label + " - Access Granted!", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3, cv2.LINE_AA)

                elif check == 'Decline':
                    print("User not recognized - Access Denied!")
                    #cv2.putText(frame, "User not recognized - Access Denied!", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_AA)

            else:
                print("No Face detected. Please try again!")

        elif event == 'Take Shot':        
            print("Takeshot frame: "+ str(start_x))
            if start_x:
                directory = '.\images\snap_shot'
                filename = "testshot_input_pipeline_gui.png"
                
                take_shot(directory, filename, frame, start_x, start_y, end_x, end_y)
            else:
                print("No Face detected. Please try again!")
        

        elif event == 'Register New Person':
            print('Register new person')
            if start_x:
                passwordpopup = sg.popup_get_text('Password for autenthication required', 'Autenthication')
                if passwordpopup == password:
                    print("Password correct - Access to database granted")
                    embedding, augmented_imgs = align_embed(frame, start_x, start_y, end_x, end_y)
                    label = sg.popup_get_text('Name', 'Registration')
                    print(label)
                    register(augmented_imgs, label)
                    #print(text + " resgistered to the database.")
                    database_list.append(label)    
                else:
                    print("Password incorrect - Access to data base denied")
            else:
                print("No Face detected. Please try again!")

        elif event == 'Database':
            passwordpopup = sg.popup_get_text('Password for autenthication required', 'Autenthication')
            if passwordpopup == password:
                print("Password correct - Access to database granted")
                for i in database_list:
                    sg.Print(i)
            if passwordpopup != password:
                print("Access to data base denied")

        elif event == 'Clear Database':
            registration_database.clean_database()
            database_list = []
            print("database was cleaned...")
        
        start_x = None

if __name__ == '__main__':
    main()
    sys.exit(0)


