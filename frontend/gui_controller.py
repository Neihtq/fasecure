#!/usr/bin/env python
import PySimpleGUI as sg
import cv2
import numpy as np
from gui_logic import crop_img, align_embed, take_shot, fps, face_detection, register, verify
import time
import sys

from backend.face_recognition.models.FaceNet import get_model
from backend.database.RegistrationDatabase import RegistrationDatabase

face_embedding_model = get_model()
face_embedding_model.eval()

fixed_initial_registration_threshold = 98.5
registration_database = RegistrationDatabase(fixed_initial_registration_threshold)

detection_threshold = 0.5


def register(augmented_imgs, label):
    '''for aug_img in augmented_imgs:
        img_embedding_tensor = face_embedding_model(aug_img)
        registration_database.face_registration(label, img_embedding_tensor)
    print("registration for ", label, " successful")
    '''
    pass

def main():
    sg.theme('Reddit')
    w, h = sg.Window.get_screen_size()

    # define the window layout
    layout = [[sg.Text('fasecure', size=(41, 1), justification='center', font='OpenSans-ExtraBold 40')],
                [#sg.Button('Face Recognition', size=(25, 2), font='OpenSans-Regular 15'),
                sg.Button('Register New Person', size=(27, 1), font='OpenSans-Regular 18'),
                sg.Button('Database', size=(27, 1), font='OpenSans-Regular 18'),
                sg.Button('Clear Database', size=(26, 1), font='OpenSans-Regular 18')],
                [sg.Text('', key='-TEXT-', background_color='blue', size=(46, 1), font='OpenSans-ExtraBold 35')],
              [sg.Image(filename='', key='image')],
              [sg.Output(key='-OUT-', size=(165, 8))]
              ]

    # create the window and show it without the plot
    window = sg.Window('fasecure - Face Recognition', layout, location=(0, 0))

    cap = cv2.VideoCapture(0)
    password = "1234"
    database_list = []
    label = ""
    t = 15
    box_color =(255, 0, 0)

    prev_frame_time = 0
    new_frame_time = 0

    while True:
        event, values = window.read(timeout=20)

        _, frame = cap.read()
        h, w = frame.shape[:2]

        # FACE DETECTION
        start_x , start_y, end_x, end_y = face_detection(frame, h, w)
        if start_x:
            cv2.rectangle(frame, (start_x-30, start_y-30), (end_x+30, end_y+30), box_color, 3)

        # SHOW FPS
        frame, new_frame_time, prev_frame_time = fps(frame, new_frame_time, prev_frame_time)

        # SHOW WEBCAM
        frame_rezised = cv2.resize(frame, (1010, 570))
        imgbytes = cv2.imencode('.png', frame_rezised)[1].tobytes()  # ditto
        window['image'].update(data=imgbytes)


        # --- FUNCTION BUTTONS ---
        if t % 10 == 0:
        #if event == 'Face Recognition':
            if start_x:
                closest_label, check = verify(frame, start_x, start_y, end_x, end_y)

                if check:
                    print("User recognized - " + closest_label + " - Access Granted!")
                    window['-TEXT-'].update('                                                   Access Granted')
                    window['-TEXT-'].update(background_color='green')
                    #cv2.putText(frame, "User recognized - " + closest_label + " - Access Granted!", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3, cv2.LINE_AA)
                    box_color =(0, 255, 0)
                else:
                    print("User not recognized - Access Denied!")
                    window['-TEXT-'].update('                                                    Access Denied')
                    window['-TEXT-'].update(background_color='red')
                    #cv2.putText(frame, "User not recognized - Access Denied!", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_AA)
                    box_color =(0, 0, 255)
            else:
                print("No Face detected. Please try again!")

        if event == 'Take Shot':
            if start_x:
                directory = '.\images\snap_shot'
                filename = "testshot_input_pipeline_gui.png"

                take_shot(directory, filename, frame, start_x, start_y, end_x, end_y)
            else:
                print("No Face detected. Please try again!")

        elif event == 'Register New Person':
            if start_x:
                passwordpopup = sg.popup_get_text('Password for autenthication required', 'Autenthication')
                if passwordpopup == password:
                    print("Password correct - Access to database granted")
                    label = sg.popup_get_text('Name', 'Registration')
                    response = register(frame, start_x, start_y, end_x, end_y, label)
                    if response != 0:
                        print("register failed")
                    else:
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
            else:
                print("Access to data base denied")

        elif event == 'Clear Database':
            passwordpopup = sg.popup_get_text('Password for autenthication required', 'Autenthication')
            if passwordpopup == password:
                print("Password correct - Access to database granted")
                registration_database.clean_database()
                database_list = []
                print("database was cleaned...")
            else:
                print("Access to data base denied")

        elif event == 'Exit System' or event == sg.WIN_CLOSED:
            break

        start_x = None
        t += 1

if __name__ == '__main__':
    main()
    sys.exit(0)
