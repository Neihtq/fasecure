import sys
import time
import cv2
import threading
import PySimpleGUI as sg

from viewcontroller import wipe_database, take_shot, face_detection, register, verify, get_registered
from utils.constants import ACCESS_DENIED, ACCESS_GRANTED, DB_ACCESS_DENIED, TITLE, SUCCESS, FAIL, NO_FACE, DB_ACCESS_GRANTED, KEEP_IN_FRAME, LOGO

password = "1234"
box_color = (252, 188, 109)


def init_window():
    # define the window layout
    # create the window  
    sg.theme('Reddit')

    #Thien Layout
    """
    layout = [
        [sg.Image(filename='', key='logo'), sg.Text(TITLE, size=(15, 1), justification='center', font='OpenSans-ExtraBold 34')],
        [sg.Button('Register New Person', size=(17, 1), font='OpenSans-Regular 18'),
         sg.Button('Database', size=(8, 1), font='OpenSans-Regular 18'),
         sg.Button('Clear Database', size=(13, 1), font='OpenSans-Regular 18')],
        [sg.Image(filename='', key='image')],
        [sg.Text('', key='-TEXT-', justification='center', background_color='blue', size=(22, 1), font='OpenSans-ExtraBold 35')]
    ]
    """

    #Cao layout
    layout = [
        [sg.Image(filename='', key='logo'), sg.Text(TITLE, size=(20, 1), justification='center', font='OpenSans-ExtraBold 34', text_color="#666666")],
        [sg.Button('Register New Person', button_color = ('white', '#6DBCFC'), size=(23, 1), font='OpenSans-Regular 18'),
         sg.Button('Database', button_color = ('white', '#6DBCFC'), size=(23, 1), font='OpenSans-Regular 18'),
         sg.Button('Clear Database', button_color = ('white', '#6DBCFC'), size=(23, 1), font='OpenSans-Regular 18')],

        [sg.Image(filename='', key='image')],
        [sg.Text('', key='-TEXT-', justification='center', background_color='#6DBCFC', size=(42, 1), font='OpenSans-ExtraBold 31')]
    ]
    window = sg.Window('fasecure - Face Recognition', layout, location=(0, 0))

    return window


def thread_verify(frame, start_x, start_y, end_x, end_y, window):
    global box_color

    if start_x:
        closest_label, check = verify(frame, start_x, start_y, end_x, end_y)
        if check == "out of frame":
            print(KEEP_IN_FRAME)
            window['-TEXT-'].update(KEEP_IN_FRAME)
            window['-TEXT-'].update(background_color='#F63E3E')
            box_color = (0, 0, 255)
        elif check:
            print(ACCESS_GRANTED, closest_label)
            window['-TEXT-'].update('Access Granted')
            window['-TEXT-'].update(background_color='#56E87C')
            box_color = (86, 232, 124)
        else:
            print(ACCESS_DENIED)
            window['-TEXT-'].update('Access Denied')
            window['-TEXT-'].update(background_color='#F63E3E')
            box_color = (0, 0, 255)
    else: 
        print(ACCESS_DENIED)
        window['-TEXT-'].update('Access Denied')
        window['-TEXT-'].update(background_color='#F63E3E')
        box_color = (0, 0, 255)


def fps(frame, prev_frame_time):
    new_frame_time = time.time()
    framerate = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    framerate = str(int(framerate))
    cv2.putText(frame, framerate, (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 0), 3, cv2.LINE_AA)
    return frame, prev_frame_time


def main():
    global box_color
    window = init_window()
    cap = cv2.VideoCapture(0)

    t = 15
    set_logo = True
    prev_frame_time = 0
    while True:
        event, values = window.read(timeout=20)

        if set_logo:
            logo = cv2.imread(LOGO)
            logo_resized = cv2.resize(logo, (225, 200))
            img_bytes = cv2.imencode('.png', logo_resized)[1].tobytes()
            window['logo'].update(data=img_bytes)
            set_logo = False

        _, frame = cap.read()
        h, w = frame.shape[:2]

        start_x, start_y, end_x, end_y, frame = face_detection(frame, h, w, box_color)
        frame, prev_frame_time = fps(frame, prev_frame_time)

        # SHOW WEBCAM
        #rame_resized = cv2.resize(frame, (1300, 731))
        frame_resized = cv2.resize(frame, (1010, 570))
        img_bytes = cv2.imencode('.png', frame_resized)[1].tobytes()
        window['image'].update(data=img_bytes)

        if t % 10 == 0:
            # Face recognition

            if start_x:
                try:
                    _worker = threading.Thread(target=thread_verify, args=(frame, start_x, start_y, end_x, end_y, window))
                    _worker.start()
                except:
                    print("Thread error")

        if event == 'Register New Person':
            if start_x:
                password_dialog = sg.popup_get_text('Password for autenthication required', 'Autenthication', password_char='*')
                if password_dialog == password:
                    print(DB_ACCESS_GRANTED)
                    label = sg.popup_get_text('Name', 'Registration')
                    response = register(frame, start_x, start_y, end_x, end_y, label)
                    if response != 0:
                        print(FAIL)
                else:
                    print(DB_ACCESS_DENIED)
            else:
                print(NO_FACE)

        elif event == 'Database':
            password_dialog = sg.popup_get_text('Password for autenthication required', 'Autenthication', password_char='*')
            if password_dialog == password:
                list_names = get_registered()
                print(DB_ACCESS_GRANTED)
                if list_names:
                    for i in list_names:
                        sg.Print(i)
            else:
                print(DB_ACCESS_DENIED)

        elif event == 'Clear Database':
            password_dialog = sg.popup_get_text('Password for autenthication required', 'Autenthication', password_char='*')
            if password_dialog == password:
                print(DB_ACCESS_GRANTED)
                response = wipe_database()
                if response != 0:
                    print(FAIL)
                else:
                    print(SUCCESS)
            else:
                print(DB_ACCESS_DENIED)

        elif event == 'Exit System' or event == sg.WIN_CLOSED:
            cap.release()
            break

        t += 1

    window.close()

if __name__ == '__main__':
    main()
    sys.exit(0)
