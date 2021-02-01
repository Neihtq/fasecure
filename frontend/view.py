import sys
import time
import cv2
import threading
import PySimpleGUI as sg

from viewcontroller import wipe_database, take_shot, face_detection, register, verify, get_registered
from utils.constants import ACCESS_DENIED, ACCESS_GRANTED, DB_ACCESS_DENIED, TITLE, SUCCESS, FAIL, NO_FACE, DB_ACCESS_GRANTED, KEEP_IN_FRAME

password = "1234"
box_color = (255, 0, 0)

thread = threading.Thread()

start_x, start_y, end_x, end_y, frame = None, None, None, None, None
access = False
closest_label = ""

class VerificationThread(threading.Thread):
    def __init__(self):
        super(VerificationThread, self).__init__()

    def set_variables(self, frame, start_x, start_y, end_x, end_y, window):
        self.frame, self.start_x, self.start_y, self.end_x, self.end_y, self.window = frame, start_x, start_y, end_x, end_y, window

    def run(self):
        verify(self.frame, self.start_x, self.start_y, self.end_x, self.end_y, self.window)


def init_window():
    # define the window layout
    # create the window
    sg.theme('Reddit')
    layout = [
        [sg.Text(TITLE, size=(38, 1), justification='center', font='OpenSans-ExtraBold 40')],
        [sg.Button('Register New Person', size=(23, 1), font='OpenSans-Regular 18'),
         sg.Button('Database', size=(23, 1), font='OpenSans-Regular 18'),
         sg.Button('Clear Database', size=(23, 1), font='OpenSans-Regular 18')],
        [sg.Text('', key='access_label', background_color='blue', size=(38, 1), font='OpenSans-ExtraBold 35')],
        [sg.Image(filename='', key='image')]
    ]
    window = sg.Window('fasecure - Face Recognition', layout, location=(0, 0))

    return window


def thread_verify():
    global closest_label
    global access
    global start_x
    global start_y
    global end_x
    global end_y
    global frame
    global closest_label

    if start_x:
        closest_label, access = verify(frame, start_x, start_y, end_x, end_y)


def fps(frame, prev_frame_time):
    new_frame_time = time.time()
    framerate = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    framerate = str(int(framerate))
    cv2.putText(frame, framerate, (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 0), 3, cv2.LINE_AA)
    return frame, prev_frame_time


def main():
    global box_color
    global access
    global start_x
    global start_y
    global end_x
    global end_y
    global frame
    global closest_label
    window = init_window()
    cap = cv2.VideoCapture(0)

    t = 15

    worker = threading.Thread(target=thread_verify)
    worker.start()
    prev_frame_time = 0
    while True:
        event, values = window.read(timeout=20)

        _, frame = cap.read()
        h, w = frame.shape[:2]

        start_x, start_y, end_x, end_y, frame = face_detection(frame, h, w, box_color)
        frame, prev_frame_time = fps(frame, prev_frame_time)

        # SHOW WEBCAM
        frame_resized = cv2.resize(frame, (1010, 570))
        img_bytes = cv2.imencode('.png', frame_resized)[1].tobytes()
        window['image'].update(data=img_bytes)

        if access == "out of frame":
            print(KEEP_IN_FRAME)
            window['access_label'].update(KEEP_IN_FRAME)
            window['access_label'].update(background_color='red')
            box_color = (0, 0, 255)
        elif access:
            print(ACCESS_GRANTED, closest_label)
            window['access_label'].update('Access Granted')
            window['access_label'].update(background_color='green')
            box_color = (0, 255, 0)
        else:
            print(ACCESS_DENIED)
            window['access_label'].update('Access Denied')
            window['access_label'].update(background_color='red')
            box_color = (0, 0, 255)

        # --- FUNCTION BUTTONS ---
        if event == 'Take Shot':
            if start_x:
                directory = '.\images\snap_shot'
                filename = "testshot_input_pipeline_gui.png"

                take_shot(directory, filename, frame, start_x, start_y, end_x, end_y)
            else:
                print(NO_FACE)

        elif event == 'Register New Person':
            if start_x:
                password_dialog = sg.popup_get_text('Password for autenthication required', 'Autenthication')
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
            password_dialog = sg.popup_get_text('Password for authenthication required', 'Autenthication')
            if password_dialog == password:
                list_names = get_registered()
                print(DB_ACCESS_GRANTED)
                if list_names:
                    for i in list_names:
                        sg.Print(i)
            else:
                print(DB_ACCESS_DENIED)

        elif event == 'Clear Database':
            password_dialog = sg.popup_get_text('Password for autenthication required', 'Autenthication')
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
