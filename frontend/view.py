import sys
import time
import cv2
import PySimpleGUI as sg

from gui_logic import wipe_database, take_shot, face_detection, register, verify

password = "1234"
box_color = (255, 0, 0)


def init_window():
    # define the window layout
    # create the window
    sg.theme('Reddit')
    layout = [
        [sg.Text('fasecure', size=(41, 1), justification='center', font='OpenSans-ExtraBold 40')],
        [sg.Button('Register New Person', size=(27, 1), font='OpenSans-Regular 18'),
         sg.Button('Database', size=(27, 1), font='OpenSans-Regular 18'),
         sg.Button('Clear Database', size=(26, 1), font='OpenSans-Regular 18')],
        [sg.Text('', key='-TEXT-', background_color='blue', size=(46, 1), font='OpenSans-ExtraBold 35')],
        [sg.Image(filename='', key='image')],
        [sg.Output(key='-OUT-', size=(165, 8))]
    ]
    window = sg.Window('fasecure - Face Recognition', layout, location=(0, 0))

    return window


def fps(frame, prev_frame_time):
    new_frame_time = time.time()
    framerate = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    framerate = str(int(framerate))
    cv2.putText(frame, framerate, (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 0), 3, cv2.LINE_AA)
    return frame, prev_frame_time


def detect_face(frame, h, w):
    start_x, start_y, end_x, end_y = face_detection(frame, h, w)
    if start_x:
        cv2.rectangle(frame, (start_x - 30, start_y - 30), (end_x + 30, end_y + 30), box_color, 3)

    return start_x, start_y, end_x, end_y, frame

def main():
    global box_color

    window = init_window()
    cap = cv2.VideoCapture(0)

    database_list = []
    t = 15

    prev_frame_time = 0
    while True:
        event, values = window.read(timeout=20)

        _, frame = cap.read()
        h, w = frame.shape[:2]

        start_x, start_y, end_x, end_y, frame = detect_face(frame, h, w)
        frame, prev_frame_time = fps(frame, prev_frame_time)

        # SHOW WEBCAM
        frame_resized = cv2.resize(frame, (1010, 570))
        img_bytes = cv2.imencode('.png', frame_resized)[1].tobytes()
        window['image'].update(data=img_bytes)

        if t % 10 == 0:
            # Face recognition
            if start_x:
                closest_label, check = verify(frame, start_x, start_y, end_x, end_y)
                if check:
                    print(f"User recognized - {closest_label} - Access Granted!")
                    window['-TEXT-'].update('                                                   Access Granted')
                    window['-TEXT-'].update(background_color='green')
                    box_color = (0, 255, 0)
                else:
                    print("User not recognized - Access Denied!")
                    window['-TEXT-'].update('                                                    Access Denied')
                    window['-TEXT-'].update(background_color='red')
                    box_color = (0, 0, 255)


        # --- FUNCTION BUTTONS ---
        if event == 'Take Shot':
            if start_x:
                directory = '.\images\snap_shot'
                filename = "testshot_input_pipeline_gui.png"

                take_shot(directory, filename, frame, start_x, start_y, end_x, end_y)
            else:
                print("No Face detected. Please try again!")

        elif event == 'Register New Person':
            if start_x:
                password_dialog = sg.popup_get_text('Password for autenthication required', 'Autenthication')
                if password_dialog == password:
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
            password_dialog = sg.popup_get_text('Password for authenthication required', 'Autenthication')
            if password_dialog == password:
                print("Password correct - Access to database granted")
                for i in database_list:
                    sg.Print(i)
            else:
                print("Access to data base denied")

        elif event == 'Clear Database':
            password_dialog = sg.popup_get_text('Password for autenthication required', 'Autenthication')
            if password_dialog == password:
                print("Password correct - Access to database granted")
                response = wipe_database()
                if response != 0:
                    print("Wiping database failed.")
                else:
                    database_list = []
                    print("database was cleaned...")
            else:
                print("Access to data base denied")

        elif event == 'Exit System' or event == sg.WIN_CLOSED:
            cap.release()
            break

        t += 1

if __name__ == '__main__':
    main()
    sys.exit(0)
