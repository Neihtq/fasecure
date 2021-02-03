import cv2
import numpy as np

from os.path import join, dirname, abspath


if __name__ == '__main__':
    from face_alignment import align_img
else:
    from face_detection.face_alignment import align_img

from utils.constants import FACE_ALIGNMENT_ERROR, DETECTION_THRESHOLD


absolute_dir = dirname(abspath(__file__))
PROTO_TXT = join(absolute_dir, "model", "deploy.prototxt")
MODEL = join(absolute_dir, "model", "res10_300x300_ssd_iter_140000.caffemodel")
THRESHOLD = 0.5
face_detection_model = cv2.dnn.readNetFromCaffe(PROTO_TXT, MODEL)


def main():
    cap = cv2.VideoCapture(0)
    while True:
        _, frame = cap.read()
        h, w = frame.shape[:2]
        detections = detect(frame)
        for i in np.arange(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence < DETECTION_THRESHOLD:
                continue

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            start_x, start_y, end_x, end_y = box.astype("int")
            cv2.rectangle(frame, (start_x - 30, start_y - 30), (end_x + 30, end_y + 30), (255, 0, 0), 3)

        _ = align(frame, start_x, start_y, end_x, end_y)
        cv2.imshow('Webcam', frame)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def detect(frame):
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    face_detection_model.setInput(blob)
    detections = face_detection_model.forward()
    return detections


def crop_img(img, start_x, start_y, end_x, end_y):
    height, width = end_y - start_y, end_x - start_x
    cropped_img = img[start_y:start_y + height, start_x:start_x + width]

    return cropped_img


def align(frame, start_x, start_y, end_x, end_y):
    cropped_img = crop_img(frame, start_x - 20, start_y - 20, end_x + 20, end_y + 20)  
    aligned_img = align_img(cropped_img, start_x, start_y, end_x, end_y)

    directory = "./" #"C:\Users\caoso\OneDrive\Dokumente\GitHub\IBM-labcourse\images\snap_shot"
    filename = "demo_test_img.png"

    write_root = join(directory, filename)
    cv2.imwrite(write_root, aligned_img)

    if aligned_img is None:
        print(FACE_ALIGNMENT_ERROR)
        return

    return aligned_img


if __name__ == '__main__':
    main()