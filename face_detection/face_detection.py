import cv2
import torch
import imutils
import time
import numpy as np
import sys
from face_detection.face_alignment import FaceAlignment

from os.path import join, dirname, abspath
from models.FaceNet import FaceNet
from registration_database.RegistrationDatabase import RegistrationDatabase


absolute_dir = dirname(abspath(__file__))
PROTO_TXT = join(absolute_dir, "model", "deploy.prototxt")
MODEL = join(absolute_dir, "model", "res10_300x300_ssd_iter_140000.caffemodel")
THRESHOLD = 0.5

def face_detection(callback=None):
    net = cv2.dnn.readNetFromCaffe(PROTO_TXT, MODEL)

    prev_frame_time = 0
    new_frame_time = 0
    access = 0

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
        
        
        if cv2.waitKey(20) & 0xFF == ord('1'):
            access = 1
            cv2.putText(frame, "User recognized - Access Granted!", (10, 1000), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 3, cv2.LINE_AA)
        if cv2.waitKey(20) & 0xFF == ord('2'):
            access = 2
            cv2.putText(frame, "User not recognized - Access Denied!", (10, 1000), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 3, cv2.LINE_AA)
        if cv2.waitKey(20) & 0xFF == ord('3'):
            access = 0
        
        
        if access == 1:
            cv2.putText(frame, "User recognized - Access Granted!", (10, 1000), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 3, cv2.LINE_AA)
        elif access == 2:
            cv2.putText(frame, "User not recognized - Access Denied!", (10, 1000), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 3, cv2.LINE_AA)
        

        cv2.imshow('Webcam', frame)
        
        take_shot = False
        
        if cv2.waitKey(20) & 0xFF == ord('4'):
            take_shot = True
        if take_shot:
            #crop big image
            cropped_inference = crop_img(frame, start_x-20, start_y-20, end_x+20, end_y+20)
            #align image
            fa = FaceAlignment()
            cropped_aligned_inference = fa.align(cropped_inference)
            img_item = "aligned-image.png"
            cv2.imwrite(img_item, cropped_aligned_inference)

            
            
            #pretrained model
            embedding_model = FaceNet()
            embedding_model.eval()
            
            tensor_cropped_aligned_inference = torch.from_numpy(cropped_aligned_inference).double()

            #inference_embedding_tensor = embedding_model(tensor_cropped_aligned_inference.permute(2, 1, 0).unsqueeze(0))
            test_tensor = tensor_cropped_aligned_inference.permute(2, 1, 0).unsqueeze(0)
            print(test_tensor.type())
            
            
            #registration and recognition
            database = RegistrationDatabase()

            #registration
            #newname = "Cao"
            #database.face_registration(newname, inference_embedding_tensor)

            #inference to recognition process
            closest_label, check = database.face_recognition(inference_embedding_tensor)
            if check == 'Access':
                cv2.putText(frame, "User recognized - " + closest_label + " - Access Granted!", (10, 1000), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 3, cv2.LINE_AA)
            elif check == 'Decline':
                cv2.putText(frame, "User not recognized - Access Denied!", (10, 1000), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 3, cv2.LINE_AA)
            take_shot = False
            
            
        
        
        if cv2.waitKey(20) & 0xFF == 27:#ord('q'):
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


if __name__ == '__main__':
    face_detection()
    sys.exit(0)
    