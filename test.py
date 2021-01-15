# import numpy as np
# import cv2
# from sklearn.externals import joblib

def calc_hist(img):
    histogram = [0] * 3
    for j in range(3):
        histr = cv2.calcHist([img], [j], None, [256], [0, 256])
        histr *= 255.0 / histr.max()
        histogram[j] = histr
    return np.array(histogram)

modelFile = "./pretrained_model/res10_300x300_ssd_iter_140000.caffemodel"
configFile = "./pretrained_model/deploy.prototxt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
clf = joblib.load('./pretrained_model/face_spoofing.pkl')
cap = cv2.VideoCapture(0)
#cap.open(0)
# width = 320
# height = 240
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)

sample_number = 1
count = 0
measures = np.zeros(sample_number, dtype=np.float)

while True:
    ret, img = cap.read()
    print("----------", cap.isOpened())
    print("----", ret)
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0,(300, 300), (104.0, 177.0, 123.0))
    
    net.setInput(blob)
    faces3 = net.forward()

    measures[count%sample_number]=0
    height, width = img.shape[:2]
    for i in range(faces3.shape[2]):
        confidence = faces3[0, 0, i, 2]
        if confidence > 0.5:
            box = faces3[0, 0, i, 3:7] * np.array([width, height, width, height])
            (x, y, x1, y1) = box.astype("int")
            # cv2.rectangle(img, (x, y), (x1, y1), (0, 0, 255), 5)
            roi = img[y:y1, x:x1]

            point = (0,0)
            
            img_ycrcb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCR_CB)
            img_luv = cv2.cvtColor(roi, cv2.COLOR_BGR2LUV)
    
            ycrcb_hist = calc_hist(img_ycrcb)
            luv_hist = calc_hist(img_luv)
    
            feature_vector = np.append(ycrcb_hist.ravel(), luv_hist.ravel())
            feature_vector = feature_vector.reshape(1, len(feature_vector))
    
            prediction = clf.predict_proba(feature_vector)
            prob = prediction[0][1]
    
            measures[count % sample_number] = prob
    
            cv2.rectangle(img, (x, y), (x1, y1), (255, 0, 0), 2)
    
            point = (x, y-5)
    
            print (measures, np.mean(measures))
            if 0 not in measures:
                text = "True"
                if np.mean(measures) >= 0.7:
                    text = "False"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(img=img, text=text, org=point, fontFace=font, fontScale=0.9, color=(0, 0, 255),
                                thickness=2, lineType=cv2.LINE_AA)
                else:
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(img=img, text=text, org=point, fontFace=font, fontScale=0.9,
                                color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
        
    count+=1
    cv2.imshow('img_rgb', img)
    
    key = cv2.waitKey(1)
    if key & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

# -------------------------------------------------------------------------------------------

# import cv2
# import torch
# #import imutils
# import time
# import numpy as np
# import sys
# #from face_alignment import FaceAlignment

# from os.path import join, dirname, abspath

# absolute_dir = dirname(abspath(__file__))
# PROTO_TXT = "./pretrained_model/deploy.prototxt"
# MODEL = "./pretrained_model/res10_300x300_ssd_iter_140000.caffemodel"
# THRESHOLD = 0.5

# def face_detection(callback=None):
#     net = cv2.dnn.readNetFromCaffe(PROTO_TXT, MODEL)

#     prev_frame_time = 0
#     new_frame_time = 0
#     access = 0

#     cam = cv2.VideoCapture()
#     while True:
#         _, frame = cam.read()
        
#         h, w = frame.shape[:2]
        
#         blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
#         net.setInput(blob)
#         detections = net.forward()
#         for i in np.arange(0, detections.shape[2]):
#             confidence = detections[0, 0, i, 2]
            
#             if confidence < THRESHOLD:
#                 continue
            
#             box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
#             start_x , start_y, end_x, end_y = box.astype("int")
                
#             color =(255, 0, 0)
#             stroke = 3
#             cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), color, stroke)
#             cropped_frame = crop_img(frame, start_x-20, start_y-20, end_x+20, end_y+20)
            

#             if callback:
#                 tensor = callback(frame)
#                 print(tensor.shape)
#                 print(tensor)
#                 cam.release()
#                 cv2.destroyAllWindows()
#                 return


#         new_frame_time = time.time()
#         fps = 1 / (new_frame_time - prev_frame_time)
#         prev_frame_time = new_frame_time
#         fps = str(int(fps))
#         cv2.putText(frame, fps, (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 0), 3, cv2.LINE_AA)
#         """
#         if cv2.waitKey(20) & 0xFF == ord('1'):
#             access = 1
#         if cv2.waitKey(20) & 0xFF == ord('2'):
#             access = 2
#         if cv2.waitKey(20) & 0xFF == ord('3'):
#             access = 0
#         if access == 1:
#             cv2.putText(frame, "User recognized - Access Granted!", (10, 1000), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 3, cv2.LINE_AA)#, bottomLeftOrigin=True)
#         elif access == 2:
#             cv2.putText(frame, "User not recognized - Access Denied!", (10, 1000), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 3, cv2.LINE_AA)#, bottomLeftOrigin=True)
#         """
#         cv2.imshow('Webcam', frame)
#         """
#         take_shot = False
#         if cv2.waitKey(20) & 0xFF == ord('4'):
#             take_shot = True
#         if take_shot:
#             fa = FaceAlignment()
#             aligned_img = fa.align(cropped_frame)
#             img_item = "aligned-image.png"
#             cv2.imwrite(img_item, aligned_img)
#             take_shot = False
#         """
#         if cv2.waitKey(20) & 0xFF == 27:#ord('q'):
#             break

#     cam.release()
#     cv2.destroyAllWindows()


# def detect(image, net):
#     blob = cv2.dnn.blobFromImage(image, 0.007843, (300, 300), 127.5)
#     net.setInput(blob)
#     detections = net.forward()
#     return detections


# def crop_img(img, start_x, start_y, end_x, end_y):
#     height, width = end_y - start_y, end_x - start_x
#     crop_img = img[start_y:start_y+height, start_x:start_x+width]
#     crop_img = cv2.resize(crop_img, (400, 400))
#     return crop_img


# if __name__ == '__main__':
#     face_detection()
#     sys.exit(0)