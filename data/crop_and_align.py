import os
import numpy as np

from PIL import Image
from face_detection.face_alignment import FaceAlignment

#output = "./data/lfw_aligned/"
#output = "./data/vgg_aligned/"

def detect_and_align(pair):
    img_path, output = pair
    head, fpath = os.path.split(img_path)
    _, folder = os.path.split(head)
    dest = os.path.join(output, folder)
    if not os.path.exists(dest):
        try:
            os.mkdir(dest)
        except:
            print(dest, "already exists")

    save_path = os.path.join(dest, fpath)
    if os.path.exists(save_path):
        return
    
    img = np.array(Image.open(img_path))

    fa = FaceAlignment()
    detections = fa.detector(img, 0)
    try:
        x, y, w, h = select_closest_face(detections,  img.shape[:2])
        cropped_img = crop_img(img, x-20, y-20, w+20, h+20)
    except:
        return
    
    try:
        aligned_img = fa.align(cropped_img)
    except:
        aligned_img = cropped_img

    try:
        aligned_img = Image.fromarray(aligned_img)
        aligned_img.save(save_path)
    except:
        pass

def select_closest_face(detections, shape):
    face_dict = {}
    areas = []
    for face in detections:        
        x = face.left()
        y = face.top() 
        w = face.right() - face.left()
        h = face.bottom() - face.top()
        width, height = w +40, h + 40
        area = width * height
        face_dict[area] = (x, y, w, h)
        areas.append(area)

    return face_dict[np.array(areas).max()]

def crop_img(img, x, y, w, h):
    crop_img = img[x-20:x+w+20, y-20:y+h+20]
    crop_img = Image.fromarray(crop_img).resize((224,224))
    return np.array(crop_img)
