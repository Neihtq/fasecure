import os
import pathlib  
import argparse
import tqdm
import numpy as np
import multiprocessing as mp

from PIL import Image
from face_detection.face_alignment import FaceAlignment

try:
    from deepface import DeepFace
except ImportError:
    raise ImportError("Could not import deepface")

try:
    from face_detection.input_pipeline import detect, crop_img, THRESHOLD
except ImportError:
    raise ImportError("Could not import CV2 face detector")

parser = argparse.ArgumentParser(description='Detect, crop and align faces.')

parser.add_argument('--dir', type=str, help='Directory of images')

parser.add_argument('--o', type=str, help='Directory where aligned images will be saved.')

parser.add_argument('--deepface', action='store_true', help='User deepface library.')

args = parser.parse_args()


def deepface_align(pair):
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

    try:
        detected_face = DeepFace.detectFace(img_path, detector_backend='mtcnn')
        img = Image.fromarray(np.uint8(detected_face*255)).convert('RGB')
    except:
        img = Image.open(img_path)
   
    img.save(save_path)


def detect_and_align_cv2(pair):
    img_path, output = pair
    try:
        img = Image.open(img_path).convert("RGB")
        img = np.array(img)
    except:
        return
                     
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
    detections = detect(img)
    try:
        x, y, w, h = select_closest_face(detections,  img.shape[:2], use_cv2=True)
        cropped_img = crop_img(img, x-20, y-20, w+20, h+20)
    except:
        cropped_img = img
    
    try:
        aligned_img = fa.align(cropped_img)
    except:
        aligned_img = cropped_img

    try:
        aligned_img = Image.fromarray(aligned_img)
        aligned_img.save(save_path)
    except:
        Image.open(img_path).save(save_path)


def detect_and_align(pair):
    img_path, output = pair
    try:
        img = Image.open(img_path).convert("RGB")
        img = np.array(img)
    except:
        return
                     
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


def select_closest_face(detections, shape, use_cv2=False):
    face_dict = {}
    areas = []
    height, width = shape

    if use_cv2:
        for i in np.arange(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            if confidence < THRESHOLD:
                continue
            
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            start_x , start_y, end_x, end_y = box.astype("int")
            height, width = end_y - start_y, end_x - start_x
            area = height * width
            face_dict[area] = (start_x, start_y, end_x, end_y)
            areas.append(area)
    else:
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


def main():
    src = args.dir
    
    output = pathlib.Path(args.o)
    output.mkdir(parents=True, exist_ok=True)
    
    deepface = args.deepface

    paths = []
    for folder in os.listdir(src):
        tmp = os.path.join(src, folder)
        dest = os.path.join(output, folder)
        pathlib.Path(dest).mkdir(parents=True, exist_ok=True)

        for img in os.listdir(tmp):
            fpath = os.path.join(tmp, img)
            paths.append((fpath, str(output)))

    with mp.Pool(processes=os.cpu_count()) as pool:
        if deepface:
            res = list(tqdm.tqdm(pool.imap(deepface_align, paths), total=len(paths)))
        else:
            res = list(tqdm.tqdm(pool.imap(detect_and_align_cv2, paths), total=len(paths)))
        
    for folder in os.listdir(output):
        tmp = os.path.join(output, folder)
        if len(os.listdir(tmp)) == 0:
            os.rmdir(tmp)


if __name__ == '__main__':
    main()