import sys
import numpy as np
import torch
import torch.nn.functional as F

from face_detection.face_detection import face_detection
from face_recognition.faceEmbeddingModel import faceEmbeddingModel
from face_recognition.prep import load_and_transform_img, show_tensor_img
from face_recognition.reg_database import RegistrationDatabase
from face_recognition.reg_dataset import RegistrationDataset

def main():
    embedding_model = faceEmbeddingModel()

    reg_dataset = RegistrationDataset("./face_recognition/registered_images", "ppm")
    reg_loader = torch.utils.data.DataLoader(dataset=reg_dataset, batch_size=1,
                                                num_workers=0, shuffle=False)

    database = RegistrationDatabase(embedding_model, reg_loader)
    callback = database.load_and_transform_img
    face_detection(callback=callback)


if __name__ == '__main__':
    main()