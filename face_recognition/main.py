# imports ...
import torch.nn.functional as F
from faceRecognitionModel import faceRecognitionModel
from prep import load_and_transform_img, show_tensor_img

model = faceRecognitionModel()

# adapt folders
path1 = '../facenet-pytorch-master/data/test_images_aligned/faces_db/Lleyton_Hewitt_0003.ppm'
path2 = '../facenet-pytorch-master/data/test_images_aligned/faces_db/Lleyton_Hewitt_0004.ppm'

img1, img2 = load_and_transform_img(path1, path2)

# do forward pass (128 dimensional embedding)
embed1, embed2 = model(img1), model(img2)

# compute the distance using euclidean distance of image embeddings (0 if the same)
euclidean_distance = F.pairwise_distance(embed1, embed2)


# we use 1.5 threshold to decide whether images are genuine or impostor
threshold = 1.5

genuine = euclidean_distance <= threshold

print(genuine)

print(euclidean_distance)

show_tensor_img(img1)