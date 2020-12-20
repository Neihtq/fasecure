# Input: 
# - path where dataset is stored (LFW, as it has less images per person on average. Thus closer to one-shot learning)
# - Face Detection & Alignment model
# - Face Embedding model
# - RegistrationDatabase

# Procedure:
# 1. Read all images and shuffle them randomly (so that images of same person don´t appear right after each other)
#       Use random seed for shuffling, so that order is always the same
# for-loop (each image)
    # 2. Pass image through whole pipeline (detection & alignment # embedding)
    # 3. Try to recognize person and then adapt TA, TR, ... (for first person, directly register --- for first people, use fixed threshold)
    # 4. Register person to database
    # print every 1000 images intermediate results
# 5. Calculate overall accuracy

# Output: Overall Accuracy

# Questions: 
# - Will it take too long? How to decrease computation time?
#   -> Create intermediate results (e.g. crop all images once and then skip this step, the same if the embedding model if no changes)
#   -> Choose subset instead of complete LFW

# todo:
# - in order to use face detection & alignment, download and setup libraries
# - put data augmentation techniques inside registration process
# - fixed threshold for first 10 people or so

# Own evaluation for one-shot learning:
# - same procedure as above. However, only register person if not already registered (skip registration process)
#    -> register every person just once!


import glob
from os.path import join

# Remove None
def overall_evaluation(target_folder=None, FaceDetectionModel=None, FaceEmbeddingModel=None, RegistrationDatabase=None):
    image_folder = "./evaluation_data/lfw"
    # List containing the paths to all the images
    image_filenames = glob.glob(join(image_folder, "**/*.jpg"), recursive=True)
    
    # Create Dataset
    
    # Create DataLoader with Batchsize 1 (random seed always the same)
    


    return