# Input: path where dataset is stored (download labeled faces in the wild, as it has less images per person on average)

# Procedure:
# 1. Read all images and shuffle them randomly (so that images of same person don´t appear right after each other)
# for-loop (each image)
    # 2. Pass image through whole pipeline (detection & alignment # embedding)
    # 3. Try to recognize person and then adapt TA, TR, ... (for first person, directly register --- for first people, use fixed threshold)
    # 4. Register person to database

# 5. Calculate overall accuracy

# Output: Overall Accuracy

# Questions: 
# - Will it take too long? How to decrease computation time?
# 

# todo:
# - download dataset
# - in order to use face detection & alignment, download and setup libraries

