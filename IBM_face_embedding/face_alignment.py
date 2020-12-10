from deepface import DeepFace
import matplotlib.pyplot as plt

#takes as input an image and rotates the image as much necessary that you could draw a straight line between the two eyes
#returns a 224 x 224 x 3 image

# Requirement for the library is the the faces are not already completly cropped out, however just a face detector which shows whole head would be fine

def face_alignment(imgname):
    detected_face = DeepFace.detectFace(imgname)
    #plt.imshow(detected_face)
    return detected_face


# Example:
#plt.imshow(facealignment('/exampleimg.jpg'))
