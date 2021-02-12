<br />
<p align="center">
  <a href="https://github.com/othneildrew/Best-README-Template">
    <img src="images/logo.png" alt="Logo" width="200" height="200">
  </a>

  <h3 align="center">Fasecure</h3>

  <p align="center">
    <a href="https://github.com/Neihtq/IBM-labcourse">View Demo</a>
    ·
    <a href="https://github.com/Neihtq/IBM-labcourse/issues">Report Bug</a>
    ·
    <a href="https://github.com/Neihtq/IBM-labcourse/issues">Request Feature</a>
  </p>
</p>


- Our trained model can be downloaded [here](https://drive.google.com/file/d/1FYhgSwUQyOr9JtHUA8JhXWQYFUQ3Ll6Y/view?usp=sharing)
- The pretrained model can be downloaded [here](https://github.com/tamerthamoqa/facenet-pytorch-vggface2)
- The dataset we used for evaluation can be downloaded [here](https://drive.google.com/file/d/1YMmTYqmHnpcnRb5mdd1eUYNW1lADKt89/view?usp=sharing)


<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
      <ul>
        <li><a href="#face-recognition-pipeline">Face Recognition Pipeline</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
      </ul>
    </li>
    <li>
    <a href="#training-and-evaluation">Training and Evaluation</a>
        <ul>
            <li><a href="#training-data">Training Data</a></li>
        </ul>
        <ul>
            <li><a href="#training">Training</a></li>
        </ul>
        <ul>
            <li><a href="#evaluation-data">Evaluation Data</a></li>
        </ul>
        <ul>
            <li><a href="#evaluation">Evaluation</a></li>
        </ul>
        <ul>
            <li><a href="#face-spoofing">Face Spoofing</a></li>
        </ul>
    </li>
    <li><a href="#running-fasecure">Running Fasecure</a>
      <ul>
        <li><a href="#backend">Backend</a></li>
      </ul>
      <ul>
        <li><a href="#frontend">Frontend</a></li>
      </ul>
    </li>
    <li><a href="#issues">Issues</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgements">Acknowledgements</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
# About The Project
![Product Name Screen Shot][product-screenshot]
Facsecure is an application that simulates an access control system through face recognition. This project provides the whole training pipeline for training a model with an own selected dataset. On top of this, an application utilizes the model as the core of the facial recognition backend logic.

Fasecure was developed in the context of the advanced practical course "Application Challenges for Machine Learning on the example of IBM Power AI" at the Technical University of Munich. Our main task was to build a complete facial recognition system.

## Face Recognition Pipeline
The main focus of this project is implementation of the face recognition. For that we used the implementation described in the paper ["FaceNet: A Unified Embedding for Face Recognition and Clustering"](https://arxiv.org/abs/1503.03832). 

Additionally, we implemented all tasks in which face recognition can be broken down to: Detection, Alignment, Embedding and Recognition/Registration.


## Built With
* [Python3](https://www.python.org/)
* [PyTorch](https://pytorch.org/)
* [PyTorchLightning](https://www.pytorchlightning.ai/)
* [NumPy](https://numpy.org/)
* [pandas](https://pandas.pydata.org/)
* [OpenCV](https://github.com/skvark/opencv-python)
* [dlib](https://github.com/davisking/dlib)
* [PySimpleGUI](https://pysimplegui.readthedocs.io/en/latest/)
* [Flask](https://flask.palletsprojects.com/en/1.1.x/)


# Getting Started
```sh 
$ git clone https://github.com/Neihtq/IBM-labcourse.git
``` 
and you are ready to go.

## Prerequisites
You can install all dependencies easily with [pip](https://pypi.org/project/pip/).
Simply run:

```sh
$ pip install -r requirements.txt
```
followed by 
```sh
$ cd backend
$ pip install -e .
```
to install the Fasecure model.

Also make sure to have a working webcam.

# Training and Evaluation

## Training data
The [VGG Face Dataset](https://www.robots.ox.ac.uk/~vgg/data/vgg_face/) consists of multiple images from 2622 distinct identities. Overall the dataset took 69 GB of storage. Triplets were generated and fed into [Triplet Loss Function](https://pytorch.org/docs/stable/generated/torch.nn.TripletMarginLoss.html) for learning.

### Training

Please refer to the [wiki page](https://github.com/Neihtq/facesecure/wiki/Training) on how to train the model.


## Evaluation Data
[Labeled Face in the Wild](http://vis-www.cs.umass.edu/lfw/) was used evaluating both embedding and recognition pipeline. On our local machines, we used [DeepFace](https://github.com/serengil/deepface) beforehand for cropping and aligning the faces on each image.

### Evaluation

Please refer to the [wiki page](https://github.com/Neihtq/facesecure/wiki/Evaluation) on how to evaluate the pipeline and its model.

## Face Spoofing

We came also up with the idea of integrating face spoofing/liveness detection into the pipeline. However, we did not have enough time to develop a model with sufficient accuracy. Nonetheless, the face spoofing module can be tested sperately:

```sh
$ cd backend
$ python face_recognition/face_spoofing.py
```

# Running Fasecure

## Backend
Run:
 ```sh
 $ cd backend
 $ python server.py
 ``` 

## Frontend
Run:
```sh
$ cd frontend
$ python view.py
``` 


# Issues

See the [open issues](hhttps://github.com/Neihtq/IBM-labcourse/issues) for a list of proposed features (and known issues).


<!-- LICENSE -->
# License

Distributed under the MIT License. See `LICENSE` for more information.

<!-- CONTACT -->
# Contact

Cao Son Ngoc Pham - [@cacao](https://github.com/xcacao) - caoson@hotmail.de -
<a href="https://www.linkedin.com/in/xcacao/">
    <img height=17 src="https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white" />
</a>

Quang Thien Nguyen - [@Neihtq](https://github.com/Neihtq) - q.thien.nguyen@outlook.de - <a href="https://www.linkedin.com/in/thien-quang-nguyen-808101143/">
    <img height=17 src="https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white" />
</a>

Simon Felderer - [@simonfelderer](https://github.com/simonfelderer) - simon.felderer@tum.de - <a href="https://www.linkedin.com/in/simon-felderer-976b9b154/">
    <img height=17 src="https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white" />
</a>

Tobias Zeulner - [@Zeulni](https://github.com/Zeulni) - ge93yan@mytum.de - <a href="https://www.linkedin.com/in/tobias-zeulner-893080169/">
    <img height=17 src="https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white" />
</a>


# Acknowledgements
* Special thanks to [Omar Shouman](https://www.linkedin.com/in/omar-shouman/) for giving his best effort to support us during this course
* Special thanks for IBM to provides us with the IBM POWER Architecture for training
* [Face Recognition using Tensorflow](https://github.com/davidsandberg/facenet)
* [Best-README-Template](https://github.com/othneildrew/Best-README-Template)
* We were heavily inspired by [tamerthamoqa's implementation](https://github.com/tamerthamoqa/facenet-pytorch-vggface2)

[product-screenshot]: images/preview_screenshot.png
