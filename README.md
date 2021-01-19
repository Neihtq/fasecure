<br />
<p align="center">
  <a href="https://github.com/othneildrew/Best-README-Template">
    <img src="images/logo.png" alt="Logo" width="100" height="100">
  </a>

  <h3 align="center">Face Recognition with FaceNet</h3>

  <p align="center">
    <a href="https://github.com/Neihtq/IBM-labcourse">View Demo</a>
    ·
    <a href="https://github.com/Neihtq/IBM-labcourse/issues">Report Bug</a>
    ·
    <a href="https://github.com/Neihtq/IBM-labcourse/issues">Request Feature</a>
  </p>
</p>



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
    <li><a href="#training-data">Training Data</a></li>
    <li>
    <a href="#training">Training</a>
        <ul>
            <li><a href="#triplet-loss-training">Triplet Loss Training</a></li>
        </ul>
        <ul>
            <li><a href="#face-embedding-facenet-with-resnet50-backbone">Face Embedding FaceNet with ResNet50 backbone</a></li>
        </ul>
        <ul>
            <li><a href="#face-verification-with-knn-and-adaptive-thresholding
">Face Face Verification with KNN and Adaptive Thresholding</a></li>
        </ul>
    </li>
    <li><a href="#performance">Performance</a></li>
    <li><a href="#issues">Issues</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgements">Acknowledgements</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
# About The Project

This project was developed in the context of the advanced practical course "Application Challenges for Machine Learning on the example of IBM Power AI" at the Technical University of Munich. Our task was to build a complete face recognition system.

During the project we came up with an additional pseudo real world use case. The idea is to use face recognition to let the a computer or laptop automatically lock if the registered owneder does not appear in front of the camera within a given time frame. While this makes it more comfortable for the user, it is also helpful against data theft.

## Face Recognition Pipeline
The main focus of this project is implementation of the face verification. For that we used the implementation described in the paper ["FaceNet: A Unified Embedding for Face Recognition and Clustering"](https://arxiv.org/abs/1503.03832). 

Additionally, we implemented all tasks in which face recognition can be broken down to: Detection, Alignment, Embedding and Verification.


## Built With
* [Python](https://www.python.org/)
* [PyTorch](https://pytorch.org/)
* [PyTorchLightning](https://www.pytorchlightning.ai/)
* [Pytorch Metric Learning](https://github.com/KevinMusgrave/pytorch-metric-learning)
* [NumPy](https://numpy.org/)
* [pandas](https://pandas.pydata.org/)
* [OpenCV](https://github.com/skvark/opencv-python)

* [PyQt](https://www.riverbankcomputing.com/software/pyqt/)
* [Flask](https://flask.palletsprojects.com/en/1.1.x/)


# Getting Started
```sh 
git clone https://github.com/Neihtq/IBM-labcourse.git
``` 
and you are ready to go.

## Prerequisites
You can install all dependencies easily with [pip](https://pypi.org/project/pip/).
Simply run:

```sh
pip install -r requirements.txt
```

# Training data
The [Labeled Faces in the Wild](http://vis-www.cs.umass.edu/lfw/) dataset has been used for training. Thie data set consists of total of 13233 images over 5749 identities. Additionally, we preprocessed the imags with [DeepFace](https://github.com/serengil/deepface) beforehand cropping and aligning the faces on each image.

This repository provides a script for downloading the processed data. It can be executed by running:

```sh
python download_LFW_aligned.py
```

# Training

## Triplet loss training

The whole architecture is based on a siamese network and one shot learning with online triplet mining. Therefore we use the [Triplet Loss Function](https://pytorch.org/docs/stable/generated/torch.nn.TripletMarginLoss.html) in order for the network to create accurate face embeddings.

### Dataset structure

1. It is assumed that the training dataset is arrangedas below, i.e. where each class is a subdirectory containing the training samples belong to that class.
```
Aaron_Eckhart
    Aaron_Eckhart_001.jpg

Aaron_Guiel
    Aaron_Guiel_0001.jpg

Aarong_Patterson
    Aaron_Patterson_0001.jpg

Aaron_Peirsol
    Aaron_Peirsol_0001.jpg
    Aaron_Peirsol_0002.jpg
    Aaron_Peirsol_0003.jpg
    ...
```
2. The ```main.py``` script will initiate the training. Following flags are available to setup the process:
```
--num-epochs: Number of epochs to train(default: 200)

--batch-size: Batch Size for dataset (default: 220)

--num-workers: Number of worker for DataLoader (default: 0)

--learning-rate: Learning Rate (default: 0.001)

--margin: Margin for TripletLoss (default: 0.5)

--train-data-dir: Path to dataset (default: ```'./data/images/lfw_crop'```)

--model-dir: Path where trained model should be save (default: ```./models/results```)

--weight-decay: Decay learning rate (default: 0)

--pretrained: Will load pretrained model, if set.

--load-last: Start training from last checkpoint, if set.

--no-pytorch-lightning: Will perform regular training without PyTorchLightning, if set,

```

3. Execute the script with desired parameters, e.g.:
```sh
python main.py --num-epochs 175 --batch-size 128 learning-rate 0.005 --train-data-dir ./data/images/lfw_crop
```

## Face Embedding FaceNet with ResNet50 backbone

The FaceNet architecture includes a deep neural network followed by a 128 dimensional linear layer. We chose for the deep neural network [ResNet50](https://arxiv.org/abs/1512.03385), which also already included in PyTorch.

## Face Verification with KNN and Adaptive Thresholding
TBD

# Performance
TBD


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


## Acknowledgements
* Special thanks to Omar Shouman for giving his best effort to support us during this course
* Special thanks for IBM to provides us with the IBM POWER Architecture for training
* [Face Recognition using Tensorflow](https://github.com/davidsandberg/facenet)
* [Best-README-Template](https://github.com/othneildrew/Best-README-Template)
* [Pretrained Weight for Face Recognition, also heavily inspired in terms of building the archecture](https://github.com/tbmoon/facenet)