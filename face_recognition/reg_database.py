# import libraries
import pandas as pd
import os
import torch
from torchvision import transforms
from PIL import Image
from sklearn.neighbors import NearestNeighbors
import numpy as np
from statistics import mode

# - implement database class which stores the labels and embeddings (pandas dataframe)
#       attibutes: pandas dataframe, faceRecognitionModel
#       main methods:   - constructor
#                       - face recognition
#                       - face registration (one-shot learning)
#                       - face deregistration

# open questions:
# to handle the case, that different amounts of images are availabe for one person in the registration process:
# -> batch size one and then store all the embeddings with the same label in a list, then calculate the mean

class RegistrationDatabase():

    # Register people or load registered people
    def __init__(self, faceEmbeddingModel, dataloader=None):

        # Set model to eval, as training is over when we use it here for inference
        self.embedding_model = faceEmbeddingModel.eval()
        self.recognition_model = NearestNeighbors(n_neighbors=5)

        # have a look if database (pickle file) already available, if yes, load it and save it into pandas dataframe and return
        model_dir = "./reg_database"
        os.makedirs(model_dir, exist_ok=True)

        self.database_file = os.path.join(model_dir, 'database.pkl')

        # If no dataloader specified, then try to load database
        if dataloader == None:
            # If path exists, then load pickle file and safe it into pandas dataframe
            # (Saves file according to specified path. If want to make sure that registration is reloaded, then delete DB)
            if os.path.exists(self.database_file):
                # load pickle file and save it into class attribute database
                print('Database already exists. Pickle file will be loaded...')
                self.database = pd.read_pickle(self.database_file)
            else: 
                raise Exception('No database availabe. You have to specifiy a dataloader containing all the images for registration (Batch-size: 1)')


        # If dataloader specified, then overwrite database (or create a new one, if none existing)
        else:
            print('A dataloader was specified. A new pickle file containing the images in the dataloader will be created...')

            # create pandas dataframe 
            self.database = pd.DataFrame(columns=['label','embedding'])

            for i, data in enumerate(dataloader):
                img, label = data

                # if torch.cuda.is_available():
                #     img = img.to("cuda")
                #     label = target.to("cuda")

                # Calculate embedding and convert to numpy array
                # detach() to remove computational graph from tensor
                img_embedding = self.calculate_embedding(img)

                # use img_folder_path to get labels and embeddings
                self.database = self.database.append({'label': label[0], 'embedding': img_embedding}, ignore_index=True)


            # Save it as a pickle file
            self.save_database()

        # Calculate length of embeddings list and embeddings list itself
        self.update_embeddings()

        # In the end after running constructor: saved database as pickle file and database attribute contains registration database

    def load_and_transform_img(self, path):
        #prepare preprocess pipeline
        preprocess_pipelines = [transforms.Resize(224),  
                            transforms.ToTensor(), 
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                    std=[0.229, 0.224, 0.225])]

        trfrm = transforms.Compose(preprocess_pipelines)

        # read the image and transform it into tensor then normalize it with our trfrm function pipeline
        img = trfrm(Image.open(path)).unsqueeze(0)
    
        return img

    # Save database as pickle file
    def save_database(self):
        self.database.to_pickle(self.database_file)

    # Update length of embedding list and embedding list itself
    def update_embeddings(self):
        self.len_embeddings_list = len(self.database.index)
        self.embeddings_list = [self.database.iloc[i,1][0] for i in range(self.len_embeddings_list)]
        # self.name_list = np.array([self.database.iloc[i,0] for i in range(self.len_embeddings_list)])
        self.recognition_model.fit(self.embeddings_list)

    # Get tensor img as input and output numpy array of embedding
    def calculate_embedding(self, img):
        return self.embedding_model(img).detach().cpu().numpy()

    # Either pass image in shape 1x3x244x244 or path to image
    def face_recognition(self, new_img=None, path=None):
        # get new image -> calculate embedding
        if isinstance(new_img, torch.Tensor) and path == None:
            print("Image passed as Tensor")
            img_embedding = self.calculate_embedding(new_img)
        elif new_img == None and isinstance(path, str):
            print("Path passed as String")
            new_img = self.load_and_transform_img(path)
            img_embedding = self.calculate_embedding(new_img)
        else:
            raise Exception('You have to pass either a img as a tensor (1x3x224x224) or a path (string) where the image is located')

        # Use KNN based on database to find nearest neighbor
        print(self.recognition_model.kneighbors(img_embedding))

        label_indices = self.recognition_model.kneighbors(img_embedding)[1].tolist()[0]
        nearest_labels = self.database.iloc[label_indices,0]
        print(nearest_labels)

        # Calculate distance to nearest neighbor and check, if it´s below threshold
        closest_embedding_dist = self.recognition_model.kneighbors(img_embedding)[0].tolist()[0][0]
        print("Closest embedding: ", closest_embedding_dist)

        if closest_embedding_dist > 1.5:
            print("Unknown person")

        # return label or unknown

    def face_registration(self, name, reg_img):
        # name: Name of the new person who should be registred
        # reg_img: Tensor (shape 1x3x224x224) of a new person who should be registered

        # Check, if name already in database available. If yes, then return
        # Extend and check, if embedding already in database: distance to Nearest Neighbor should be at least 1.5? Possible at all?
        if (self.database['label'] == name).any():
            print('Specified name already in database registered. User can not be registered again!')
            return            

        # Calculate embedding and convert to numpy array
        img_embedding = self.calculate_embedding(reg_img)

        # use img_folder_path to get labels and embeddings
        self.database = self.database.append({'label': name, 'embedding': img_embedding}, ignore_index=True)

        # Save it as a pickle file
        self.save_database()

        # Update length of embeddings list and embeddings list itself
        self.update_embeddings()

        print(self.database)

    def face_deregistration(self, name):
        # name: Name of the person who should be deregistered

        # delete all entries in database with specified name
        drop_indices = self.database[ self.database['label'] == name ].index
        if len(drop_indices) == 0:
            print('Specified name not in database registered. User can not be deregistered!')
            return
        self.database.drop(drop_indices, inplace=True)

        # Save it as a pickle file
        self.save_database()

        # Update length of embeddings list and embeddings list itself
        self.update_embeddings() 

        print(self.database)
        
        

 


        
