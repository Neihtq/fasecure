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
#                       - face unregistration

# Calculate embeddings list in face_recognition in init and update in face_registration / unregistration and not for every face_recognition
# same for length of embedding list and KNN Definition
# -> pack them into methods!!

# open questions:
# to handle the case, that different amounts of images are availabe for one person, who has to be registered:
# -> batch size one and then store all the embeddings with the same label in a list, then calculate the mean

class RegistrationDatabase():

    # Register people or load registered people
    def __init__(self, faceRecognitionModel, dataloader=None):

        self.model = faceRecognitionModel

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
                img_embedding = self.model(img).detach().cpu().numpy()

                # use img_folder_path to get labels and embeddings
                self.database = self.database.append({'label': label[0], 'embedding': img_embedding}, ignore_index=True)


            # Save it as a pickle file
            self.database.to_pickle(self.database_file)

        # Calculate length of embeddings list and embeddings list itself
        self.len_embeddings_list = len(self.database.index)
        self.embeddings_list = [self.database.iloc[i,1][0] for i in range(self.len_embeddings_list)]
        # self.name_list = np.array([self.database.iloc[i,0] for i in range(self.len_embeddings_list)])

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

    # Either pass image in shape 1x3x244x244 or path to image
    def face_recognition(self, new_img=None, path=None):
        # get new image -> calculate embedding
        if isinstance(new_img, torch.Tensor) and path == None:
            print("Image passed as Tensor")
            img_embedding = self.model(img).detach().cpu().numpy()
        elif new_img == None and isinstance(path, str):
            print("Path passed as String")
            img = self.load_and_transform_img(path)
            img_embedding = self.model(img).detach().cpu().numpy()
        else:
            raise Exception('You have to pass either a img as a tensor (1x3x224x224) or a path (string) where the image is located')

        # Use KNN based on database to find nearest neighbor
        neigh = NearestNeighbors(n_neighbors=5)
        neigh.fit(self.embeddings_list)
        print(neigh.kneighbors(img_embedding))

        label_indices = neigh.kneighbors(img_embedding)[1].tolist()[0]

        nearest_labels = self.database.iloc[label_indices,0]
        print(type(nearest_labels))
        # print(self.name_list[label_indices])
        # print(mode(self.name_list[label_indices]))

        # Calculate distance to nearest neighbor and check, if it´s below threshold

        # return label or unknown

    def face_registration(self, name, reg_img):
        # name: Name of the new person who should be registred
        # reg_img: Tensor (shape 1x3x224x224) of a new person who should be registered

        # Calculate embedding and convert to numpy array
        img_embedding = self.model(img).detach().cpu().numpy()

        # use img_folder_path to get labels and embeddings
        self.database = self.database.append({'label': name, 'embedding': img_embedding}, ignore_index=True)

        # Save it as a pickle file
        self.database.to_pickle(self.database_file)

        # Update length of embeddings list and embeddings list itself
        self.len_embeddings_list = len(self.database.index)
        self.embeddings_list = [self.database.iloc[i,1][0] for i in range(self.len_embeddings_list)]

        
        

 


        
