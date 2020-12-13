# import libraries
import pandas as pd
import os
import torch
from PIL import Image
from sklearn.neighbors import NearestNeighbors
import numpy as np
from statistics import mode

import sys

# - implement database class which stores the labels and embeddings (pandas dataframe)
#       attibutes: pandas dataframe, faceRecognitionModel
#       main methods:   - constructor
#                       - face recognition
#                       - face registration (one-shot learning)
#                       - face deregistration

# with newer torchvision version, one can also transform tensor batches (but cannot update torchvision)
# Thus, I have to convert it to an PIL image first

class RegistrationDatabase():

    # Register people or load registered people
    def __init__(self, dataloader=None, mode='inner_product'):

        # Set model to eval, as training is over when we use it here for inference
        ###self.embedding_model = faceEmbeddingModel.eval()
        
        # Choose similarity calculation between "inner product" and "euclidean distance"
        self.mode = mode
        if self.mode == 'euclidean_distance':
            self.recognition_model = NearestNeighbors(n_neighbors=1)

        

        self.len_embeddings_list = 0

        ###self.pil_transforms = transforms.Compose([
        ###    transforms.Resize(224),
        ###   transforms.ToTensor(),
        ###    transforms.Normalize(mean=[0.485, 0.456, 0.406],
        ###                         std=[0.229, 0.224, 0.225])])

        ###self.augmentation_1 = transforms.Compose([
        ###    transforms.ToPILImage(),
        ###    transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0),
        ###    transforms.ToTensor()])       

        ###self.augmentation_2 = transforms.Compose([
        ###    transforms.ToPILImage(),
        ###    transforms.RandomHorizontalFlip(p=1),
        ###    transforms.ToTensor()])   

        ###self.augmentation_3 = transforms.Compose([
        ###    transforms.ToPILImage(),
        ###    transforms.RandomPerspective(distortion_scale=0.1, p=1),
        ###    transforms.ToTensor()]) 
                             

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
                # For the case, that the pickle file was changed externally
                self.update_embeddings()
            else: 
                print("No database availabe. Empty database will be created...")
                self.database = pd.DataFrame(columns=['label','embedding','threshold'])


        # If dataloader specified, then overwrite database (or create a new one, if none existing)
        else:
            print('A dataloader was specified. A new database file containing the images in the dataloader will be created...')

            # create pandas dataframe 
            self.database = pd.DataFrame(columns=['label','embedding','threshold'])

            for i, data in enumerate(dataloader):
                img, label = data

                # if torch.cuda.is_available():
                #     img = img.to("cuda")
                #     label = target.to("cuda")

                # Face registration
                self.face_registration(label[0], img)


            # Save it as a pickle file
            self.save_database()

        # In the end after running constructor: saved database as pickle file and database attribute contains registration database

    ###def load_and_transform_img(self, path):
        # read the image and transform it into tensor then normalize it with our trfrm function pipeline
    ###    img = self.pil_transforms(Image.open(path)).unsqueeze(0)
    ###    return img

    # Save database as pickle file
    def save_database(self):
        self.database.to_pickle(self.database_file)

    # Update length of embedding list and embedding list itself
    def update_embeddings(self):
        self.len_embeddings_list = len(self.database.index)
        self.embeddings_list = [self.database.iloc[i,1][0] for i in range(self.len_embeddings_list)]
        # self.name_list = np.array([self.database.iloc[i,0] for i in range(self.len_embeddings_list)])

        # Calculate and update inner product thresholds (+add pseudo embeddings to avoid problem with to less registered embeddings for adaptive threshold)
        # Adapt threshold for first embedding
        if self.database['label'].nunique() == 1:
            self.database.iloc[:,2] = 98.0
            #print(self.database)
        elif self.database['label'].nunique() > 1:
            # Loop over all embeddings in the database
            for i in range(self.len_embeddings_list):

                # Calculate the similarity score between a selected embedding and all the other embeddings
                temp_embeddings_list = self.embeddings_list.copy()
                temp_embedding = temp_embeddings_list[i]

                # get label from index, then get all indices from database with that label
                cur_label = self.get_label([i])
                cur_label_indices = self.database[self.database['label']==cur_label].index.values.astype(int).tolist()

                # print("current label: ", cur_label)
                # print("current label indices: ", cur_label_indices)

                # delete all these labels from temp_embeddings_list (compare to all others, but not belonging to the same person)
                # reverse order so that one don´t throw off the subsequent indices
                for index in sorted(cur_label_indices, reverse=True):
                    del temp_embeddings_list[index]


                if self.mode == 'inner_product':
                    # Inner product is 100, when two vectors are identical (as vectors lie on a hypersphere scaled by alpha=10 -> length(vector)^2)
                    inner_products = np.inner(temp_embedding,temp_embeddings_list)

                    # Set the inner product threshold of the corresponding embedding...
                    # as the maximum value among all facial embeddings not belonging to the same person                  
                    self.database.iloc[i,2] = np.max(inner_products) 

                elif self.mode == 'euclidean_distance':
                    # print(type(temp_embedding))
                    # print(temp_embedding.reshape((1,128)).shape)
                    # sys.exit()
                    self.recognition_model.fit(temp_embeddings_list)
                    closest_embedding_dist = self.recognition_model.kneighbors(temp_embedding.reshape((1,128)))[0].tolist()[0][0]
                    self.database.iloc[i,2] = closest_embedding_dist


        if self.len_embeddings_list > 0 and self.mode == 'euclidean_distance':
            self.recognition_model.fit(self.embeddings_list)

    # Get tensor img as input and output numpy array of embedding
    def convert_to_numpy(self, img_embedding_tensor):
        # detach() to remove computational graph from tensor
        return img_embedding_tensor.detach().cpu().numpy()

    # Get the label of the person by providing the index of the dataframe
    def get_label(self, index):
        label = self.database.iloc[index,0].reset_index(drop=True)[0]
        return label

    def get_similarity_threshold(self, index):
        similarity_threshold = self.database.iloc[index,2].reset_index(drop=True)[0]
        return similarity_threshold
    
    # Find closest embedding based on euclidean distance (use KNN with k=1) and fixed threshold
    # Can also adapt to adaptive threshold (see paper)
    def closest_embedding_euclidean_distance(self, img_embedding):
        # print(self.recognition_model.kneighbors(img_embedding))

        label_index = self.recognition_model.kneighbors(img_embedding)[1].tolist()[0]
        closest_label = self.get_label(label_index)
        #print("Closest person: ", closest_label)

        # Calculate distance to nearest neighbor and check, if it´s below threshold
        max_similarity = self.recognition_model.kneighbors(img_embedding)[0].tolist()[0][0]
        # print("Closest embedding: ", max_similarity)
        similarity_threshold = self.get_similarity_threshold(label_index)

        # Here it´s the opposite of the inner product
        if max_similarity <= similarity_threshold:
            check = "Access"
        else:
            check = "Intruder"

        return closest_label, check

    # Find closest embedding based on inner product and adaptive thresholds and thus decide, if person known or unknown
    def closest_embedding_inner_product(self, img_embedding):
        # calculate the inner product to all other embeddings
        inner_products = np.inner(img_embedding,self.embeddings_list)

        # Get index with hightest value (which is the closest vector) and convert it into a list (so get_label works for knn and inner product)
        label_index = [np.argmax(inner_products)]

        # Use index to get the label
        closest_label = self.get_label(label_index)
        #print("Closest person: ", closest_label)

        # Check, if the maximal computed similarity higher is then the threshold similarity of that person
        # If yes, then it is that person. Otherwise, it´s an intruder and the authentification request will be rejected
        max_similarity = np.max(inner_products)
        similarity_threshold = self.get_similarity_threshold(label_index)
        # print("sim thres of closest person: ", similarity_threshold)
        # print("sim: ", max_similarity)
        if max_similarity >= similarity_threshold:
            check = "Access"
        else:
            check = "Intruder"

        return closest_label, check
        
    # Find the closest person in the embedding space and decide then, whether access or intruder
    def face_recognition(self, img_embedding_tensor):
        
        if self.len_embeddings_list == 0:
            print("Person is unkown")
            return
        
        # get new image -> calculate embedding
        ###if isinstance(new_img, torch.Tensor) and path == None:
        ###    print("Image passed as Tensor")
        ###    img_embedding = self.calculate_embedding(new_img)
        ###elif new_img == None and isinstance(path, str):
        ###    print("Path passed as String")
        ###    new_img = self.load_and_transform_img(path)
        ###    img_embedding = self.calculate_embedding(new_img)
        ###else:
        ###    raise Exception('You have to pass either a img as a tensor (1x3x224x224) or a path (string) where the image is located')

        img_embedding_numpy = self.convert_to_numpy(img_embedding_tensor)

        # Use KNN based on database to find nearest neighbor (with fixed threshold)
        if self.mode == 'inner_product':
            closest_label, check = self.closest_embedding_inner_product(img_embedding_numpy)
        elif self.mode == 'euclidean_distance':
            closest_label, check = self.closest_embedding_euclidean_distance(img_embedding_numpy)
        

        return closest_label, check

    def face_registration(self, name, img_embedding_tensor):
        # name: Name of the new person who should be registred
        # reg_img: Tensor (shape 1x3x224x224) of a new person who should be registered

        # Check, if name already in database available. If yes, then return
        # Extend and check, if embedding already in database: distance to Nearest Neighbor is 0, then image already exists. 
        # if (self.database['label'] == name).any():
        #     print('Specified name already in database registered. User can not be registered again!')
        #     return            

        # Data augmentation: random noise, horizontal flip
        ###reg_img_1 = reg_img
        #####reg_img_2 = self.augmentation_1(reg_img.squeeze(0)).unsqueeze(0)
        ###reg_img_3 = self.augmentation_2(reg_img.squeeze(0)).unsqueeze(0)
        ###reg_img_4 = self.augmentation_3(reg_img.squeeze(0)).unsqueeze(0)

        # Calculate embedding and convert to numpy array
        img_embedding_numpy = self.convert_to_numpy(img_embedding_tensor)
        ###img_embedding_2 = self.calculate_embedding(reg_img_2)
        ###img_embedding_3 = self.calculate_embedding(reg_img_3)
        ###img_embedding_4 = self.calculate_embedding(reg_img_4)

        # Add label, embedding and threshold to database (threshold first of all set to 0, will be udpated later on)
        self.database = self.database.append({'label': name, 'embedding': img_embedding_numpy, 'threshold': 0}, ignore_index=True)
        ###self.database = self.database.append({'label': name, 'embedding': img_embedding_2, 'threshold': 0}, ignore_index=True)
        ###self.database = self.database.append({'label': name, 'embedding': img_embedding_3, 'threshold': 0}, ignore_index=True)
        ###self.database = self.database.append({'label': name, 'embedding': img_embedding_4, 'threshold': 0}, ignore_index=True)

        # Update length of embeddings list and embeddings list itself
        self.update_embeddings()

        # Save it as a pickle file
        self.save_database()

    def face_deregistration(self, name):
        # name: Name of the person who should be deregistered

        # delete all entries in database with specified name
        drop_indices = self.database[ self.database['label'] == name ].index
        if len(drop_indices) == 0:
            print('Specified name not in database registered. User can not be deregistered!')
            return
        # print(drop_indices)
        self.database.drop(drop_indices, inplace=True)
        # reset index, so that it counts again from zero if person is deregistered from the middle
        self.database.reset_index(drop=True,inplace=True)

        # Save it as a pickle file
        self.save_database()

        # Update length of embeddings list and embeddings list itself
        self.update_embeddings() 
        
        

 


        
