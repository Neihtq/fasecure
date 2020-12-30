# import libraries
import pandas as pd
import os
import torch
from PIL import Image
from sklearn.neighbors import NearestNeighbors
import numpy as np

# input: 128 dim embedding as tensor (convert it internally to numpy array)
#               - registration: embedding + name
#               - deregistration: name
#               - recognition: embedding
# ---------------------------------------------------------------------------
# functions:    - registration
#               - deregistration
#               - recognition
#               - clean_database
# ---------------------------------------------------------------------------
# output:       - registration: "registered successfully"
#               - deregistration: "deregistered successfully"
#               - recognition: closest person + access/intruder


class RegistrationDatabase():

    # Initialize database
    def __init__(self, fixed_threshold, mode='inner_product'):
        
        # Choose similarity calculation between "inner product" and "euclidean distance"
        self.mode = mode
        if self.mode == 'euclidean_distance':
            self.recognition_model = NearestNeighbors(n_neighbors=1)
        
        # ----- Define default value for it
        self.fixed_threshold = fixed_threshold
      
        self.len_embeddings_list = 0
                             
        model_dir = "./reg_database"
        os.makedirs(model_dir, exist_ok=True)

        self.database_file = os.path.join(model_dir, 'database.pkl')

        # If path exists, then load pickle file and safe it into pandas dataframe
        # (Saves file according to specified path. If want to make sure that registration is reloaded, then delete database first)
        if os.path.exists(self.database_file):
            # load pickle file and save it into class attribute database
            print('Database already exists. Pickle file will be loaded...')
            self.database = pd.read_pickle(self.database_file)
            # For the case, that the pickle file was changed externally
            self.update_embeddings()
        else: 
            print("No database availabe. Empty database will be created...")
            self.database = pd.DataFrame(columns=['label','embedding','threshold'])


            # Save it as a pickle file
            self.save_database()

        # In the end after running constructor: loaded availabe database or created a new one

    # Save database to pickle file
    def save_database(self):
        self.database.to_pickle(self.database_file)

    def clean_database(self):
            if os.path.exists(self.database_file):
                print('database.pkl exists and will be cleaned...')
                self.database = pd.DataFrame(columns=['label','embedding','threshold'])
                self.update_embeddings()
                # Save it as a pickle file
                self.save_database()
            else: 
                print("No database.pkl file exists. Hence, it cannot be cleaned...")      

    # Checks, if database contains specified label
    def contains(self, label):
        # delete all entries in database with specified name
        label_indices = self.database[ self.database['label'] == label ].index
        if len(label_indices) == 0:
            return False
        elif len(label_indices) > 0:
            return True

    # Update length of embedding list, embedding list itself and the thresholds of every person
    def update_embeddings(self):
        self.len_embeddings_list = len(self.database.index)
        self.embeddings_list = [self.database.iloc[i,1][0] for i in range(self.len_embeddings_list)]
        # self.name_list = np.array([self.database.iloc[i,0] for i in range(self.len_embeddings_list)])

        # Calculate and update inner product thresholds
        # Adapt threshold for first embedding (can be deleted?)
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
                    closest_embedding_dist = np.max(inner_products) 
                    if closest_embedding_dist > self.fixed_threshold:
                        self.database.iloc[i,2] = closest_embedding_dist
                    else:
                        self.database.iloc[i,2] = self.fixed_threshold

                elif self.mode == 'euclidean_distance':
                    # print(type(temp_embedding))
                    # print(temp_embedding.reshape((1,128)).shape)
                    # sys.exit()
                    self.recognition_model.fit(temp_embeddings_list)
                    print("----------- Fixed threshold not defined so far! --------------")

                    closest_embedding_dist = self.recognition_model.kneighbors(temp_embedding.reshape((1,128)))[0].tolist()[0][0]
                    self.database.iloc[i,2] = closest_embedding_dist


        if self.len_embeddings_list > 0 and self.mode == 'euclidean_distance':
            self.recognition_model.fit(self.embeddings_list)

    # Get torch tensor embedding as input and outputs numpy embedding
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
    
    # Find closest embedding based on euclidean distance (use KNN with k=1)
    def closest_embedding_euclidean_distance(self, img_embedding):

        label_index = self.recognition_model.kneighbors(img_embedding)[1].tolist()[0]
        closest_label = self.get_label(label_index)

        # Calculate distance to nearest neighbor
        max_similarity = self.recognition_model.kneighbors(img_embedding)[0].tolist()[0][0]

        # Get threshold value of nearest neighbor
        similarity_threshold = self.get_similarity_threshold(label_index)

        # Here it´s the opposite of the inner product (the smaller the distance the closer the embeddings to each other)
        if max_similarity <= similarity_threshold:
            check = "Access"
        else:
            check = "Decline"

        return closest_label, check

    # Find closest embedding based on inner product and adaptive thresholds and thus decide, if person known or unknown
    def closest_embedding_inner_product(self, img_embedding):

        # calculate the inner product to all other embeddings
        inner_products = np.inner(img_embedding,self.embeddings_list)

        # Get index with hightest value (which is the closest vector) and convert it into a list (so get_label works for knn and inner product)
        label_index = [np.argmax(inner_products)]

        # Use index to get the label
        closest_label = self.get_label(label_index)

        # Calculate similarity to closest person
        max_similarity = np.max(inner_products)



        # Get threshold value of closest person
        similarity_threshold = self.get_similarity_threshold(label_index)


        # The larger the similarity the closer the embeddings to each other
        if max_similarity >= similarity_threshold:
            check = "Access"

        else:
            check = "Decline"

        return closest_label, check
        
    # Find the closest person in the embedding space and decide then, whether access or intruder
    def face_recognition(self, img_embedding_tensor):
        
        if self.len_embeddings_list == 0:
            print("Person is unkown")
            return

        img_embedding_numpy = self.convert_to_numpy(img_embedding_tensor)

        if self.mode == 'inner_product':
            closest_label, check = self.closest_embedding_inner_product(img_embedding_numpy)
        elif self.mode == 'euclidean_distance':
            closest_label, check = self.closest_embedding_euclidean_distance(img_embedding_numpy)  

        return closest_label, check

    def face_registration(self, name, img_embedding_tensor):

        # Calculate embedding and convert to numpy array
        img_embedding_numpy = self.convert_to_numpy(img_embedding_tensor)

        # Add label, embedding and threshold to database (threshold first of all set to 0, will be udpated later on)
        self.database = self.database.append({'label': name, 'embedding': img_embedding_numpy, 'threshold': 0}, ignore_index=True)

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

        self.database.drop(drop_indices, inplace=True)
        # reset index, so that it counts again from zero if person is deregistered from the middle
        self.database.reset_index(drop=True,inplace=True)

        # Update length of embeddings list and embeddings list itself
        self.update_embeddings() 

        # Save it to pickle file
        self.save_database()
        
        

 


        
