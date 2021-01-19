import os
import torch
import numpy as np
import pandas as pd

from os.path import join, dirname, abspath

from PIL import Image
from sklearn.neighbors import NearestNeighbors

# input: 128 dim embedding as tensor (convert it internally to numpy array)
#               - registration: embedding + name
#               - deregistration: name
#               - recognition: embedding
# ---------------------------------------------------------------------------
# output:       - registration: "registered successfully"
#               - deregistration: "deregistered successfully"
#               - recognition: closest person + access/intruder

ABSOLUTE_DIR = dirname(abspath(__file__))

class RegistrationDatabase():
    def __init__(self, fixed_threshold, mode='inner_product'):        
        super().__init__()
        # Choose similarity calculation between "inner product" and "euclidean distance"
        self.mode = mode
        if self.mode == 'euclidean_distance':
            self.recognition_model = NearestNeighbors(n_neighbors=1)
        
        self.fixed_threshold = fixed_threshold
        self.len_embeddings_list = 0
                             
        save_dir = join(ABSOLUTE_DIR, "./reg_database")
        os.makedirs(save_dir, exist_ok=True)
        self.database_file = os.path.join(save_dir, 'database.pkl')

        if os.path.exists(self.database_file):
            print('Database already exists. Pickle file will be loaded...')
            self.database = pd.read_pickle(self.database_file)
            self.update_embeddings()
        else: 
            print("No database availabe. Empty database will be created...")
            self.database = pd.DataFrame(columns=['label','embedding','threshold'])
            self.save_database()
    
    def save_database(self):
        '''Save database to pickle file'''
        self.database.to_pickle(self.database_file)

    def clean_database(self):
        '''wipes database'''
        if os.path.exists(self.database_file):
            print('database.pkl exists and will be cleaned...')
            self.database = pd.DataFrame(columns=['label','embedding','threshold'])
            self.update_embeddings()
            self.save_database()
        else: 
            print("No database.pkl file exists. Hence, it cannot be cleaned...")      

    def contains(self, label):
        '''Check if database contains specified label'''
        label_indices = self.database[ self.database['label'] == label ].index
        if len(label_indices) == 0:
            return False
        elif len(label_indices) > 0:
            return True

    
    def update_embeddings(self):
        '''Update length of embedding list, embedding list itself and the thresholds of every person'''
        self.len_embeddings_list = len(self.database.index)
        self.embeddings_list = [self.database.iloc[i,1][0] for i in range(self.len_embeddings_list)]

        # Calculate and update inner product thresholds
        # Adapt threshold for first embedding (can be deleted?)
        if self.database['label'].nunique() == 1:
            self.database.iloc[:,2] = 98.0
        elif self.database['label'].nunique() > 1:
            # Calculate the similarity score between a selected embedding and all the other embeddings
            for i in range(self.len_embeddings_list):
                temp_embeddings_list = self.embeddings_list.copy()
                temp_embedding = temp_embeddings_list[i]

                cur_label = self.get_label([i])
                cur_label_indices = self.database[self.database['label']==cur_label].index.values.astype(int).tolist()
                for index in sorted(cur_label_indices, reverse=True): # reverse order --> doesn´t throw off subsequent indices
                    del temp_embeddings_list[index]

                if self.mode == 'inner_product':
                    # Inner product is 100, when two vectors are identical
                    inner_products = np.inner(temp_embedding,temp_embeddings_list)

                    closest_embedding_dist = np.max(inner_products) # Inner product threshold
                    if closest_embedding_dist > self.fixed_threshold:
                        self.database.iloc[i,2] = closest_embedding_dist
                    else:
                        self.database.iloc[i,2] = self.fixed_threshold

                elif self.mode == 'euclidean_distance':
                    self.recognition_model.fit(temp_embeddings_list)
                    print("----------- Fixed threshold not defined so far! --------------")

                    closest_embedding_dist = self.recognition_model.kneighbors(temp_embedding.reshape((1,128)))[0].tolist()[0][0]
                    self.database.iloc[i,2] = closest_embedding_dist


        if self.len_embeddings_list > 0 and self.mode == 'euclidean_distance':
            self.recognition_model.fit(self.embeddings_list)

    def convert_to_numpy(self, img_embedding_tensor):
        '''Converts torch tensor to numpy array'''
        return img_embedding_tensor.detach().cpu().numpy()

    def get_label(self, index):
        '''Get the label at given index in database'''
        label = self.database.iloc[index,0].reset_index(drop=True)[0]
        return label

    def get_similarity_threshold(self, index):
        similarity_threshold = self.database.iloc[index,2].reset_index(drop=True)[0]
        return similarity_threshold
    
    def closest_embedding_euclidean_distance(self, img_embedding):
        '''Find closest embedding based on euclidean distance (use KNN with k=1)'''
        label_index = self.recognition_model.kneighbors(img_embedding)[1].tolist()[0]
        closest_label = self.get_label(label_index)

        max_similarity = self.recognition_model.kneighbors(img_embedding)[0].tolist()[0][0]
        similarity_threshold = self.get_similarity_threshold(label_index)

        # The smaller the distance the closer the embeddings to each other
        if max_similarity <= similarity_threshold:
            check = "Access"
        else:
            check = "Decline"

        return closest_label, check

    def closest_embedding_inner_product(self, img_embedding):
        '''Find closest embedding based on inner product and adaptive thresholds'''
        inner_products = np.inner(img_embedding,self.embeddings_list)

        # hightest value = closest vector
        # convert to list (so get_label works for knn and inner product)
        label_index = [np.argmax(inner_products)]
        closest_label = self.get_label(label_index)

        max_similarity = np.max(inner_products)
        similarity_threshold = self.get_similarity_threshold(label_index)

        # The larger the similarity the closer the embeddings to each other
        if max_similarity >= similarity_threshold:
            check = "Access"

        else:
            check = "Decline"

        return closest_label, check
        
    def face_recognition(self, img_embedding_tensor):
        '''Grants access of closes person in embedding space could be found; denies acces otherwise'''
        if self.len_embeddings_list == 0:
            print("Person is unkown")
            closest_label = None
            check = "Decline"
            return closest_label, check

        img_embedding_numpy = self.convert_to_numpy(img_embedding_tensor)

        if self.mode == 'inner_product':
            closest_label, check = self.closest_embedding_inner_product(img_embedding_numpy)
        elif self.mode == 'euclidean_distance':
            closest_label, check = self.closest_embedding_euclidean_distance(img_embedding_numpy)  

        return closest_label, check

    def face_registration(self, name, img_embedding_tensor):
        '''Register new face to database using face embeddings'''
        img_embedding_numpy = self.convert_to_numpy(img_embedding_tensor)
        self.database = self.database.append({'label': name, 'embedding': img_embedding_numpy, 'threshold': 0}, ignore_index=True)
        self.update_embeddings()
        self.save_database()

    def face_deregistration(self, name):
        '''Removes face and its embedding from database'''
        drop_indices = self.database[ self.database['label'] == name ].index
        if len(drop_indices) == 0:
            print('Specified name not in database registered. User can not be deregistered!')
            return

        self.database.drop(drop_indices, inplace=True)
        self.database.reset_index(drop=True,inplace=True)
        self.update_embeddings() 
        self.save_database()