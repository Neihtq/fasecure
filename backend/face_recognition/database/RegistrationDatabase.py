import os
import numpy as np
import pandas as pd

from sklearn.neighbors import NearestNeighbors

from face_recognition.utils.constants import UNKNOWN_PERSON, CANNOT_WIPE_DATBABASE, USER_NOT_REGISTERED, DATABASE_DIR, DATABASE_EXIST, CREATE_NEW_DATABASE, EUCLIDEAN_DISTANCE, WIPE_DATABASE, INNER_PRODUCT, UNDEFINED_THRESHOLD


class RegistrationDatabase():
    '''128 dim embeddings arrays stored in Pandas DataFrame
    input: 
        - registration: embedding + name
        - deregistration: name
        - recognition: embedding
    output:
        - registration: "registered successfully"
        - deregistration: "deregistered successfully"
        - recognition: closest person + access/intruder'''
    def __init__(self, fixed_initial_threshold, mode=INNER_PRODUCT):        
        # Choose similarity calculation between "inner product" and "euclidean distance"
        self.mode = mode
        if self.mode == EUCLIDEAN_DISTANCE:
            self.recognition_model = NearestNeighbors(n_neighbors=1)
        
        self.fixed_threshold = fixed_initial_threshold
        self.len_embeddings_list = 0
        self.database_file = DATABASE_DIR

        if os.path.exists(self.database_file):
            print(DATABASE_EXIST)
            self.database = pd.read_pickle(self.database_file)
            self.update_embeddings()
        else: 
            print(CREATE_NEW_DATABASE)
            self.database = pd.DataFrame(columns=['label','embedding','threshold'])
            self.save_database()
    
    def save_database(self):
        '''Save database to pickle file'''
        self.database.to_pickle(self.database_file)

    def clean_database(self):
        '''wipes database'''
        if os.path.exists(self.database_file):
            print(WIPE_DATABASE)
            self.database = pd.DataFrame(columns=['label','embedding','threshold'])
            self.update_embeddings()
            self.save_database()
            return 0
        print(CANNOT_WIPE_DATBABASE)
        return -1

    def contains(self, label):
        '''Check if database contains specified label'''
        label_indices = self.database[self.database['label'] == label].index
        
        return len(label_indices) > 0
    
    def update_embeddings(self):
        '''Update length of embedding list, embedding list itself and the thresholds of every person'''
        self.len_embeddings_list = len(self.database.index)
        self.embeddings_list = [self.database.iloc[i,1][0] for i in range(self.len_embeddings_list)]

        # Calculate and update inner product thresholds
        # Adapt threshold for first embedding (can be deleted?)
        if self.database['label'].nunique() == 1:
            self.database.iloc[:,2] = 98.0
        else:
            # Calculate the similarity score between a selected embedding and all the other embeddings
            for i in range(self.len_embeddings_list):
                temp_embeddings_list = self.embeddings_list.copy()
                temp_embedding = temp_embeddings_list[i]

                cur_label = self.get_label([i])
                cur_label_indices = self.database[self.database['label'] == cur_label].index.values.astype(int).tolist()
                for index in sorted(cur_label_indices, reverse=True):
                    del temp_embeddings_list[index]

                if self.mode == INNER_PRODUCT:
                    # Inner product is 100, when two vectors are identical
                    inner_products = np.inner(temp_embedding,temp_embeddings_list)

                    closest_embedding_dist = np.max(inner_products) # Inner product threshold
                    if closest_embedding_dist > self.fixed_threshold:
                        self.database.iloc[i,2] = closest_embedding_dist
                    else:
                        self.database.iloc[i,2] = self.fixed_threshold

                elif self.mode == EUCLIDEAN_DISTANCE:
                    self.recognition_model.fit(temp_embeddings_list)
                    print(UNDEFINED_THRESHOLD)

                    closest_embedding_dist = self.recognition_model.kneighbors(temp_embedding.reshape((1,128)))[0].tolist()[0][0]
                    self.database.iloc[i,2] = closest_embedding_dist


        if self.len_embeddings_list > 0 and self.mode == EUCLIDEAN_DISTANCE:
            self.recognition_model.fit(self.embeddings_list)

    def convert_to_numpy(self, img_embedding_tensor):
        '''Converts torch tensor to numpy array'''
        return img_embedding_tensor.detach().cpu().numpy()

    def get_label(self, index):
        '''Get the label at given index in database'''
        return self.database.iloc[index,0].reset_index(drop=True)[0]

    def get_similarity_threshold(self, index):
        return self.database.iloc[index,2].reset_index(drop=True)[0]
    
    def closest_embedding_euclidean_distance(self, img_embedding):
        '''Find closest embedding based on euclidean distance (use KNN with k=1)'''
        label_index = self.recognition_model.kneighbors(img_embedding)[1].tolist()[0]
        closest_label = self.get_label(label_index)

        max_similarity = self.recognition_model.kneighbors(img_embedding)[0].tolist()[0][0]
        similarity_threshold = self.get_similarity_threshold(label_index)

        # The smaller the distance the closer the embeddings to each other
        access = max_similarity <= similarity_threshold
        
        return closest_label, access

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
        access = max_similarity >= similarity_threshold

        return closest_label, access
        
    def face_recognition(self, img_embedding_tensor):
        '''Grants access of closes person in embedding space could be found; denies acces otherwise'''
        if self.len_embeddings_list == 0:
            print(UNKNOWN_PERSON)
            closest_label = None
            access = False
            return closest_label, access

        img_embedding_numpy = self.convert_to_numpy(img_embedding_tensor)

        if self.mode == INNER_PRODUCT:
            closest_label, access = self.closest_embedding_inner_product(img_embedding_numpy)
        elif self.mode == EUCLIDEAN_DISTANCE:
            closest_label, access = self.closest_embedding_euclidean_distance(img_embedding_numpy)  

        return closest_label, access

    def face_registration(self, name, img_embedding_tensor):
        '''Register new face to database using face embeddings'''
        img_embedding_numpy = self.convert_to_numpy(img_embedding_tensor)
        self.database = self.database.append({'label': name, 'embedding': img_embedding_numpy, 'threshold': 0}, ignore_index=True)
        self.update_embeddings()
        self.save_database()

        return 0

    def face_deregistration(self, name):
        '''Removes face and its embedding from database'''
        drop_indices = self.database[ self.database['label'] == name ].index
        if len(drop_indices) == 0:
            print(USER_NOT_REGISTERED)
            return -1

        self.database.drop(drop_indices, inplace=True)
        self.database.reset_index(drop=True,inplace=True)
        self.update_embeddings() 
        self.save_database()

        return 0