# import libraries
import pandas as pd
import os
import torch

# - implement database class which stores the labels and embeddings (pandas dataframe)
#       attibutes: pandas dataframe
#       methods:    - save_dataframe
#                   - read dataframe from path
#                   - create dataframe (database with labels and embeddings)
#                       input: path to folder where images are located (name convention: name_01.png, name_02.png,...)
#                       function: use path to get label, calculate all embeddings for one person (and center) and save both in DB
#                       output: nothing
#                   - delete database (in the end, to make sure, that when pictures are added to specified image path, that it creates a new DB)
#                   - face registration
#                   - one-shot learning

# open questions:
# to handle the case, that different amounts of images are availabe for one person, who has to be registered:
# -> batch size one and then store all the embeddings with the same label in a list, then calculate the mean

class RegistrationDatabase():

    # Register people or load registered people
    def __init__(self, faceRecognitionModel, dataloader=None):

        self.model = faceRecognitionModel

        # have a look if database (csv file) already available, if yes, load it and save it into pandas dataframe and return
        model_dir = "./reg_database"
        os.makedirs(model_dir, exist_ok=True)

        database_file = os.path.join(model_dir, 'database.csv')

        # If no dataloader specified, then try to load database
        if dataloader == None:
            # If path exists, then load csv file and safe it into pandas dataframe
            # (Saves file according to specified path. If want to make sure that registration is reloaded, then delete DB)
            if os.path.exists(database_file):
                # load csv file and save it into class attribute database
                print('already exists')
                self.database = pd.read_csv(database_file,index_col=0)
            else: 
                raise Exception('No database availabe. You have to specifiy a dataloader containing all the images for registration')


        # If dataloader specified, then overwrite database (or create a new one, if none existing)
        else:
            print('Save new database')

            # create pandas dataframe 
            self.database = pd.DataFrame(columns=['label','embedding'])

            for i, data in enumerate(dataloader):
                img, label = data

                # if torch.cuda.is_available():
                #     img = img.to("cuda")
                #     label = target.to("cuda")

                print("before")
                embedding_1 = self.model(img)

                # use img_folder_path to get labels and embeddings
                self.database = self.database.append({'label': label[0], 'embedding': embedding_1}, ignore_index=True)


            # Save it as a csv file
            self.database.to_csv(database_file)


        # In the end after running constructor: saved database as csv file and database attribute contains registration database


        

 


        
