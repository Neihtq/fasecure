# Input: 
# - path where dataset is stored (LFW, as it has less images per person on average. Thus closer to one-shot learning)
# - path were the evaluation log should be stored
# - Face Detection & Alignment model
# - Face Embedding model
# - RegistrationDatabase

# Class
# Methods:
# - run()
# - plot_results()

# Procedure run: 
# 1. Read all images and shuffle them randomly (so that images of same person don´t appear right after each other)
#       Use random seed for shuffling, so that order is always the same
# for-loop (each image)
    # 2. Pass image through whole pipeline (detection & alignment # embedding)
    # 3. Try to recognize person and then adapt TA, TR, ... (for first person, directly register --- for first people, use fixed threshold)
    # 4. Register person to database (only if it doesn´t already exist (one-shot learning). Use function for augmentation which also the main programm uses)
    # print every 100 images intermediate results
# 5. Calculate overall accuracy


# --- Output: Overall Accuracy

# todo:
# - evalation normally on which dataset? new one?
# - implement code, that if cuda available, if moves everything on the gpu
# - print names and access and have a look, if metric is computed correctly
# - Improve accuracy (start with fixed threshold, until one neighbor is closer)
#   -> only update threshold if new max is above old threshold (fixed threshold at beginning)
#   -> How to set fixed threshold? grid-search?
#   (maybe that was ment by "initialized thresholds with fixed ones" in the paper)

from LFWEvaluationDataset import LFWEvaluationDataset
from LFWEvaluationDatasetCropped import LFWEvaluationDatasetCropped
from prep import img_transform_augmentations
import torch
import numpy as np
from PIL import Image
import sys
import matplotlib.pyplot as plt

from scipy import interpolate
from scipy.optimize import fsolve

class PipelineEvaluation():

    def __init__(self, dataset_path, eval_log_path, face_embedding_model,
                         registration_database, face_detection_model=None):
        
        self.dataset_path = dataset_path
        self.eval_log_path = eval_log_path
        self.face_detection_model = face_detection_model
        self.face_embedding_model = face_embedding_model
        self.evaluation_database = registration_database

        # If don´t pass a face detection & alignment model,
        # then it directly tries to load cropped images and skips this step
        if face_detection_model == None:
            self.eval_dataset = LFWEvaluationDatasetCropped(self.dataset_path)
            self.skip_face_detection = True
        else:
            self.eval_dataset = LFWEvaluationDataset(self.dataset_path)

    def run(self):
        
        # ---- SHUFFLE DATASET RANDOMLY --------
        # divide dataset size by twenty 10, so that processing is faster
        subset_size = 10
        n_samples = int(self.eval_dataset.__len__()/subset_size)

        # Shuffle indices with np.random.permutation, also fix seed if you want reproducability
        shuffled_indices = np.random.RandomState(seed=42).permutation(n_samples)

        # select shuffled set from original dataset
        eval_dataset_shuffled = torch.utils.data.Subset(self.eval_dataset, indices=shuffled_indices)   

        # ---- CREATE DATALOADER --------
        batch_size = 1
        eval_loader = torch.utils.data.DataLoader(dataset=eval_dataset_shuffled,
                                                    batch_size=batch_size,
                                                    num_workers=0,
                                                    shuffle=False, 
                                                    sampler=None,
                                                    collate_fn=None)

        # ---- LOOP OVER EVERY IMAGE --------

        # move models to cuda


        self.evaluation_database.clean_database()

        fa = 0  # False accept
        wa = 0  # Wrong answer
        fr = 0  # False reject
        accept = 0
        reject = 0

        rec_number = 0

        for i, (label, image_path) in enumerate(eval_loader):
            label = label[0]
            # ---- FACE DETECTION AND ALIGNMENT --------
            if self.skip_face_detection == True:
                detected_face = image_path[0].unsqueeze(0)
            else:
                try:
                    detected_face_numpy = self.face_detection_model.detectFace(image_path[0])
                except:
                    # Sometimes, if more than one face is on an image, deepFace returns an error
                    # However, it doesn´t count to overall accuracy, as we will get only images with one face
                    print("image not recognized: ", image_path[0])
                    continue

                # ---- RESHAPE OUTPUT --------
                detected_face = detected_face_numpy.copy()
                # Afterwards: Pytorch tensor 1x3x250x250
                detected_face = torch.from_numpy(detected_face).permute(2,0,1).unsqueeze(0)
                # print(detected_face.shape)
                #plt.imshow(detected_face.permute(1, 2, 0))


            
            if torch.cuda.is_available():
                detected_face = detected_face.to("cuda")
                #target = target.to("cuda")       
            
            # ---- TRANSFORMATION AND AUGMENTATIONS --------
            # (to standardize already not-augmented image)
            aug_img_1, aug_img_2, aug_img_3, aug_img_4, aug_img_5, aug_img_6, aug_img_7 = img_transform_augmentations(detected_face)
            
            # ---- FACE EMBEDDING --------
            embedding = self.face_embedding_model(aug_img_1)
            
            # ---- FACE RECOGNITION -------- 
            # (No face recognition for first person, as no faces registered so far)
            if rec_number > 0:
                closest_label, check = self.evaluation_database.face_recognition(embedding)
                # print("True label: ", label)    
                # print("Predicted label:", closest_label, " --- Access?: ", check)
                
                # Allegedly known
                # 3 cases with Joe in the image and he gets access:
                # 1) Joe gets access, although he is not registered at all
                # 2) Joe gets access, he is registered, but the model has mixed up Joe with someone else who is registered
                # 3) Joe gets access, he is registered and the model recognizes him (Correct)
                if check == 'Access':
                    accept += 1
                    # Case 1)
                    if not self.evaluation_database.contains(label):
                        fa += 1  # False accept
                    # Case 2)
                    elif closest_label != label:
                        wa += 1  # Recognition error
                    
                # Allegedly Intruder
                # 2 cases with Joe in the image and he gets declined
                # 1) Joe gets declined, although is is registered
                # 2) Joe gets declined, since he is not registered (Correct)
                if check == 'Decline':
                    reject += 1
                    if self.evaluation_database.contains(label):
                        fr += 1  # False reject
            
            # # ---- FACE REGISTRATION -------- 
            # (only if label not already in database, as one-shot learning)
            if not self.evaluation_database.contains(label):
                # If we register, then augmentation step between Face Detection/Alignemnt and Face Embedding
                # Can I also write that in one line with **?
                img_embedding_tensor_2 = self.face_embedding_model(aug_img_2)
                img_embedding_tensor_3 = self.face_embedding_model(aug_img_3)
                img_embedding_tensor_4 = self.face_embedding_model(aug_img_4)
                img_embedding_tensor_5 = self.face_embedding_model(aug_img_5)
                img_embedding_tensor_6 = self.face_embedding_model(aug_img_6)
                img_embedding_tensor_7 = self.face_embedding_model(aug_img_7)
                
                # The first one we already calculated
                self.evaluation_database.face_registration(label,embedding)
                self.evaluation_database.face_registration(label,img_embedding_tensor_2)
                self.evaluation_database.face_registration(label,img_embedding_tensor_3)
                self.evaluation_database.face_registration(label,img_embedding_tensor_4)
                self.evaluation_database.face_registration(label,img_embedding_tensor_5)
                self.evaluation_database.face_registration(label,img_embedding_tensor_6)
                self.evaluation_database.face_registration(label,img_embedding_tensor_7)
                
            if (rec_number > 0) and (rec_number % 100 == 0):      
                # Calculate error
                self.show_and_save(fa, fr, wa, accept, reject, rec_number, self.eval_log_path)
            # Only increases rec_number, if face detected
            rec_number += 1


    def green_print(self, line):
        print('\033[92m'+line+'\033[0m')

    def calculate_error(self, fa, fr, wa, accept, reject):
        if accept > 0:
            # false accept rate -> as low as possible
            far = float(fa/accept)
            # wrong accept rate -> as low as possible
            war = float(wa/accept)
        else:
            far = 0
            war = 0

        if reject > 0:
            # false reject rate -> as low as possible
            frr = float(fr/reject)
        else:
            frr = 0
        
        error = (fa+fr+wa)/(accept+reject)
        return far, frr, war, error        

    def show_and_save(self, fa, fr, wa, accept, reject, compare_num, filepath):
        # Calculate error
        far, frr, war, error = self.calculate_error(fa, fr, wa, accept, reject)

        # String formatting: every %f oder %d is an index in the following tuple (f: float, d: double?)
        # However, this type of string formatting is not recommended!!
        info = 'compare_num: %d\nfar:%f(%d/%d), frr:%f(%d/%d), war:%f(%d/%d), acc:%.4f(%d/%d)\n' % \
                (compare_num, far, fa, accept, frr, fr, reject, 
                war, wa, accept, 1-error, (accept+reject)-(fa+fr+wa), (accept+reject))
        # Print result
        self.green_print(info)
        # Save result
        with open(filepath, 'a') as file:
            file.write(info)

    def get_rate(self, key, index, rates):
        rate = [rates[i][index] for i in range(len(rates))]
        rate = [v.split(key)[-1] for v in rate]
        rate = [v.split('(')[0] for v in rate]
        rate = [float(v) for v in rate]
        return rate

    def findIntersection(fun1, fun2, x0):
        return fsolve(lambda x : fun1(x) - fun2(x), x0)

    def plot_results(self):
        # Read file
        with open(self.eval_log_path, 'r') as file:
            all_data = file.read().split('\n')
        if all_data[-1] == '':
            del all_data[-1]

        # Split x and y data
        compare_num = []
        rates = []
        for indx, v in enumerate(all_data):
            if indx%2 == 0:
                compare_num.append(float(v.split(': ')[-1]))
            else:
                rates.append(v.split(', '))

        # Get y axis data
        far = self.get_rate('far:', 0, rates)
        frr = self.get_rate('frr:', 1, rates)
        war = self.get_rate('war:', 2, rates)
        acc = self.get_rate('acc:', 3, rates)

        # Find max accuracy
        max_acc_index = np.argmax(acc)
        best_acc = acc[max_acc_index]
        best_num = compare_num[max_acc_index]
        best_far = far[max_acc_index]
        best_frr = frr[max_acc_index]

        # Find EER point
        f1 = interpolate.interp1d(compare_num, far)
        f2 = interpolate.interp1d(compare_num, frr)
        f3 = interpolate.interp1d(compare_num, acc)

        #compare_num_EER = self.findIntersection(f1, f2, start)

        # Plot begin
        plt.figure(figsize=(14,6))

        # Error rates
        plt.subplot(121)
        plt.plot(compare_num, far, label='FAR')
        plt.plot(compare_num, frr, label='FRR')
        plt.plot(compare_num, war, label='WAR')
        
        # Points
        #EER_label = ('EER: %.4f/%d' % (f1(compare_num_EER), compare_num_EER))
        acc_far_label = ('FAR: %.4f/%d' % (best_far, best_num))
        acc_frr_label = ('FRR: %.4f/%d' % (best_frr, best_num))
        #plt.plot(compare_num_EER, f1(compare_num_EER), 'ro', label=EER_label)
        plt.plot(best_num, best_far, 'go', label=acc_far_label)
        plt.plot(best_num, best_frr, 'bo', label=acc_frr_label)
        
        plt.xlabel('Number of recognition/registration tests')
        plt.ylabel('Error rate')
        plt.title('Adaptive threshold - Error rates')
        plt.legend(loc=4)
        
        # Accuracy
        plt.subplot(122)
        plt.plot(compare_num, acc)
        max_label = ('Best: %.4f/%d' % (best_acc, best_num))
        #acc_EER_label = ('EER: %.4f/%d' % (f3(compare_num_EER), compare_num_EER))
        plt.plot(best_num, best_acc, 'ro', label=max_label)
    #     plt.plot(compare_num_EER, f3(compare_num_EER), 'bo', label=acc_EER_label)
    #     plt.xlabel('Max number of class compared')
    #     plt.ylabel('Accuracy')
    #     plt.title('Adaptive threshold - Accuracy')
    #     plt.legend()


        plt.show()