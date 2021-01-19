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
import torch
import sys
import numpy as np
import matplotlib.pyplot as plt
from os.path import split, join
import glob

from PIL import Image
from scipy import interpolate
from scipy.optimize import fsolve

from dataset.LFWDataset import LFWEvaluationDataset
from utils.prep import img_augmentation


class PipelineEvaluation():
    def __init__(self, dataset_path, eval_log_path, face_embedding_model,
                         registration_database, face_detection_model=None):
        self.dataset_path = dataset_path
        self.eval_log_path = eval_log_path
        self.face_detection_model = face_detection_model
        self.face_embedding_model = face_embedding_model
        self.evaluation_database = registration_database

        if face_detection_model == None:
            # directly load cropped images
            self.eval_dataset = LFWEvaluationDataset(self.dataset_path, cropped_faces=True)
            self.skip_face_detection = True
        else:
            self.eval_dataset = LFWEvaluationDataset(self.dataset_path)

    def run(self):
        subset_size = 10
        n_samples = int(self.eval_dataset.__len__()/subset_size)
        shuffled_indices = np.random.RandomState(seed=42).permutation(n_samples)
        eval_dataset_shuffled = torch.utils.data.Subset(self.eval_dataset, indices=shuffled_indices)   
        batch_size = 1
        eval_loader = torch.utils.data.DataLoader(dataset=eval_dataset_shuffled,
                                                    batch_size=batch_size,
                                                    num_workers=0,
                                                    shuffle=False, 
                                                    sampler=None,
                                                    collate_fn=None)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.evaluation_database.clean_database()

        fa = 0  # False accept
        wa = 0  # Wrong accept
        fr = 0  # False reject
        accept = 0
        reject = 0
        rec_number = 0
        for i, (label, image_path) in enumerate(eval_loader):
            label = label[0]
            
            # Face detection and alignment
            if self.skip_face_detection:
                detected_face = image_path[0].unsqueeze(0)
            else:
                try:
                    detected_face_numpy = self.face_detection_model.detectFace(image_path[0])
                except:
                    # Sometimes, if more than one face is on an image, deepFace returns an error
                    # However, it doesn´t count to overall accuracy, as we will get only images with one face
                    print("image not recognized: ", image_path[0])
                    continue

                detected_face = detected_face_numpy.copy()
                detected_face = torch.from_numpy(detected_face).permute(2,0,1).unsqueeze(0)
            
            detected_face = detected_face.to(device)


            augmented_imgs = img_augmentation(detected_face)            
            embedding = self.face_embedding_model(augmented_imgs[0])

            # Face recognition
            if rec_number > 0:
                closest_label, check = self.evaluation_database.face_recognition(embedding)
                
                # Allegedly known
                # 3 cases with Joe in the image and he gets access:
                # 1) Joe gets access, although he is not registered at all
                # 2) Joe gets access, he is registered, but the model has mixed up Joe with someone else who is registered
                # 3) Joe gets access, he is registered and the model recognizes him (Correct)
                if check == 'Access':
                    accept += 1
                    # Case 1)
                    if not self.evaluation_database.contains(label):
                        fa += 1
                    # Case 2)
                    elif closest_label != label:
                        wa += 1
                    
                # Allegedly unknown
                # 2 cases with Joe in the image and he gets declined
                # 1) Joe gets declined, although is is registered
                # 2) Joe gets declined, since he is not registered
                if check == 'Decline':
                    reject += 1
                    if self.evaluation_database.contains(label):
                        fr += 1
            
            # Face registration
            if not self.evaluation_database.contains(label):
                #self.evaluation_database.face_registration(label,embedding)
                for aug_img in augmented_imgs:
                    img_embedding_tensor = self.face_embedding_model(aug_img)
                    self.evaluation_database.face_registration(label, img_embedding_tensor)
                
            if (rec_number > 0) and (rec_number % 10 == 0):      
                self.show_and_save(fa, fr, wa, accept, reject, rec_number, self.eval_log_path)
            
            # Only increases rec_number, if face detected
            rec_number += 1

    def green_print(self, line):
        print('\033[92m'+line+'\033[0m')

    def calculate_error(self, fa, fr, wa, accept, reject):
        '''
        fa: false accept
        wa: wrong accept
        fr: false reject
        '''
        if accept > 0:
            fa_rate = float(fa/accept)
            wa_rate = float(wa/accept)
        else:
            fa_rate = 0
            wa_rate = 0

        if reject > 0:
            fr_rate = float(fr/reject)
        else:
            fr_rate = 0
        
        error = (fa+fr+wa)/(accept+reject)
        return fa_rate, fr_rate, wa_rate, error        

    def show_and_save(self, fa, fr, wa, accept, reject, compare_num, filepath):
        fa_rate, fr_rate, wa_rate, error = self.calculate_error(fa, fr, wa, accept, reject)

        # String formatting: every %f oder %d is an index in the following tuple (f: float, d: double?)
        # However, this type of string formatting is not recommended!!
        info = 'compare_num: %d\nfa_rate:%f(%d/%d), fr_rate:%f(%d/%d), wa_rate:%f(%d/%d), acc:%.4f(%d/%d)\n' % \
                (compare_num, fa_rate, fa, accept, fr_rate, fr, reject, 
                wa_rate, wa, accept, 1-error, (accept+reject)-(fa+fr+wa), (accept+reject))
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

    def find_intersection(self, fun1, fun2, x0):
        return fsolve(lambda x : fun1(x) - fun2(x), x0)

    def plot_results(self):
        with open(self.eval_log_path, 'r') as file:
            all_data = file.read().split('\n')

        if all_data[-1] == '':
            del all_data[-1]

        # Split x and y data
        compare_num = []
        rates = []
        for indx, v in enumerate(all_data):
            if indx % 2 == 0:
                compare_num.append(float(v.split(': ')[-1]))
            else:
                rates.append(v.split(', '))

        # Get y axis data
        fa_rate = self.get_rate('fa_rate:', 0, rates)
        fr_rate = self.get_rate('fr_rate:', 1, rates)
        wa_rate = self.get_rate('wa_rate:', 2, rates)
        acc = self.get_rate('acc:', 3, rates)

        # Find max accuracy
        max_acc_index = np.argmax(acc)
        best_acc = acc[max_acc_index]
        best_num = compare_num[max_acc_index]
        best_fa_rate = fa_rate[max_acc_index]
        best_fr_rate = fr_rate[max_acc_index]

        # Find EER point
        f1 = interpolate.interp1d(compare_num, fa_rate)
        f2 = interpolate.interp1d(compare_num, fr_rate)
        f3 = interpolate.interp1d(compare_num, acc)

        plt.figure(figsize=(14,6))

        # Error rates
        plt.subplot(121)
        plt.plot(compare_num, fa_rate, label='FA Rate')
        plt.plot(compare_num, fr_rate, label='FR Rate')
        plt.plot(compare_num, wa_rate, label='WA Rate')
        
        # Points
        acc_fa_rate_label = ('FA Rate: %.4f/%d' % (best_fa_rate, best_num))
        acc_fr_rate_label = ('FR Rate: %.4f/%d' % (best_fr_rate, best_num))
        plt.plot(best_num, best_fa_rate, 'go', label=acc_fa_rate_label)
        plt.plot(best_num, best_fr_rate, 'bo', label=acc_fr_rate_label)
        
        plt.xlabel('Number of recognition/registration tests')
        plt.ylabel('Accuracy')
        plt.title('Adaptive threshold - Performance measures')
        plt.legend(loc=4)
        
        # Accuracy
        plt.subplot(122)
        plt.plot(compare_num, acc)
        max_label = ('Best: %.4f/%d' % (best_acc, best_num))
        plt.plot(best_num, best_acc, 'ro', label=max_label)

        plt.show()

    # Compare different evaluations (normally they differ in the fixed threshold)
    def compare_evaluations(self):
        eval_folder = split(self.eval_log_path)[0]
        eval_results_filenames = glob.glob(join(eval_folder, "**/*.txt"), recursive=True)

        plt.figure(figsize=(14,6))

        for i in range(len(eval_results_filenames)):

            eval_filename = eval_results_filenames[i]

            # In this setting, the fixed threshold is the label
            label = str(eval_filename.split('_')[-1].split('.txt')[0])

            # Read file
            with open(eval_filename, 'r') as file:
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
            acc = self.get_rate('acc:', 3, rates)

            # Plot accuracy
            plt.plot(compare_num, acc, label=label)

        plt.xlabel('Number of recognition/registration tests')
        plt.ylabel('Accuarcy')
        plt.title('Comparison of multiple evaluations')
        plt.legend(loc=0)

        plt.show()