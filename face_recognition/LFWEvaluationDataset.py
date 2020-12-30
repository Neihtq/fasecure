
from torch.utils.data import Dataset
from torchvision import transforms
import glob
from os.path import join
from os import listdir
from PIL import Image

class LFWEvaluationDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.labels = []
        self.mask = []
        self.transform = transform
        for label in listdir(root):
            img_path = join(root, label)
            if len(listdir(img_path)) > 5:
                for imgs_per_folder in range(len(listdir(img_path))):
                    self.labels.append(label)
                    self.mask.append(True)
            else:
                for imgs_per_folder in range(len(listdir(img_path))):
                    self.mask.append(False)

        self.image_filenames = glob.glob(join(root, "**/*.jpg"), recursive=True)
        # Use mask to filter classes with less than certain amount of images
        self.image_filenames = list(compress(self.image_filenames, self.mask))

    def __len__(self):
        # returns amount of classes
        return len(self.labels)
    
    def __getitem__(self, idx):
        label = self.labels[idx]
        img_path = self.image_filenames[idx]
        
        return label, img_path