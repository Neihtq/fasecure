
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
        self.transform = transform
        for label in listdir(root):
            img_path = join(root, label)
            for imgs_per_folder in range(len(listdir(img_path))):
                self.labels.append(label)

        self.image_filenames = glob.glob(join(root, "**/*.jpg"), recursive=True)

    def __len__(self):
        # returns amount of classes
        return len(self.labels)
    
    def __getitem__(self, idx):
        label = self.labels[idx]
        img_path = self.image_filenames[idx]

        # transform image if necessary
        # image = self.get_image(img_path)
        
        return label, img_path
        
    def get_image(self, img_path):
        img = Image.open(img_path)
        img_tensor = transforms.ToTensor()(img)
        
        if self.transform:
            img_tensor = self.transform(img_tensor)
        
        return img_tensor