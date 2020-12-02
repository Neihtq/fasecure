from torch.utils.data import Dataset
from torchvision.transforms import transforms
import os
from os.path import join
from PIL import Image
import glob

class RegistrationDataset(Dataset):
    """ Traffic Signs dataset."""
    
    # define the constructor of this dataset object
    def __init__(self, dataset_folder, datatype):
        """
        Args:
            dataset_folder(string): Path to the main folder containing the dataset.
        """
        self.dataset_folder = dataset_folder
        self.datatype = datatype
        
        self.image_filenames = self.read_file_paths()
        
        self.img_transforms = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])
        
    
    def read_file_paths(self):
        return glob.glob(join(self.dataset_folder, "**/*."+self.datatype), recursive=True)
            
    
    # methods to override, length and get_item
    
    def __len__(self):
        return len(self.image_filenames)
    
    def __getitem__(self, idx):
        
        filename = self.image_filenames[idx]
         # -- Input --
        image = Image.open(filename)
        image = self.img_transforms(image)

        # -- Label --
        ending = os.path.basename(filename)
        label = ending.split("_")[0]
        
        return image, label