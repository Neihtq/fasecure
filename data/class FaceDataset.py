from torch.utils.data import Dataset
from torchvision import transforms


class FaceDataset(Dataset):
    def __init__(self, root):
        super(TempDataset, self).__init__()
        self.transforms = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])