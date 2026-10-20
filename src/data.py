from pathlib import Path
import os 
from torch.utils.data import Dataset
from PIL import Image

data_url = Path(__file__).resolve().parent.parent / "data"

label_mapping = {
    "glioma": 0,
    "meningioma": 1,
    "notumor": 2,
    "pituitary": 3
}

class TrainDataset(Dataset):
    def __init__(self, root, transform=None):
        self.transform = transform
        self.image = []
        self.image_labels = []
        for subfolder in os.listdir(root):
            label = label_mapping[subfolder]
            folder_path = os.path.join(root, subfolder)
            for file in os.listdir(folder_path):
                self.image.append(os.path.join(folder_path, file))
                self.image_labels.append(label)
    def __len__(self):
        return len(self.image)
    
    def __getitem__(self, index):
        img_path = self.image[index]
        img = Image.open(img_path)
        if self.transform:
            img = self.transform(img)
        label = self.image_labels[index]
        return img, label

class TestDataset(Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        self.image = 

    def __len__(self):
            return len(self.image)
    
    def __getitem__(self, index):
        img_path = self.image[index]
        img = Image.open(img_path)
        if self.transform:
            img = self.transform(img)
        label = self.image_labels[index]
        return img, label       
    