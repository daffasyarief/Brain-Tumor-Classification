from torchvision import transforms
from torch.utils.data import random_split, Dataset
from pathlib import Path
from PIL import Image

def invalid_images(root):
    invalid_file = []
    valid_extensions = (".jpg", ".jpeg", ".png", ".bmp")
    for file in Path(root).rglob("*"):
        if not file.is_file():
            continue
        if file.suffix.lower() not in valid_extensions:
            continue
        try:
            with Image.open(file) as img:
                img.verify()
        except Exception as e:
            invalid_file.append({"path": str(file), "error": str(e)})
    return invalid_file

def train_transform():
    return transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def test_transform():
    return transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

class TransformSubset(Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        image, label = self.subset[idx]
        if self.transform:
            image = self.transform(image)

        return image, label

def split_dataset(dataset, train_ratio=0.8, seed=42):
    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size
    generator = __import__("torch").Generator().manual_seed(seed)

    train_subset, val_subset = random_split(dataset, [train_size, val_size],  generator=generator)
    train_dataset = TransformSubset(train_subset, train_transform())
    val_dataset = TransformSubset(val_subset, test_transform())

    return train_dataset, val_dataset