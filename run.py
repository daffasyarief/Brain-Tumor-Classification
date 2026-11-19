import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import models
from src.data import TrainDataset, TestDataset
from src.preprocess import invalid_images, split_dataset, test_transform
from src.model import HybridModel
from src.train import training_loop

train_url = Path(__file__).resolve().parent / "data" / "Training"
test_url = Path(__file__).resolve().parent / "data" / "Testing"
SEED = 42


def set_reproducible_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

def main():
    set_reproducible_seed(SEED)

    invalid_train = invalid_images(train_url)
    invalid_test = invalid_images(test_url)
    print(f"Invalid train: {len(invalid_train)} | Invalid test: {len(invalid_test)}")

    full_train_dataset = TrainDataset(root=train_url, transform=None)
    train_dataset, val_dataset = split_dataset(full_train_dataset, train_ratio=0.8, seed=SEED)
    test_dataset = TestDataset(root=test_url, transform=test_transform())

    loader_generator = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0, generator=loader_generator)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=0)

    print(f"Train: {len(train_dataset)} | Validation: {len(val_dataset)} | Test: {len(test_dataset)}")

    model1 = HybridModel(num_classes=4)
    model1 = training_loop(20, model=model1, train_loader=train_loader, val_loader=val_loader)
    model2 = models.convnext_tiny(pretrained=True)
    model2 = training_loop(20, model=model2, train_loader=train_loader, val_loader=val_loader)
    model3 = models.resnet50(pretrained=True)
    model3 = training_loop(20, model=model3, train_loader=train_loader, val_loader=val_loader)

if __name__ =="__main__":
    main()