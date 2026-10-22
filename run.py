from src.data import TrainDataset, TestDataset
from src.preprocess import invalid_images, split_dataset, test_transform
from src.model import HybridModel
from src.train import training_loop
from torch.utils.data import DataLoader
from pathlib import Path

train_url = Path(__file__).resolve().parent / "data" / "Training" 
test_url = Path(__file__).resolve().parent / "data" / "Testing" 

def main():
    invalid_train = invalid_images(train_url)
    invalid_test = invalid_images(test_url)
    print(f"Invalid train: {len(invalid_train)} | Invalid test: {len(invalid_test)}")

    full_train_dataset = TrainDataset(root=train_url, transform=None)
    train_dataset, val_dataset = split_dataset(full_train_dataset, train_ratio=0.8, seed=42)
    test_dataset = TestDataset(root=test_url, transform=test_transform())

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=0)
 
    print(f"Train: {len(train_dataset)} | Validation: {len(val_dataset)} | Test: {len(test_dataset)}")

    model = HybridModel(num_classes=4)
    model = training_loop(20, model=model, train_loader=train_loader, val_loader=val_loader)

if __name__ =="__main__":
    main()