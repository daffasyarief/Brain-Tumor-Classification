from src.data import TrainDataset, TestDataset
from src.preprocess import train_transform, test_transform
from src.model import HybridModel
from src.train 
from torch.utils.data import DataLoader
from pathlib import Path

train_url = Path(__file__).resolve().parent.parent / "data" / "Training" 
test_url = Path(__file__).resolve().parent.parent / "data" / "Testing" 

def main():
    train_dataset = TrainDataset(root=train_url, transform=train_transform)
    test_dataset = TrainDataset(root=test_url, transform=train_transform)
    TrainLoader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    TestLoader = DataLoader(test_dataset, batch_size=32, shuffle=True, num_workers=0)
    

if __name__ =="__main__":
    main()