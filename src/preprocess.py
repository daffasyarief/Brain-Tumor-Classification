from torchvision import transforms
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

def resize_image():
    pass

def train_transform():
    return transforms.Compose([
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
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])