import os
import torch
from torch.utils.data import random_split, DataLoader, Dataset
from torchvision.datasets import OxfordIIITPet
from torchvision.io import read_image
from PIL import Image

# Funkcia na rozdelenie datasetu na trenovacie, validacne a testovacie data loadery.
def get_data_loaders(data, train_size: float=0.8, val_size: float=0.1, test_size: float=0.1, batch_size: int=32):
    train_size = int(train_size * len(data)) 
    val_size = int(val_size * len(data)) 
    test_size = len(data) - (train_size + val_size) 

    train_dataset, val_dataset, test_dataset = random_split(data, [train_size, val_size, test_size]) 

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

# Naša vlastná verzia datasetu na binary klasifikáciu: Cat vs Dog
class OxfordPetsBinary(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        
        self.dataset = OxfordIIITPet(root=root, download=True, target_types="category")
        
        self.data = []
        list_path = os.path.join(root, 'oxford-iiit-pet', 'annotations', 'list.txt')
        with open(list_path, 'r') as f:
            lines = f.readlines()[6:]
        
        for line in lines:
            parts = line.strip().split()
            img_name = parts[0] + '.jpg'
            class_label = int(parts[2])
            binary_label = 0 if class_label == 2 else 1  # 0=Cat, 1=Dog
            self.data.append((img_name, binary_label))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_name, label = self.data[idx]
        img_path = os.path.join(self.root, 'oxford-iiit-pet', 'images', img_name)

        image = Image.open(img_path).convert("RGB")  # načítať ako PIL image
        if self.transform:
            image = self.transform(image)

        return image, label
