# src/data_loaders.py

import os
from torch.utils.data import Dataset
from torchvision.datasets import OxfordIIITPet
from torchvision.io import read_image
from PIL import Image

class OxfordPetsMulticlass(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform

        self.dataset = OxfordIIITPet(root=root, download=False, target_types='category')

        # Priprav mapovanie mena plemena na cislo
        self.class_names = sorted(list(set([
            os.path.splitext(file)[0].rsplit('_', 1)[0].lower() for file in os.listdir(os.path.join(root, 'oxford-iiit-pet', 'images'))
        ])))
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(self.class_names)}

        # Nacitaj obrazky a labely
        self.data = []
        images_dir = os.path.join(root, 'oxford-iiit-pet', 'images')
        for img_file in os.listdir(images_dir):
            if img_file.endswith('.jpg'):
                breed_name = img_file.rsplit('_', 1)[0].lower()
                label = self.class_to_idx[breed_name]
                self.data.append((img_file, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name, label = self.data[idx]
        img_path = os.path.join(self.root, 'oxford-iiit-pet', 'images', img_name)

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, label

    def get_class_names(self):
        return self.class_names
