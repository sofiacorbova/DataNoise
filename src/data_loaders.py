# === src/data_loaders.py ===
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import OxfordIIITPet
from sklearn.model_selection import train_test_split

class OxfordPetsMulticlass(Dataset):
    def __init__(self, root, transform=None, split='train'):
        self.root = root
        self.transform = transform
        self.split = split  # 'train', 'val', 'test'
        self.data = []
        self.class_to_idx = {}

        self._prepare_data()

    def _prepare_data(self):
        image_dir = os.path.join(self.root, 'oxford-iiit-pet', 'images')
        all_images = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]

        # Všetky triedy (plemená)
        class_names = sorted(list(set(fname.rsplit('_', 1)[0].lower() for fname in all_images)))
        self.class_to_idx = {name: idx for idx, name in enumerate(class_names)}

        images_by_class = {name: [] for name in class_names}
        for fname in all_images:
            class_name = fname.rsplit('_', 1)[0].lower()
            images_by_class[class_name].append(fname)

        for class_name, images in images_by_class.items():
            images.sort()
            train_imgs, test_imgs = train_test_split(images, test_size=0.2, random_state=42)
            val_imgs, test_imgs = train_test_split(test_imgs, test_size=0.5, random_state=42)

            if self.split == 'train':
                selected = train_imgs
            elif self.split == 'val':
                selected = val_imgs
            else:
                selected = test_imgs

            for img in selected:
                path = os.path.join(image_dir, img)
                label = self.class_to_idx[class_name]
                self.data.append((path, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path, label = self.data[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

    def get_class_names(self):
        return list(self.class_to_idx.keys())
