# === src/my_transform.py ===

import torchvision.transforms as T
import torch
import random

# === Noise transforms ===

class AddGaussianNoise(torch.nn.Module):
    def __init__(self, mean=0., std=0.05):
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, tensor):
        return tensor + torch.randn_like(tensor) * self.std + self.mean

class AddPoissonNoise(torch.nn.Module):
    def __init__(self, lam=20):
        super().__init__()
        self.lam = lam

    def forward(self, tensor):
        tensor = (tensor + 1.0) / 2.0  # Normalize to [0,1]
        vals = self.lam
        noisy = torch.poisson(tensor * vals) / vals
        noisy = noisy * 2.0 - 1.0  # Return back to [-1,1]
        return noisy


class AddSaltPepperNoise(torch.nn.Module):
    def __init__(self, amount=0.05):
        super().__init__()
        self.amount = amount

    def forward(self, tensor):
        c, h, w = tensor.shape
        num_salt = int(self.amount * h * w / 2)
        num_pepper = int(self.amount * h * w / 2)

        coords = [torch.randint(0, i, (num_salt + num_pepper,)) for i in (h, w)]
        salt_coords = coords[0][:num_salt], coords[1][:num_salt]
        pepper_coords = coords[0][num_salt:], coords[1][num_salt:]

        tensor[:, salt_coords[0], salt_coords[1]] = 1
        tensor[:, pepper_coords[0], pepper_coords[1]] = -1
        return tensor

# === Main transform ===

def transform_data(gaus=False, pois=False, snp=False, train=False):
    transforms = [
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]

    if train:
        transforms.insert(1, T.RandomHorizontalFlip())
        transforms.insert(2, T.RandomRotation(degrees=15))
        transforms.insert(3, T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05))
        transforms.insert(4, T.RandomResizedCrop(size=224, scale=(0.9, 1.0)))

    if gaus:
        transforms.append(AddGaussianNoise())
    if pois:
        transforms.append(AddPoissonNoise())
    if snp:
        transforms.append(AddSaltPepperNoise())

    return T.Compose(transforms)
