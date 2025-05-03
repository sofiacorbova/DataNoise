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
    def forward(self, tensor):
        tensor = (tensor + 1.0) / 2.0  # z [-1, 1] na [0, 1]
        tensor = torch.clamp(tensor, min=0.0)  # zabezpečí nezáporné hodnoty
        vals = 2 ** torch.ceil(torch.log2(torch.tensor(tensor.numel(), dtype=torch.float32)))
        noisy = torch.poisson(tensor * vals) / vals
        noisy = noisy * 2.0 - 1.0  # späť na [-1, 1]
        return torch.clamp(noisy, -1.0, 1.0)

class AddSaltPepperNoise(torch.nn.Module):
    def __init__(self, amount=0.03):
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

        noise_transforms = []
        if gaus:
            noise_transforms.append(AddGaussianNoise(std=0.05))
        if pois:
            noise_transforms.append(AddPoissonNoise())
        if snp:
            noise_transforms.append(AddSaltPepperNoise(amount=0.03))

        if noise_transforms:
            transforms.append(T.RandomApply(noise_transforms, p=0.3))

    return T.Compose(transforms)