from torchvision import transforms
import torchvision.transforms.functional as F
import random
import torch

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.mean = mean
        self.std = std
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

class AddSaltPepperNoise(object):
    def __init__(self, amount=0.01):
        self.amount = amount

    def __call__(self, tensor):
        noisy = tensor.clone()
        num_salt = int(self.amount * tensor.nelement())
        coords = [torch.randint(0, i - 1, (num_salt,)) for i in tensor.shape]
        noisy[tuple(coords)] = 1

        num_pepper = int(self.amount * tensor.nelement())
        coords = [torch.randint(0, i - 1, (num_pepper,)) for i in tensor.shape]
        noisy[tuple(coords)] = 0

        return noisy

class AddPoissonNoise(object):
    def __call__(self, tensor):
        # Prenormovanie do [0,1]
        tensor = (tensor + 1.0) / 2.0  # pôvodne bolo -1..1 -> teraz 0..1
        vals = len(torch.unique(tensor))
        vals = 2 ** torch.ceil(torch.log2(torch.tensor(vals, dtype=torch.float32)))
        noisy = torch.poisson(tensor * vals) / float(vals)
        noisy = noisy * 2.0 - 1.0  # vrátime späť do -1..1
        return noisy


def transform_data(gaus=False, pois=False, snp=False, std=0.1):
    noise_transforms = []
    if gaus:
        noise_transforms.append(AddGaussianNoise(std=std))
    if pois:
        noise_transforms.append(AddPoissonNoise())
    if snp:
        noise_transforms.append(AddSaltPepperNoise(amount=0.01))

    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        *noise_transforms,  # Aplikuj šumy, ak sú vybraté
        transforms.Normalize((0.5,), (0.5,))
    ])