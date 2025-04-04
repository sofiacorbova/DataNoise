# otestovanie modelu na jednom obrazku (moze sa neskor pouzit pri softverovej implementacii)

import torch
from torchvision import models
from PIL import Image

from my_transform import transform_data


model = models.resnet18(pretrained=False)

# state_dict = torch.load('model_state_dict.pth') # , map_location=torch.device('cpu')
# model.load_state_dict(state_dict)

model.eval()  # prepnutie do eval módu

# Definícia transformácií pre vstupný obrázok
preprocess = transform_data(gaus=False, pois=False, snp=False)

img = Image.open('path_to_image.jpg').convert('RGB')

# Aplikácia transformácií
img_tensor = preprocess(img)

# Pridanie batch dimension (tvar sa mení na [1, 3, 224, 224])
img_tensor = img_tensor.unsqueeze(0)

# Inferencia: prechod obrázka cez model bez výpočtu gradientov
with torch.no_grad():
    output = model(img_tensor)

# Prevod výstupu na pravdepodobnosti pomocou softmax (voliteľné)
probabilities = torch.nn.functional.softmax(output[0], dim=0)

print(probabilities)
