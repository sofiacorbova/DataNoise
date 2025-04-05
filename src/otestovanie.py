# otestovanie modelu na jednom obrazku (moze sa neskor pouzit pri softverovej implementacii)

import torch
from torchvision import models
from PIL import Image
from torchvision.transforms import transforms
import matplotlib.pyplot as plt

from my_transform import transform_data


model = models.resnet18()

state_dict = torch.load('DataNoise/models/model_1_state_dict.pth') # , map_location=torch.device('cpu')
model.load_state_dict(state_dict)

model.eval()  # prepnutie do eval módu

# Definícia transformácií pre vstupný obrázok
preprocess = transform_data(gaus=False, mean=0.5, std=0.5, pois=False, lam=1, snp=False)

#img = Image.open('data/oxford-iiit-pet/images/great_pyrenees_147.jpg')
img = Image.open('dod1.jpg')

# Aplikácia transformácií
img_tensor = preprocess(img)

# zobrazenie upraveneho obrazka
conversion_to_image = transforms.ToPILImage()
img_for_showcase = conversion_to_image(img_tensor)

plt.subplot(1, 2, 1)
plt.title("Pôvodný obrázok")
plt.imshow(img)

plt.subplot(1, 2, 2)
plt.title("Upravený obrázok")
plt.imshow(img_for_showcase)
plt.show()

# Pridanie batch dimension (tvar sa mení na [1, 3, 224, 224])
img_tensor = img_tensor.unsqueeze(0)

# Inferencia: prechod obrázka cez model bez výpočtu gradientov
with torch.no_grad():
    output = model(img_tensor)

# vysledok
class_names = ["Cat", "Dog"] 
predicted_class = torch.argmax(output, dim=1).item() # 0 alebo 1
predicted_name = class_names[predicted_class]
print(f"Predikovaná trieda: {predicted_name}\n")
