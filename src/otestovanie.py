# otestovanie modelu na jednom obrazku (moze sa neskor pouzit pri softverovej implementacii)

import torch
import torchvision
from PIL import Image
import matplotlib.pyplot as plt

from my_transform import transform_data


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = torchvision.models.resnet18().to(device)

state_dict = torch.load('models/model_1_state_dict.pth')
model.load_state_dict(state_dict)

model.eval()  # prepnutie do eval módu

# Definícia transformácií pre vstupný obrázok
preprocess = transform_data(gaus=False, pois=False, snp=False)

img = Image.open('../data/oxford-iiit-pet/images/great_pyrenees_147.jpg')
#img = Image.open('dog1.jpg')

# Aplikácia transformácií
img_tensor = preprocess(img)

# zobrazenie upraveneho obrazka
conversion_to_image = torchvision.transforms.ToPILImage()
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
    output = model(img_tensor).to(device)


# vysledok
class_names = ["Cat", "Dog"]

# Tieto triedy sa budu pouzivat ked sa urobi model na 37 tried namiesto 2 tried
#class_names = ['Abyssinian', 'American Bulldog', 'American Pit Bull Terrier', 'Basset Hound', 'Beagle', 
#            'Bengal', 'Birman', 'Bombay', 'Boxer', 'British Shorthair', 'Chihuahua', 'Egyptian Mau', 
#            'English Cocker Spaniel', 'English Setter', 'German Shorthaired', 'Great Pyrenees', 'Havanese', 
#            'Japanese Chin', 'Keeshond', 'Leonberger', 'Maine Coon', 'Miniature Pinscher', 'Newfoundland', 
#            'Persian', 'Pomeranian', 'Pug', 'Ragdoll', 'Russian Blue', 'Saint Bernard', 'Samoyed', 'Scottish Terrier', 
#            'Shiba Inu', 'Siamese', 'Sphynx', 'Staffordshire Bull Terrier', 'Wheaten Terrier', 'Yorkshire Terrier'] 

predicted_class = torch.argmax(output, dim=1).item() # 0 alebo 1
predicted_name = class_names[predicted_class]
print(f"Predikovaná trieda: {predicted_name}\n")
