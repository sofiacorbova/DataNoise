import torch
import torchvision
import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage
from my_transform import transform_data, add_gaussian_noise, add_poisson_noise, add_salt_and_pepper_noise
from data_loaders import OxfordPetsBinary
from torchvision.models import resnet18, ResNet18_Weights
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Načítanie modelu
def load_model(model_path):
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model.to(device)

# Predikcia obrázka
def predict_image(model, img_tensor):
    img_tensor = img_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img_tensor)
    probabilities = torch.nn.functional.softmax(output, dim=1)
    confidence, predicted = torch.max(probabilities, dim=1)
    return predicted.item(), confidence.item()

# Vizualizácia výsledkov
def show_examples(num_examples=5):
    class_names = ["Cat", "Dog"]
    to_pil = ToPILImage()

    # Načítanie všetkých modelov
    model_paths = {
        'clean': 'models/model_clean.pth',
        'gaussian': 'models/model_gaussian.pth',
        'poisson': 'models/model_poisson.pth',
        'saltpepper': 'models/model_saltpepper.pth',
        'combined': 'models/model_combined.pth'
    }
    models = {name: load_model(path) for name, path in model_paths.items()}

    # Dataset (čisté obrázky)
    dataset = OxfordPetsBinary(root='./data', transform=transform_data(gaus=False, pois=False, snp=False))

    indices = torch.randperm(len(dataset))[:num_examples]

    os.makedirs("results", exist_ok=True)

    for idx in indices:
        img, true_label = dataset[idx]
        img = img.to('cpu')

        fig, axes = plt.subplots(2, 6, figsize=(25, 8))

        # Varianty obrázka: čistý + šumy
        variants = [
            ('Original', img),
            ('Gaussian Noise', add_gaussian_noise(img, mean=0.0, std=0.2)),
            ('Poisson Noise', add_poisson_noise(img, lam=20)),
            ('Salt & Pepper', add_salt_and_pepper_noise(img, salt_prob=0.05, pepper_prob=0.05)),
            ('Combined Noise', add_salt_and_pepper_noise(add_poisson_noise(add_gaussian_noise(img, std=0.2), lam=20), salt_prob=0.05, pepper_prob=0.05))
        ]

        for i, (title, variant_img) in enumerate(variants):
            axes[0, i].imshow(to_pil(variant_img))
            axes[0, i].set_title(title, fontsize=14)
            axes[0, i].axis('off')

            pred_text = ""
            for model_name, model in models.items():
                pred, conf = predict_image(model, variant_img)
                pred_text += f"{model_name}: {class_names[pred]} ({conf*100:.1f}%)\n"

            axes[1, i].text(0.5, 0.5, pred_text, fontsize=10, ha='center', va='center')
            axes[1, i].axis('off')

        axes[0, 5].remove()
        axes[1, 5].remove()

        plt.suptitle(f"True Label: {class_names[true_label]}", fontsize=16)
        plt.tight_layout()

        plt.savefig(f"results/example_{idx}.png")
        plt.show()

if __name__ == "__main__":
    show_examples(num_examples=5)