# src/visualize_results.py

import torch
import torchvision
import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage
import pandas as pd
from my_transform import transform_data, AddGaussianNoise, AddPoissonNoise, AddSaltPepperNoise
from data_loaders import OxfordPetsMulticlass
from torchvision.models import resnet34, ResNet34_Weights
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def recover_image(tensor):
    return (tensor + 1.0) / 2.0

# Load model
def load_model(model_path, num_classes):
    model = resnet34(weights=ResNet34_Weights.DEFAULT)
    model.fc = torch.nn.Sequential(
        torch.nn.Dropout(0.5),
        torch.nn.Linear(model.fc.in_features, num_classes)
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model.to(device)

# Predict top-k classes
def predict_topk(model, img_tensor, k=3):
    img_tensor = img_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img_tensor)
    probabilities = torch.nn.functional.softmax(output, dim=1)
    confidences, predictions = torch.topk(probabilities, k)
    return predictions.squeeze(0).tolist(), confidences.squeeze(0).tolist()

# Visualize predictions and export to CSV
def show_examples(num_examples=5):
    to_pil = ToPILImage()

    model_path = 'models/model_multiclass_resnet34.pth'

    dataset = OxfordPetsMulticlass(root='./data', transform=transform_data(gaus=False, pois=False, snp=False))
    class_names = dataset.get_class_names()

    model = load_model(model_path, num_classes=len(class_names))

    indices = torch.randperm(len(dataset))[:num_examples]

    os.makedirs("results", exist_ok=True)

    csv_data = []

    for idx in indices:
        img, true_label = dataset[idx]
        img = img.to('cpu')

        fig, axes = plt.subplots(2, 4, figsize=(24, 10))

        variants = [
            ('Original', img),
            ('Gaussian Noise', AddGaussianNoise(mean=0.0, std=0.2)(img)),
            ('Poisson Noise', AddPoissonNoise()((img + 1.0)/2.0) * 2.0 - 1.0),
            ('Salt & Pepper', AddSaltPepperNoise(amount=0.05)(img))
        ]

        for i, (title, variant_img) in enumerate(variants):
            axes[0, i].imshow(to_pil(recover_image(variant_img)))
            axes[0, i].set_title(title, fontsize=14)
            axes[0, i].axis('off')

            preds, confs = predict_topk(model, variant_img, k=3)

            # Save to CSV
            csv_data.append({
                'ImageIndex': idx.item() if torch.is_tensor(idx) else idx,
                'Variant': title,
                'TrueLabel': class_names[true_label],
                'Top1Prediction': class_names[preds[0]],
                'Top1Confidence': confs[0]*100,
                'Top2Prediction': class_names[preds[1]],
                'Top2Confidence': confs[1]*100,
                'Top3Prediction': class_names[preds[2]],
                'Top3Confidence': confs[2]*100,
                'Correct': int(preds[0] == true_label)
            })

            # Text formatting with color
            color = 'green' if preds[0] == true_label else 'red'

            text = f"True: {class_names[true_label]}\n"
            for j in range(len(preds)):
                text += f"{j+1}. {class_names[preds[j]]} ({confs[j]*100:.1f}%)\n"

            axes[1, i].text(0.5, 0.5, text.strip(), fontsize=10, ha='center', va='center', color=color)
            axes[1, i].axis('off')

        plt.suptitle(f"Predictions for Sample {idx}", fontsize=20)
        plt.tight_layout()

        save_path = f"results/example_{idx}.png"
        plt.savefig(save_path)
        plt.close(fig)
        print(f"✅ Obrázok uložený: {save_path}")

    # Save CSV results
    csv_df = pd.DataFrame(csv_data)
    csv_df.to_csv("results/visualization_results.csv", index=False)
    print("\n✅ CSV s vysledkami ulozeny: results/visualization_results.csv")

if __name__ == "__main__":
    show_examples(num_examples=5)