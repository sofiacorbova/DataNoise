# === src/visualize_results.py ===
import torch
import torchvision
import matplotlib.pyplot as plt
import pandas as pd
from torchvision.transforms import ToPILImage
from data_loaders import OxfordPetsMulticlass
from my_transform import AddGaussianNoise, AddPoissonNoise, AddSaltPepperNoise
from torchvision.models import resnet34, ResNet34_Weights
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def visualization_transform():
    return torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor()
    ])
    

def load_model(model_path, num_classes):
    checkpoint = torch.load(model_path, map_location=device)
    model = resnet34(weights=ResNet34_Weights.DEFAULT)
    model.fc = torch.nn.Sequential(
        torch.nn.Dropout(0.5),
        torch.nn.Linear(model.fc.in_features, num_classes)
    )
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    return model.to(device), checkpoint.get('train_noise_config', {})

def predict_image(model, img_tensor):
    img_tensor = img_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img_tensor)
    probs = torch.nn.functional.softmax(output, dim=1).squeeze()
    sorted_probs, indices = torch.sort(probs, descending=True)
    return indices.tolist(), probs[indices].tolist()

def show_examples(num_examples=5):
    to_pil = ToPILImage()
    model_path = 'models/model_multiclass_resnet34.pth'

    dataset = OxfordPetsMulticlass(root='./data', transform=visualization_transform())
    class_names = dataset.get_class_names()

    # === Načítaj model + šum z tréningu ===
    checkpoint = torch.load(model_path, map_location=device)
    train_noise_config = checkpoint.get('train_noise_config', {})
    model = resnet34(weights=ResNet34_Weights.DEFAULT)
    model.fc = torch.nn.Sequential(
        torch.nn.Dropout(0.5),
        torch.nn.Linear(model.fc.in_features, len(class_names))
    )
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    model = model.to(device)

    os.makedirs("results", exist_ok=True)

    results_file = "results/visualization_results.csv"
    all_rows = []

    indices = torch.randperm(len(dataset))[:num_examples]

    for idx in indices:
        img, true_label = dataset[idx]
        img = img.to('cpu')

        fig, axes = plt.subplots(2, 5, figsize=(30, 10))  # Horný riadok: obrázky, spodný riadok: texty

        variant_imgs = [
            ('Original', img.clone()),
            ('Gaussian Noise', AddGaussianNoise(std=0.05)(img.clone())),
            ('Poisson Noise', AddPoissonNoise()(img.clone())),
            ('Salt & Pepper', AddSaltPepperNoise(amount=0.03)(img.clone())),
            ('Combined Noise', AddSaltPepperNoise(amount=0.03)(AddPoissonNoise()(AddGaussianNoise(std=0.1)(img.clone()))))
        ]

        train_noise_str = ", ".join([k for k, v in train_noise_config.items() if v]) or "Clean"

        for i, (variant_name, variant_img) in enumerate(variant_imgs):
            # Horný riadok obrázky
            axes[0, i].imshow(to_pil(variant_img))
            axes[0, i].set_title(variant_name, fontsize=14)
            axes[0, i].axis('off')

            preds, confs = predict_image(model, variant_img)
            top_pred = preds[0]
            top_conf = confs[0]
            correct = top_pred == true_label
            color = 'green' if correct else 'red'

            pred_label = class_names[top_pred]
            top3_text = "\n".join([
                f"{class_names[p]} ({c*100:.1f}%)" for p, c in zip(preds[:3], confs[:3])
            ])

            # Spodný riadok text
            text = f"Pred: {pred_label} ({top_conf*100:.1f}%)\n{top3_text}"
            axes[1, i].text(0.5, 0.5, text, fontsize=12, color=color, ha='center', va='center', wrap=True)
            axes[1, i].axis('off')

            row = {
                "Image_Index": idx.item(),
                "Variant": variant_name,
                "True_Label": class_names[true_label],
                "Predicted_Label": pred_label,
                "Confidence": f"{top_conf*100:.2f}%",
                "Top3_Predictions": "; ".join([f"{class_names[p]} ({c*100:.1f}%)" for p, c in zip(preds[:3], confs[:3])]),
                "Train_Noise": train_noise_str,
                "Test_Noise": variant_name
            }
            all_rows.append(row)

        plt.suptitle(f"🖼️ True Label: {class_names[true_label]} | Trained on: {train_noise_str}", fontsize=20)
        plt.tight_layout()
        plt.savefig(f"results/example_{idx}.png")
        plt.close(fig)
        print(f"✅ Obrázok uložený: results/example_{idx}.png")

    # CSV APPEND
    if os.path.exists(results_file):
        old = pd.read_csv(results_file)
        df = pd.concat([old, pd.DataFrame(all_rows)], ignore_index=True)
    else:
        df = pd.DataFrame(all_rows)

    df.to_csv(results_file, index=False)
    print(f"📄 Výsledky uložené v {results_file}")

if __name__ == "__main__":
    show_examples(num_examples=5)
