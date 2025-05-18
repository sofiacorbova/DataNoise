# === evaluate_models.py ===
# Vyhodnotenie multiclass modelov na rôznych testovacích šumoch pomocou accuracy, precision, recall, F1, MCC, confusion matrix

import os
import torch
import pandas as pd
import numpy as np
from torchvision.models import resnet34, ResNet34_Weights
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, matthews_corrcoef
from data_loaders import OxfordPetsMulticlass
from my_transform import transform_data

from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 37
BATCH_SIZE = 128


def load_model(path, num_classes=NUM_CLASSES):
    model = resnet34(weights=ResNet34_Weights.DEFAULT)
    model.fc = torch.nn.Sequential(
        torch.nn.Dropout(0.5),
        torch.nn.Linear(model.fc.in_features, num_classes)
    )
    checkpoint = torch.load(path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state'])
    return model.to(DEVICE).eval()


def evaluate_model(model, test_loader):
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    acc = np.mean(y_true == y_pred) * 100
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)

    return acc, report, cm, mcc


def plot_confusion_matrix(cm, model_name, variant, class_names, save_dir):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix - {model_name} ({variant})")
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    fname = f"confmat_{model_name}_{variant}.png".replace(" ", "_")
    plt.savefig(os.path.join(save_dir, fname))
    plt.close()


if __name__ == "__main__":
    results = []
    model_dir = "models_curr"
    save_dir = "resultscurr"
    os.makedirs(save_dir, exist_ok=True)

    test_noise_variants = [
        {"name": "clean", "gaus": False, "pois": False, "snp": False, "std": 0.0, "lam": 0, "amount": 0.0},
        {"name": "gaussian", "gaus": True, "pois": False, "snp": False, "std": 0.10, "lam": 0, "amount": 0.0},
        {"name": "poisson",  "gaus": False, "pois": True,  "snp": False, "std": 0.0, "lam": 10, "amount": 0.0},
        {"name": "saltpepper", "gaus": False, "pois": False, "snp": True, "std": 0.0, "lam": 0, "amount": 0.05},
        {"name": "combined", "gaus": True, "pois": True, "snp": True, "std": 0.10, "lam": 10, "amount": 0.05},
        {"name": "combined2", "gaus": True, "pois": True, "snp": True, "std": 0.01, "lam": 100, "amount": 0.005}
    ]

    for fname in os.listdir(model_dir):
        if not fname.endswith(".pth"):
            continue

        model_path = os.path.join(model_dir, fname)
        print(f"\nVyhodnocujem model: {fname}")
        model = load_model(model_path)

        for variant in test_noise_variants:
            print(f"Testovanie s variantom šumu: {variant['name']}")

            transform = transform_data(
                train=True,
                gaus=variant['gaus'],
                pois=variant['pois'],
                snp=variant['snp'],
                std=variant['std'],
                lam=variant['lam'],
                amount=variant['amount']
            )
            test_dataset = OxfordPetsMulticlass(root="./data", transform=transform, split="test")
            test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
            class_names = test_dataset.get_class_names()

            acc, report, cm, mcc = evaluate_model(model, test_loader)

            precision = report['macro avg']['precision'] * 100
            recall = report['macro avg']['recall'] * 100
            f1 = report['macro avg']['f1-score'] * 100

            results.append({
                "Model": fname,
                "Test_Noise": variant['name'],
                "Std": variant['std'],
                "Lam": variant['lam'],
                "SP_Amount": variant['amount'],
                "Accuracy (%)": acc,
                "Macro Precision (%)": precision,
                "Macro Recall (%)": recall,
                "Macro F1 (%)": f1,
                "MCC": mcc
            })

            plot_confusion_matrix(cm, model_name=fname.replace(".pth", ""), variant=variant['name'], class_names=class_names, save_dir=save_dir)

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(save_dir, "model_multiclass_metrics_all.csv"), index=False)
    print("\nVýsledky uložené do results/model_multiclass_metrics_all.csv")
