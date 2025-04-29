# src/main.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet34, ResNet34_Weights
from torch.utils.data import random_split, DataLoader
from data_loaders import OxfordPetsMulticlass
from my_transform import transform_data
from torchbearer.callbacks import EarlyStopping
import torchbearer
from torchbearer import Trial

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Vytvorenie modelu pre multiclass

def create_model(num_classes=37):
    model = resnet34(weights=ResNet34_Weights.DEFAULT)
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.fc.in_features, num_classes)
    )
    return model.to(device)

# Trenovanie modelu

def train_and_evaluate(dataset_name, dataset, model_save_path, epochs=40, batch_size=128):
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = create_model(num_classes=len(dataset.get_class_names()))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3)

    early_stopping = EarlyStopping(monitor='val_loss', patience=5)

    trial = Trial(model, optimizer, criterion, metrics=['loss', 'accuracy'], callbacks=[early_stopping]).to(device)
    trial.with_generators(train_loader, val_loader, test_loader)

    print(f"\n🔵 Trénujem model '{dataset_name}'...")
    trial.run(epochs=epochs)

    print(f"✅ Vyhodnotenie '{dataset_name}' na testovacej sade:")
    trial.evaluate(data_key=torchbearer.TEST_DATA)

    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    print(f"💾 Model uložený: {model_save_path}\n")

# --- Hlavný beh programu ---

if __name__ == "__main__":
    clean_transform = transform_data()

    dataset = OxfordPetsMulticlass(root='./data', transform=clean_transform)

    model_path = "models/model_multiclass_resnet34.pth"
    print(f"🛠️ Trénujem model ResNet34 na viac-triednu klasifikáciu")
    train_and_evaluate('multiclass', dataset, model_path, epochs=40, batch_size=128)

    print("\n✅ Tréning multiclass ResNet34 modelu hotový!")