# === src/main.py (curriculum learning 150 epóch) ===

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

# === Hyperparametre ===
BATCH_SIZE = 128
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.1
EPOCHS_PER_PHASE = 30

# === Curriculum fázy ===
CURRICULUM_PHASES = [
    (0, 30, {'gaus': False, 'pois': False, 'snp': False}),   # Fáza 1: čisté obrázky
    (30, 60, {'gaus': True, 'pois': False, 'snp': False}),   # Fáza 2: Gaussian noise
    (60, 90, {'gaus': False, 'pois': True, 'snp': False}),   # Fáza 3: Poisson noise
    (90, 120, {'gaus': False, 'pois': False, 'snp': True}),  # Fáza 4: Salt & Pepper
    (120, 150, {'gaus': True, 'pois': True, 'snp': True})    # Fáza 5: Kombinovaný šum
]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# === Model ===
def create_model(num_classes=37):
    model = resnet34(weights=ResNet34_Weights.DEFAULT)
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.fc.in_features, num_classes)
    )
    return model.to(device)

# === Tréning ===
def train_and_evaluate(dataset_name, dataset, model_save_path, model, epochs=30, batch_size=128):
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
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

# === Hlavný beh ===
if __name__ == "__main__":
    model_path = f"models/model_multiclass_resnet34.pth"
    model = create_model()  # Vytvoríme model raz a budeme ho trénovať postupne

    for start_epoch, end_epoch, noise_config in CURRICULUM_PHASES:
        print(f"🛠️ Trénujem model ResNet34: epócha {start_epoch}-{end_epoch}, šum: {noise_config}")
        train_transform = transform_data(train=True, **noise_config)
        dataset = OxfordPetsMulticlass(root='./data', transform=train_transform)
        train_and_evaluate('multiclass', dataset, model_path, model=model, epochs=end_epoch-start_epoch, batch_size=BATCH_SIZE)

    print("\n✅ Kompletný tréning modelu multiclass ResNet34 hotový!")