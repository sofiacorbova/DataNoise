# === src/main.py (curriculum learning 150 epóch + CSV logging) ===

import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torchvision.models import resnet34, ResNet34_Weights
from torch.utils.data import DataLoader
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
    (0, 30, {'gaus': False, 'pois': False, 'snp': False}),
    (30, 60, {'gaus': True, 'pois': False, 'snp': False}),
    (60, 90, {'gaus': False, 'pois': True, 'snp': False}),
    (90, 120, {'gaus': False, 'pois': False, 'snp': True}),
    (120, 150, {'gaus': True, 'pois': True, 'snp': True})
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
def train_and_evaluate(dataset_name, dataset, model_save_path, model, epochs=30, batch_size=128, phase_desc="", noise_config={}):
    train_dataset = OxfordPetsMulticlass(root='./data', transform=dataset.transform, split='train')
    val_dataset   = OxfordPetsMulticlass(root='./data', transform=dataset.transform, split='val')
    test_dataset  = OxfordPetsMulticlass(root='./data', transform=dataset.transform, split='test')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)

    trial = Trial(model, optimizer, criterion, metrics=['loss', 'accuracy'], callbacks=[early_stopping]).to(device)
    trial.with_generators(train_loader, val_loader, test_loader)

    print(f"\n🔵 Trénujem model '{dataset_name}'...")
    history = trial.run(epochs=epochs)

    print(f"✅ Vyhodnotenie '{dataset_name}' na testovacej sade:")
    trial.evaluate(data_key=torchbearer.TEST_DATA)

    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save({
        'model_state': model.state_dict(),
        'train_noise_config': noise_config
    }, model_save_path)
    print(f"💾 Model uložený: {model_save_path}")

    # Logovanie metrík
    metrics_path = "results/training_metrics.csv"
    rows = []
    for epoch, record in enumerate(history):
        rows.append({
            "Phase": phase_desc,
            "Epoch": epoch,
            "Train_Loss": record['loss'],
            "Train_Accuracy": record['accuracy'],
            "Val_Loss": record['val_loss'],
            "Val_Accuracy": record['val_accuracy']
        })

    df = pd.DataFrame(rows)
    if os.path.exists(metrics_path):
        df_old = pd.read_csv(metrics_path)
        df = pd.concat([df_old, df], ignore_index=True)

    df.to_csv(metrics_path, index=False)
    print(f"📊 Výsledky uložené do {metrics_path}")

# === Hlavný beh ===
if __name__ == "__main__":
    model_path = f"models/model_multiclass_resnet34.pth"
    model = create_model()  # Vytvoríme model raz

    for start_epoch, end_epoch, noise_config in CURRICULUM_PHASES:
        print(f"🛠️ Trénujem model ResNet34: epócha {start_epoch}-{end_epoch}, šum: {noise_config}")
        train_transform = transform_data(train=True, **noise_config)
        dataset = OxfordPetsMulticlass(root='./data', transform=train_transform)
        phase_desc = f"{start_epoch}-{end_epoch} {','.join([k for k, v in noise_config.items() if v]) or 'clean'}"
        train_and_evaluate('multiclass', dataset, model_path, model=model, epochs=end_epoch-start_epoch, batch_size=BATCH_SIZE, phase_desc=phase_desc, noise_config=noise_config)

    print("\n✅ Kompletný tréning modelu multiclass ResNet34 hotový!")