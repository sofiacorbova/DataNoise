# === src/main.py (curriculum learning 200 epóch + CSV logging + re-eval) ===

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
EPOCHS_PER_PHASE = 40  # zvýšené z 30 na 40 pre lepšiu generalizáciu

# === Curriculum fázy ===
CURRICULUM_PHASES = [
    (0, 60,  {'gaus': False, 'pois': False, 'snp': False}),   # 60 epôch na čisté
    (60, 100, {'gaus': True, 'pois': False, 'snp': False}),   # 40 epôch na gaussian
    (100, 130, {'gaus': False, 'pois': True, 'snp': False}),  # 30 epôch na poisson
    (130, 160, {'gaus': False, 'pois': False, 'snp': True}),  # 30 epôch na salt & pepper
    (160, 200, {'gaus': True, 'pois': True, 'snp': True})     # 40 epôch na kombinovaný
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

    metrics_path = "results/training_metrics.csv"
    rows = []
    for epoch, record in enumerate(history):
        # metrics = record.get(torchbearer.METRICS, {})
        rows.append({
            "Phase": phase_desc,
            "Epoch": epoch,
            "Train_Loss": record.get('loss'),
            "Train_Accuracy": record.get('accuracy'),
            "Val_Loss": record.get('val_loss'),
            "Val_Accuracy": record.get('val_accuracy')
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

    # Re-eval na čistých dátach po všetkých fázach
    print("\n🔍 Re-eval na originálnych dátach po kombinovanej fáze")
    clean_transform = transform_data(train=False, gaus=False, pois=False, snp=False)
    test_dataset = OxfordPetsMulticlass(root='./data', transform=clean_transform, split='test')
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    model.eval()

    correct = total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"✅ Presnosť na čistých testovacích dátach: {100 * correct / total:.2f}%")
    print("\n✅ Kompletný tréning modelu multiclass ResNet34 hotový!")