import torch
import torchvision
import torchbearer
from torchbearer import Trial
from torchvision.models import resnet18, ResNet18_Weights
from my_transform import transform_data
from data_loaders import get_data_loaders, OxfordPetsBinary
import os

torch.manual_seed(17)

# Hyperparametre
BATCH_SIZE = 512
EPOCHS = 10
LR = 0.001
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Definícia transformácií
clean_transform = transform_data(gaus=False, pois=False, snp=False)
gaussian_transform = transform_data(gaus=True, std=0.2, pois=False, snp=False)
poisson_transform = transform_data(gaus=False, pois=True, lam=20, snp=False)
saltpepper_transform = transform_data(gaus=False, pois=False, snp=True, salt_prob=0.03, pepper_prob=0.03)
combined_transform = transform_data(gaus=True, pois=True, snp=True)

# Datasety
datasets = {
    'clean': OxfordPetsBinary(root='./data', transform=clean_transform),
    'gaussian': OxfordPetsBinary(root='./data', transform=gaussian_transform),
    'poisson': OxfordPetsBinary(root='./data', transform=poisson_transform),
    'saltpepper': OxfordPetsBinary(root='./data', transform=saltpepper_transform),
    'combined': OxfordPetsBinary(root='./data', transform=combined_transform)
}

# Funkcia na vytvorenie modelu
def create_model():
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)  # 2 triedy (Cat/Dog)
    return model.to(device)

# Funkcia na tréning modelu
def train_and_evaluate(dataset_name, dataset, model_save_path):
    train_loader, val_loader, test_loader = get_data_loaders(dataset, batch_size=BATCH_SIZE)

    model = create_model()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    trial = Trial(model=model, optimizer=optimizer, criterion=criterion, metrics=['loss', 'accuracy'],
                  callbacks=[torchbearer.callbacks.EarlyStopping(patience=5)]).to(device)
    trial.with_generators(train_generator=train_loader, val_generator=val_loader, test_generator=test_loader)

    print(f"🔵 Trénujem model na '{dataset_name}' dátach...")
    trial.run(epochs=EPOCHS)
    
    print(f"✅ Výsledky '{dataset_name}' modelu na testovacích dátach:")
    results = trial.evaluate(data_key=torchbearer.TEST_DATA)
    print(results)

    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    print(f"💾 Model uložený do {model_save_path}\n")

# Hlavná funkcia
def main():
    for name, dataset in datasets.items():
        train_and_evaluate(name, dataset, f"models/model_{name}.pth")

if __name__ == "__main__":
    main()