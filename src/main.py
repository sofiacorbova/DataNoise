import torch, torchvision, torchbearer
from torchbearer import Trial
from torchvision.models import ResNet18_Weights
import matplotlib.pyplot as plt

from my_transform import transform_data
from data_loaders import get_data_loaders

torch.manual_seed(17)



### Hyperparametre ###
BATCH_SIZE = 512
EPOCHS = 3
LR = 0.01
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

### MACKY_A_PSY dataset (XX tried, XX kusov XXxXX obrazkov) ###
data = torchvision.datasets.OxfordIIITPet(root='./data', download=True, target_types="binary-category", transform=transform_data(gaus=False, pois=False, snp=False))
train_loader, val_loader, test_loader = get_data_loaders(data, train_size=0.8, val_size=0.1, test_size=0.1, batch_size=BATCH_SIZE)

### MODEL ###
# bol error kvoli nejakemu parametru state, pridal sa wrapper co odstrani state
class ResNetWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, x, state=None):
        return self.model(x)

base_model = torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT) # nacitanie resnet18
#base_model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
model = ResNetWrapper(base_model).to(device) # upraveny resnet 

L = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)



def main() -> None:
    """
    Hlavná funkcia pre spustenie tréningu a testovania modelu.
    """

    # Inicializacia trialu
    trial = Trial(model=model, 
              optimizer=optimizer, 
              criterion=L, 
              metrics=['loss', 'accuracy'], 
            ).to(device)
    trial.with_generators(train_generator=train_loader, val_generator=val_loader, test_generator=test_loader)

    # Trenovanie a validacia
    print("Spúšťam tréning...")
    trial.run(epochs=EPOCHS)

    # Testovanie
    print("Spúšťam testovanie...")
    print(trial.evaluate(data_key=torchbearer.TEST_DATA))

    # Ulozenie modelu
    print("Ukladám model...")
    torch.save(model.state_dict(), 'DataNoise/models/model_1_state_dict.pth')

if __name__ == '__main__':
  main()
