import torch, torchvision, torchbearer
from torchbearer import Trial

from my_transform import transform_data
from data_loaders import get_data_loaders

torch.manual_seed(17)
BATCH_SIZE = 512
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

not_noisy_transform = transform_data(gaus=False, pois=False, snp=False)
noisy_transform = transform_data(gaus=True, pois=True, snp=True)

data = torchvision.datasets.OxfordIIITPet(root='../data', download=True, target_types="binary-category", transform=noisy_transform)
train_loader, val_loader, test_loader = get_data_loaders(data, train_size=0.8, val_size=0.1, test_size=0.1, batch_size=BATCH_SIZE)


model = torchvision.models.resnet18().to(device)
state_dict = torch.load('models/model_1_state_dict.pth')
model.load_state_dict(state_dict)
model.eval()  # prepnutie do eval módu


trial = Trial(model=model, metrics=['loss', 'accuracy']).to(device)
trial.with_generators(test_generator=test_loader)

# Testovanie
print("Spúšťam testovanie...")
print(trial.evaluate(data_key=torchbearer.TEST_DATA))
