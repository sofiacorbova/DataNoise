from torch.utils.data import random_split
from torch.utils.data import DataLoader

def get_data_loaders(data, train_size: float=0.8, val_size: float=0.1, test_size: float=0.1, batch_size: int=32) -> DataLoader:

    """
    Funkcia na rozdelenie datasetu na trenovacie, validacne a testovacie data loadery.
    """

    train_size = int(train_size * len(data)) # 80% na trenovanie
    val_size = int(val_size * len(data)) # 10% na validaciu
    test_size =  len(data) - (train_size + val_size) # 10% na testovanie

    train_dataset, val_dataset, test_dataset = random_split(data, [train_size, val_size, test_size]) # rozdelenie dat

    # vytvorenie torch.utils.data.DataLoader objektu pre kazdu cast dat
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
