# Modul na definovanie transformácií pre obrázky
import torch, torchvision
import torchvision.transforms.v2 as transforms


def add_gaussian_noise(tensor: torch.Tensor, mean: float|int=0.0, std: float|int=0.1) -> torch.Tensor:
    """
    Pridá Gaussov šum do vstupného tensoru (obrázku).
    
    Vstupy:
        tensor - vstupný tensor (obrázok) s hodnotami v rozsahu [0, 1]
        mean - stredná hodnota Gaussovho rozdelenia 
                - Určuje, okolo akej hodnoty sa budú sústreďovať náhodné odchýlky, ktoré pridávame do obrázku. 
                Ak je hodnota 0, znamená to, že šum nemá systematický posun hore alebo dole.
        std (σ) - smerodajná odchýlka Gaussovho rozdelenia 
                - určuje, ako rozptýlené budú hodnoty šumu okolo strednej hodnoty (mean). 
                Vyššia hodnota std znamená, že do obrázka pridávame výraznejší a "širší" šum, 
                zatiaľ čo nižšia hodnota spôsobí jemnejší šum. 

    Výstup:
        Tensor (obrázok) s pridaným Gaussovým šumom.
    """
    noise = torch.randn(tensor.size()) * std + mean
    noisy_tensor = tensor + noise
    return noisy_tensor


def add_poisson_noise(tensor: torch.Tensor, lam: float|int=30) -> torch.Tensor:
    """
    Pridá Poissonov šum do vstupného tensoru (obrázku). Na simuláciu Poissonovho šumu v rozsahu [0, 1] najprv
    tensor vynásobíme lam, aby sme získali "počet" fotónov, potom použijeme torch.poisson() a výsledok normalizujeme späť.
    
    Vstupy:
      tensor - vstupný tensor (obrázok) s hodnotami v rozsahu [0, 1]
      lam (lambda λ) - parameter pre škálovanie; vyššia hodnota znamená menej výrazný šum, nižšia viac výrazný šum
            - Najprv sa vstupný obrázok, ktorý má hodnoty v rozsahu [0, 1], vynásobí hodnotou lambda, čím sa simuluje počet fotónov, 
            ktorý by mohol byť zaznamenaný. Potom sa na tieto "počty" aplikuje Poissonovo rozdelenie, ktoré vygeneruje náhodné hodnoty, 
            a následne sa výsledok normalizuje späť delením lambdou. Vyššia hodnota lambdy znamená, že simulovaný počet fotónov je väčší, 
            čo vedie k menej výraznému šumu.

    Výstup:
      Tensor (obrázok) s pridaným Poissonovým šumom.
    """

    noisy_tensor = torch.poisson(tensor * lam) / lam
    return noisy_tensor


def add_salt_and_pepper_noise(tensor: torch.Tensor, salt_prob: float|int=0.01, pepper_prob: float|int=0.01) -> torch.Tensor:
    """
    Pridá salt-and-pepper šum do vstupného tensoru (obrázku). 
    Salt-and-pepper šum náhodne nastaví niektoré pixely na maximálnu hodnotu (soľ) 
    alebo minimálnu hodnotu (peper) v obrázku. Táto funkcia modifikuje vstupný tensor 
    aplikovaním šumu nezávisle na každý kanál.

    Vstupy:
        tensor - Vstupný tensor (obrázok). 
        salt_prob - Pravdepodobnosť, že pixel bude nastavený na maximálnu hodnotu (soľ). Predvolená hodnota je 0.01.
        pepper_prob - Pravdepodobnosť, že pixel bude nastavený na minimálnu hodnotu (korenie). Predvolená hodnota je 0.01.
    Výstup:
        Tensor (obrázok) s pridaným salt-and-pepper šumom.
    """

    noisy_tensor = tensor.clone()
    num_channels, height, width = noisy_tensor.shape
    # Vytvorenie masky pre soľ
    salt_mask = torch.rand((height, width)) < salt_prob
    # Vytvorenie masky pre peper
    pepper_mask = torch.rand((height, width)) < pepper_prob
    
    for color_channel in range(num_channels):
        noisy_tensor[color_channel][salt_mask] = 1.0
        noisy_tensor[color_channel][pepper_mask] = 0.0
    return noisy_tensor


def transform_data(gaus: bool=False, pois: bool=False, snp: bool=False) -> transforms.Compose:
    """
    Funkcia na definovanie transformácií pre obrázky - ako sa majú obrázky spracovať pred zadaním do modelu.

    Vstupy:
        gaus - či pridať Gaussov šum
        pois - či pridať Poissonov šum
        snp - či pridať salt-and-pepper šum
    Výstup:
        transforms.Compose - zložené transformácie pre obrázky
    """
    
    my_transform = transforms.Compose([ 
        # Nejaké predspracovanie obrázkov

        torchvision.transforms.Resize((224, 224)), # kvoli resnet18
        
                #torchvision.transforms.RandomHorizontalFlip(),
                #transforms.Grayscale(num_output_channels=3), # konverzia sedotonoveho obrazka na RGB kvoli resnet18
        
        torchvision.transforms.ToTensor(),
        
                #torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], # nejake RGB normalizovanie
                #                                    std=[0.229, 0.224, 0.225]), 
                #torchvision.transforms.Normalize(mean=[0.1307], std=[0.3081]), # nejake sedotonove normalizovanie
        
        # Pridanie rôznych typov a intenzít šumu
        transforms.Lambda(lambda x: add_gaussian_noise(x, mean=0.0, std=0.1)) if gaus else transforms.Lambda(lambda x: x),
        transforms.Lambda(lambda x: add_poisson_noise(x, lam=30)) if pois else transforms.Lambda(lambda x: x),
        transforms.Lambda(lambda x: add_salt_and_pepper_noise(x, salt_prob=0.01, pepper_prob=0.01)) if snp else transforms.Lambda(lambda x: x),
    ])

    return my_transform

