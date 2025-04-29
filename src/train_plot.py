# === src/train_plot.py ===

import matplotlib.pyplot as plt
import pandas as pd
import os

# Funkcia na načítanie CSV výsledkov a vykreslenie grafov
def plot_training_curves(csv_path):
    if not os.path.exists(csv_path):
        print(f"❌ CSV súbor {csv_path} neexistuje.")
        return

    data = pd.read_csv(csv_path)

    epochs = range(1, len(data) + 1)

    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    # Loss
    axs[0].plot(epochs, data['train_loss'], label='Training Loss')
    axs[0].plot(epochs, data['val_loss'], label='Validation Loss')
    axs[0].set_title('Loss over Epochs')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].legend()

    # Accuracy
    axs[1].plot(epochs, data['train_acc'], label='Training Accuracy')
    axs[1].plot(epochs, data['val_acc'], label='Validation Accuracy')
    axs[1].set_title('Accuracy over Epochs')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Accuracy')
    axs[1].legend()

    plt.tight_layout()
    plt.savefig('results/training_curves.png')
    plt.show()
    print("✅ Grafy uložené do 'results/training_curves.png'")

if __name__ == "__main__":
    plot_training_curves('results/training_log.csv')