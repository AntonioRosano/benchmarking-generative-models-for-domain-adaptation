import matplotlib.pyplot as plt
import re
import os
from PIL import Image
import random

# --- training_loss_plot.png ---
log_path = "../results/CUT/experiment_logs/loss_log.txt"
output_dir = "../results/CUT/assets"
output_filename = "training_loss_plot.png"

os.makedirs(output_dir, exist_ok=True)

# Liste per i dati
epochs = []
G_GAN = []
NCE = []

print(f"Leggo il file {log_path}...")

try:
    with open(log_path, 'r') as f:
        for line in f:
            if "epoch:" in line and "G_GAN:" in line:
                ep = int(re.search(r'epoch:\s*(\d+)', line).group(1))
                
                if len(epochs) == 0 or ep != epochs[-1]:
                    epochs.append(ep)
                    G_GAN.append(float(re.search(r'G_GAN:\s*([\d\.]+)', line).group(1)))
                    NCE.append(float(re.search(r'NCE:\s*([\d\.]+)', line).group(1)))

    # --- CREAZIONE GRAFICO ---
    plt.figure(figsize=(12, 6))

    # Pannello 1: GAN Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, G_GAN, label='G_GAN (Generator)', color='#1f77b4', linewidth=2)
    plt.title("Stabilità del Training (Adversarial Loss)", fontsize=12)
    plt.xlabel("Epoche")
    plt.ylabel("Loss")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()

    # Pannello 2: NCE Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, NCE, label='NCE (Content)', color='#ff7f0e', linewidth=2)
    plt.title("Preservazione Contenuto (PatchNCE)", fontsize=12)
    plt.xlabel("Epoche")
    plt.ylabel("Loss")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()

    # Salvataggio
    save_path = os.path.join(output_dir, output_filename)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    print(f"Grafico salvato in: {save_path}")

except FileNotFoundError:
    print(f"Errore: Non trovo '{log_path}'.")






# --- CONFIGURAZIONE PERCORSI ---
# Adatta questi percorsi a dove si trovano i risultati del test su Kaggle
# Esempio tipico dopo aver runnato test.py:
results_dir = "./models/cut/results/ego_cut_kaggle/test_40/images"
path_real = os.path.join(results_dir, "real_A")
path_fake = os.path.join(results_dir, "fake_B")

output_dir = "assets"
output_filename = "comparison_grid.jpg"
num_samples = 4  # Quante righe vuoi nel grafico

# Crea cartella output
os.makedirs(output_dir, exist_ok=True)

try:
    # Prendi lista immagini
    files = os.listdir(path_real)
    # Filtra solo immagini
    files = [f for f in files if f.endswith(('.png', '.jpg'))]
    
    # Seleziona N immagini casuali
    selected_files = random.sample(files, min(num_samples, len(files)))
    
    # Crea la figura
    fig, axes = plt.subplots(num_samples, 2, figsize=(10, num_samples * 4))
    plt.subplots_adjust(wspace=0.1, hspace=0.1)

    print(f"🎨 Generazione griglia con {len(selected_files)} esempi...")

    for i, filename in enumerate(selected_files):
        # Carica immagini
        img_real = Image.open(os.path.join(path_real, filename))
        img_fake = Image.open(os.path.join(path_fake, filename))
        
        # Colonna 1: Reale
        axes[i, 0].imshow(img_real)
        if i == 0: axes[i, 0].set_title("Input Reale (Palazzo Bellomo)", fontsize=14)
        axes[i, 0].axis('off')
        
        # Colonna 2: Sintetico
        axes[i, 1].imshow(img_fake)
        if i == 0: axes[i, 1].set_title("Output Generato (Domain Adaptation)", fontsize=14)
        axes[i, 1].axis('off')

    # Salvataggio
    save_path = os.path.join(output_dir, output_filename)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Griglia salvata in: {save_path}")

except FileNotFoundError:
    print("❌ Errore: Non trovo le cartelle dei risultati.")
    print(f"   Ho cercato in: {results_dir}")
    print("   Hai eseguito 'test.py' prima di lanciare questo script?")