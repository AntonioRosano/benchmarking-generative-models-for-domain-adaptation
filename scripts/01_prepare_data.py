import os
import shutil


ROOT_DIR = os.path.abspath("./")

PATH_REAL_TRAIN = os.path.join(ROOT_DIR, "data/EGO-CH-OBJ-SEG/real/train/frames")
PATH_REAL_TEST  = os.path.join(ROOT_DIR, "data/EGO-CH-OBJ-SEG/real/test/frames")

PATH_SYN_TRAIN  = os.path.join(ROOT_DIR, "data/EGO-CH-OBJ-SEG/synthetic/train/frames")
PATH_SYN_TEST   = os.path.join(ROOT_DIR, "data/EGO-CH-OBJ-SEG/synthetic/test/frames")

# modelli per cui prepariamo i dati
MODELS_TO_PREPARE = ["cut", "munit", "unit"]

# nome del dataset virtuale che creiamo dentro ogni modello
DATASET_NAME = "ego_adaptation"

def prepare_dataset_folders(model_name):
    """
    Crea la struttura di cartelle e i symlink per un dato modello.
    Strategia: REAL -> SYNTHETIC
    Domain A = Real (Input)
    Domain B = Synthetic (Output)
    """
    print(f"\n[INFO] Configurazione dati per: {model_name.upper()}")

    # dove il modello cercherà i dati
    # Esempio: models/cut/datasets/ego_adaptation
    dest_path = os.path.join(ROOT_DIR, "models", model_name, "datasets", DATASET_NAME)

    if os.path.exists(dest_path):
        print(f"  -> Rimuovo vecchia configurazione in: {dest_path}")
        shutil.rmtree(dest_path)
    
    os.makedirs(dest_path)
    
    try:
        # Training Set
        os.symlink(PATH_REAL_TRAIN, os.path.join(dest_path, "trainA"))
        print(f"  -> Link creato: trainA punta a REAL (Train)")
        
        os.symlink(PATH_SYN_TRAIN, os.path.join(dest_path, "trainB"))
        print(f"  -> Link creato: trainB punta a SYNTHETIC (Train)")

        # Test Set
        os.symlink(PATH_REAL_TEST, os.path.join(dest_path, "testA"))
        print(f"  -> Link creato: testA  punta a REAL (Test)")

        os.symlink(PATH_SYN_TEST, os.path.join(dest_path, "testB"))
        print(f"  -> Link creato: testB  punta a SYNTHETIC (Test)")
        
    except OSError as e:
        print(f"[ERRORE] Creazione link fallita: {e}")


if __name__ == "__main__":
    print("=== INIZIO PREPARAZIONE DATI (Strategia: Real -> Synthetic) ===")
    
    if not os.path.exists(PATH_REAL_TRAIN):
        print(f"[ERRORE CRITICO] Non trovo i dati in: {PATH_REAL_TRAIN}")
        exit(1)

    for model in MODELS_TO_PREPARE:
        prepare_dataset_folders(model)
        
    print("\n=== COMPLETATO. I modelli sono pronti per il training. ===")