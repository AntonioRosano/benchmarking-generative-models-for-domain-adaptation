import os
import shutil

"""
Questo script prepara la struttura di cartelle e i symlink necessari
per addestrare i modelli di domain adaptation (CUT, MUNIT, UNIT)
utilizzando il dataset EGO-CH-OBJ-SEG. La strategia adottata è:
Domain A = Real (Input)
Domain B = Synthetic (Output)

Inoltre, corregge AUTOMATICAMENTE il bug noto nel codice di CUT (cut_model.py).
"""

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


def fix_cut_options_bug():
    """
    Corregge automaticamente il bug in models/cut/models/cut_model.py
    dove choices è una stringa con parentesi tonde invece di una lista.
    """
    print("\n[INFO] Controllo presenza bug in CUT model...")
    
    # 1. Puntiamo al file GIUSTO (quello che dava errore nel traceback)
    model_file = os.path.join(ROOT_DIR, "models/cut/models/cut_model.py")
    
    if not os.path.exists(model_file):
        print(f"[WARN] Non trovo il file: {model_file}")
        return

    with open(model_file, "r") as f:
        content = f.read()
    
    # 2. Cerchiamo la stringa ESATTA che causa il crash (quella con le parentesi tonde)
    # Nota: copiata dal tuo traceback di errore
    buggy_string = "choices='(CUT, cut, FastCUT, fastcut)'"
    
    # 3. La sostituiamo con la lista corretta (parentesi quadre, niente virgolette esterne)
    fixed_string = "choices=['CUT', 'cut', 'FastCUT', 'fastcut']"
    
    if buggy_string in content:
        print("  -> Bug trovato in cut_model.py! Applicazione della patch correttiva...")
        new_content = content.replace(buggy_string, fixed_string)
        
        with open(model_file, "w") as f:
            f.write(new_content)
        print("  -> File riparato con successo.")
    
    # Controllo extra: a volte le virgolette sono diverse nel codice originale
    elif 'choices="(CUT, cut, FastCUT, fastcut)"' in content:
        print("  -> Bug trovato (versione doppi apici)! Applicazione patch...")
        new_content = content.replace('choices="(CUT, cut, FastCUT, fastcut)"', fixed_string)
        with open(model_file, "w") as f:
            f.write(new_content)
        print("  -> File riparato con successo.")
        
    else:
        print("  -> Il file cut_model.py sembra già corretto.")

if __name__ == "__main__":
    print("=== INIZIO PREPARAZIONE DATI (Strategia: Real -> Synthetic) ===")
    
    if not os.path.exists(PATH_REAL_TRAIN):
        print(f"[ERRORE CRITICO] Non trovo i dati in: {PATH_REAL_TRAIN}")
        exit(1)

    for model in MODELS_TO_PREPARE:
        prepare_dataset_folders(model)

    fix_cut_options_bug()
        
    print("\n=== COMPLETATO. I modelli sono pronti per il training. ===")