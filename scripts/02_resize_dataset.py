import os
import shutil
 

# --- CONFIGURAZIONE ---
# Assicurati che i percorsi siano corretti rispetto a dove lanci lo script
SRC_FRAME_TRAIN = "../data/EGO-CH-OBJ-SEG/synthetic/train/frames"
SRC_LABEL_TRAIN = "../data/EGO-CH-OBJ-SEG/synthetic/train/labels"
SRC_FRAME_TEST = "../data/EGO-CH-OBJ-SEG/synthetic/test/frames"
SRC_LABEL_TEST = "../data/EGO-CH-OBJ-SEG/synthetic/test/labels"

DST_FRAME_TRAIN = "../data/resize_dt_synthetic/train/frame"
DST_LABEL_TRAIN = "../data/resize_dt_synthetic/train/label"
DST_FRAME_TEST = "../data/resize_dt_synthetic/test/frame"
DST_LABEL_TEST = "../data/resize_dt_synthetic/test/label"

def main():
    
    if not os.path.exists("../data/resize_dt_synthetic"):
        print("Creazione cartella '../data/resize_dt_synthetic'...")
        os.makedirs("../data/resize_dt_synthetic")

    # 1. Crea le cartelle di destinazione se non esistono
    os.makedirs(DST_FRAME_TRAIN, exist_ok=True)
    os.makedirs(DST_LABEL_TRAIN, exist_ok=True)
    os.makedirs(DST_FRAME_TEST, exist_ok=True)
    os.makedirs(DST_LABEL_TEST, exist_ok=True)

    # 2. Legge e ORDINA i file (fondamentale per mantenere l'allineamento)
    # Filtriamo per ignorare file nascosti o di sistema
    frames_train = sorted([f for f in os.listdir(SRC_FRAME_TRAIN) if not f.startswith('.')])
    labels_train = sorted([f for f in os.listdir(SRC_LABEL_TRAIN) if not f.startswith('.')])

    frames_test = sorted([f for f in os.listdir(SRC_FRAME_TEST) if not f.startswith('.')])
    labels_test = sorted([f for f in os.listdir(SRC_LABEL_TEST) if not f.startswith('.')])



    # Controllo di sicurezza
    if len(frames_train) != len(labels_train):
        print(f"ATTENZIONE: Trovati {len(frames_train)} frame e {len(labels_train)} label nel train set.")
        print("Assicurati che i file siano accoppiati correttamente prima di procedere.")
    
    if len(frames_test) != len(labels_test):
        print(f"ATTENZIONE: Trovati {len(frames_test)} frame e {len(labels_test)} label nel test set.")
        print("Assicurati che i file siano accoppiati correttamente prima di procedere.")

    # 3. Applica il slicing: Prende 1 elemento ogni 3
    # Sintassi [start:stop:step]
    selected_frames_train = frames_train[::3]
    selected_labels_train = labels_train[::3]

    print(f"Riduzione dataset train: Copio {len(selected_frames_train)} file su {len(frames_train)} originali...")

    # 4. Copia i file del train set
    for f_name, l_name in zip(selected_frames_train, selected_labels_train):
        # Percorsi completi sorgente
        src_f_path = os.path.join(SRC_FRAME_TRAIN, f_name)
        src_l_path = os.path.join(SRC_LABEL_TRAIN, l_name)
        
        # Percorsi completi destinazione
        dst_f_path = os.path.join(DST_FRAME_TRAIN, f_name)
        dst_l_path = os.path.join(DST_LABEL_TRAIN, l_name)

        # Copia effettiva
        shutil.copy2(src_f_path, dst_f_path)
        shutil.copy2(src_l_path, dst_l_path)

    # 5. Ripeti per il test set
    selected_frames_test = frames_test[::3]
    selected_labels_test = labels_test[::3]

    print(f"Riduzione dataset test: Copio {len(selected_frames_test)} file su {len(frames_test)} originali...")

    for f_name, l_name in zip(selected_frames_test, selected_labels_test):
        src_f_path = os.path.join(SRC_FRAME_TEST, f_name)
        src_l_path = os.path.join(SRC_LABEL_TEST, l_name)

        dst_f_path = os.path.join(DST_FRAME_TEST, f_name)
        dst_l_path = os.path.join(DST_LABEL_TEST, l_name)

        shutil.copy2(src_f_path, dst_f_path)
        shutil.copy2(src_l_path, dst_l_path)

    print("Fatto! Dataset ridotto creato in 'resize_dt'.")

if __name__ == "__main__":
    main()