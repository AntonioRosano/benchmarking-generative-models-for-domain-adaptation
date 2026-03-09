# Benchmarking Generative Models for Domain Adaptation

Questo repository contiene il codice, i notebook e la documentazione del progetto accademico di Deep Learning (A.A. 2025-2026) realizzato da **Marco Gionfriddo, Antonio Rosano e Angelo Spadola**.

Il progetto esplora le tecniche di **Unsupervised Domain Adaptation (UDA)** per la segmentazione semantica di opere d'arte in siti culturali, affrontando il problema del Domain Shift tra immagini sintetiche (generate da modelli 3D) e fotografie reali.


## Obiettivo del Progetto
L'addestramento di modelli di segmentazione semantica richiede enormi quantità di dati annotati manualmente, un processo costoso e soggetto a errori. L'utilizzo di dati sintetici (con maschere generate automaticamente) è un'alternativa valida, ma soffre del divario di dominio rispetto alle immagini reali.

L'obiettivo di questo studio è valutare se e come i moderni modelli generativi di Image-to-Image Translation possano colmare questo divario, traducendo le immagini reali nello stile di quelle sintetiche prima di sottoporle a una rete di segmentazione pre-addestrata.

## Architettura e Pipeline
La nostra pipeline si divide in due stadi principali:

1. **Image-to-Image Translation:** Abbiamo testato e confrontato tre diverse architetture generative per tradurre le foto reali nel dominio sintetico:
   * **[CUT]** (Contrastive Unpaired Translation)
   * **[UNIT]** (UNsupervised Image-to-image Translation)
   * **[MUNIT]** (Multimodal UNsupervised Image-to-image Translation)

2. **Segmentazione Semantica:** Abbiamo utilizzato **PSPNet** (Pyramid Scene Parsing Network). La rete è stata:
   * Pre-addestrata esclusivamente sul dataset sintetico.
   * Sottoposta a *fine-tuning* sulle immagini reali tradotte dai modelli generativi del primo step.

## Dataset
Il progetto si basa sul dataset **EGO-CH**, che comprende frame estratti da video in prima persona all'interno della Galleria Regionale di Palazzo Bellomo (Siracusa) e la rispettiva controparte sintetica (generata tramite Unity).

## Utilizzo

```bash
git clone https://github.com/TuoUsername/Benchmarking-Generative-Models.git
cd Benchmarking-Generative-Models
```
