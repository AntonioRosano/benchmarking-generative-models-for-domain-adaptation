# benchmarking-generative-models-for-domain-adaptation

This repository contains the code and experiments developed as part of a
**Deep Learning course project**, aimed at replicating and extending an
existing semantic segmentation pipeline for artworks in cultural heritage
environments.

The project focuses on the experimental analysis of generative models
for **synthetic-to-real domain adaptation**.

The work is based on the paper:

> *Semantic Object Segmentation in Cultural Sites using Real and Synthetic Data*  
> Francesco Ragusa, Daniele Di Mauro, Alfio Palermo, Antonino Furnari, Giovanni Maria Farinella

In the original work, domain adaptation between synthetic and real images
is performed using CycleGAN.  
**In this project, we keep the original pipeline unchanged and replace
CycleGAN with alternative generative models**, in order to evaluate their
impact on segmentation performance.

---

## Original Pipeline

The original architecture consists of:
- **PSPNet** for pixel-level semantic segmentation
- **Synthetic images** generated from a 3D model of the cultural site
- **CycleGAN** for synthetic-to-real domain adaptation

---

## Our Contribution

As part of a Deep Learning course project, our contribution consists of:
- Replicating the original experimental setup
- Replacing **CycleGAN** with alternative, state-of-the-art generative
  models for unpaired image-to-image translation:
  - CUT
  - UNIT
  - MUNIT
- Benchmarking their effectiveness using the same dataset, metrics and
  training protocol as the original paper

This allows a fair and controlled comparison between different
domain adaptation strategies.

---

## Dataset

We use the **EGO-CH-OBJ-SEG** dataset, which includes:
- Synthetic images generated from a 3D model (Blender)
- Real egocentric images acquired in a cultural heritage site
- Pixel-level semantic segmentation masks for 24 artworks

Dataset link:  
https://iplab.dmi.unict.it/EGO-CH-OBJ-SEG/

---

## Experimental Setup

- Segmentation model: PSPNet
- Domain adaptation: CycleGAN (baseline), CUT, UNIT, MUNIT
- Evaluation metrics:
  - Pixel Accuracy
  - Class Accuracy
  - Mean Intersection over Union (mIoU)
  - Frequency Weighted IoU

---

## Academic Context

This project was developed as part of a university-level course in
**Deep Learning**, with the goal of gaining hands-on experience in:
- reproducing results from the literature
- modifying and extending existing architectures
- performing fair experimental benchmarking of deep learning models
