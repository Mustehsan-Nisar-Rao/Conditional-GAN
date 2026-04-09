# Pix2Pix - Anime Sketch to Color Image Translation

A deep learning project that converts anime sketches into colored images using 
Pix2Pix (Conditional GAN) architecture.

---

## What This Project Does

```
Input  →  Black & White Anime Sketch
Output →  Fully Colored Anime Image
```

---

## Examples

| Input Sketch | Generated Color | Ground Truth |
|:---:|:---:|:---:|
| <img width="215" height="214" src="https://github.com/user-attachments/assets/fdecf9d2-6d84-4a78-8637-f5bd6515b063" /> | <img width="197" height="215" src="https://github.com/user-attachments/assets/76e89940-e6eb-43e8-ad45-6df0ecd64052" /> | <img width="198" height="213" src="https://github.com/user-attachments/assets/15689e65-57df-4e42-a959-51efd81aeeb7" /> |

## Project Structure

```
pix2pix-anime-colorization/
│
├── notebook/
│   └── pix2pix_anime.ipynb       ← main kaggle notebook
│
├── assets/
│   ├── sketch1.png               ← sample input images
│   ├── generated1.png            ← sample generated images
│   └── real1.png                 ← sample ground truth
│
├── checkpoints/
│   ├── best_generator.pth        ← best saved model
│   └── best_discriminator.pth
│
├── results/
│   ├── loss_plots.png            ← training loss graphs
│   └── results.png               ← visual results
│
└── README.md
```

---

## Model Architecture

### Generator - U-Net

```
Input Sketch (3, 256, 256)
        │
   ┌────▼────┐
   │ Encoder │  7 layers - image compress hoti hai
   │         │  256 → 128 → 64 → 32 → 16 → 8 → 4 → 2
   └────┬────┘
        │
   ┌────▼────┐
   │Bottleneck│  2 → 1
   └────┬────┘
        │
   ┌────▼────┐
   │ Decoder │  7 layers - image wapas banti hai
   │         │  skip connections encoder se aate hain
   └────┬────┘
        │
Output Color Image (3, 256, 256)
```

### Discriminator - PatchGAN (16×16)

```
Input: Sketch + Image concatenated (6, 256, 256)
        │
   Conv(stride=4) → 64  channels
   Conv(stride=2) → 128 channels
   Conv(stride=2) → 256 channels
   Conv(stride=2) → 1   channel
        │
Output: 16×16 patch probabilities
(each patch = real or fake)
```

---

## Dataset

**Anime Sketch Colorization Pair**  
Link: https://www.kaggle.com/datasets/ktaebum/anime-sketch-colorization-pair

```
Structure:
/data/
  /train/    ← training images
  /val/      ← validation images

Each image format:
|___________________|___________________|
|   Colored Image   |   B&W Sketch      |
|   (left half)     |   (right half)    |
|___________________|___________________|
```

---

## Training Details

| Setting | Value |
|---------|-------|
| Platform | Kaggle |
| GPU | T4 x2 (Dual GPU) |
| Image Size | 256 × 256 |
| Batch Size | 16 |
| Epochs | 50 |
| Optimizer | Adam |
| Learning Rate | 0.0002 |
| Beta 1 | 0.5 |
| Beta 2 | 0.999 |
| L1 Lambda | 100 |

---

## Loss Functions

```
Generator Loss  = Adversarial Loss + (100 × L1 Loss)
                           │                  │
                  fool discriminator    stay close to
                                        real image

Discriminator Loss = (Real Loss + Fake Loss) / 2
```

---

## Results

### Training Curves

![Loss Plot](results/loss_plots.png)

### Quantitative Evaluation

| Metric | Score |
|--------|-------|
| SSIM | 0.XXXX |
| PSNR | XX.XX dB |

> Replace XX values with your actual results after training

---

## How To Run

### On Kaggle

```
1. Open Kaggle Notebook
2. Add dataset: anime-sketch-colorization-pair
3. Enable GPU T4 x2
4. Run cells one by one (Cell 1 to Cell 19)
```

### Locally

```bash
# Clone repo
git clone https://github.com/yourusername/pix2pix-anime-colorization.git
cd pix2pix-anime-colorization

# Install dependencies
pip install -r requirements.txt

# Run notebook
jupyter notebook notebook/pix2pix_anime.ipynb
```

---

## Requirements

```
torch
torchvision
Pillow
numpy
matplotlib
tqdm
scikit-image
gradio
```

---

## Gradio App

A simple web app to test the model:

```
1. Upload an anime sketch
2. Model generates colored version
3. Result displayed instantly
```

![App Demo](assets/app_demo.png)

---

## Model Checkpoints

| File | Description |
|------|-------------|
| best_generator.pth | Best model based on validation loss |
| best_discriminator.pth | Best discriminator |
| gen_epoch_10.pth | Checkpoint at epoch 10 |
| gen_epoch_20.pth | Checkpoint at epoch 20 |
| gen_epoch_30.pth | Checkpoint at epoch 30 |
| gen_epoch_40.pth | Checkpoint at epoch 40 |
| gen_epoch_50.pth | Checkpoint at epoch 50 |

---

## Project Highlights

- U-Net Generator with skip connections
- PatchGAN Discriminator (16×16 patches)
- Mixed Precision Training (faster on GPU)
- Dual GPU support
- Best model auto-save on validation loss
- Checkpoint every 5 epochs
- Gradio web app for demo

---

## References

- [Pix2Pix Paper](https://arxiv.org/abs/1611.07004) - Image-to-Image Translation with Conditional Adversarial Networks
- [U-Net Paper](https://arxiv.org/abs/1505.04597) - Convolutional Networks for Biomedical Image Segmentation
- [Dataset](https://www.kaggle.com/datasets/ktaebum/anime-sketch-colorization-pair) - Anime Sketch Colorization Pair

---

## Author

**Mustehsan Nisar Rao**  
BS Computer Science  
