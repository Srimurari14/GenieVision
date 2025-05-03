# GenieVision - Text-to-Image Generation with Latent Diffusion Models

GenieVision is a text-to-image generation project developed as part of the CSCI 6366 (Neural Networks & Deep Learning) course at George Washington University. It leverages **Latent Diffusion Models (LDMs)** to efficiently generate high-quality, photorealistic images from textual descriptions.

This notebook-based project trains a diffusion model in **latent space** using **CLIP embeddings** and a **VAE encoder/decoder**, significantly reducing compute cost while maintaining image fidelity.

---

## Project Highlights

- Implements a **custom Latent Diffusion Model pipeline**.
- Compatible with **Conceptual Captions** and **COCO** datasets.
- Uses **CLIP-ViT-L/14** for text embeddings and **Stable Diffusion VAE** for latent conversion.
- Supports **LoRA** fine-tuning for efficient adaptation.
- Evaluates performance using **CLIP score**.
- Efficient training: starts with low resolution (128x128 or 256x256).

---

## Getting Started

> **Note:** This project is designed to be run exclusively in a Jupyter Notebook environment (`.ipynb`). Ensure you are using a GPU runtime for best results.

### Pre-requisites

Install these packages before running the notebook:

```bash
pip install torch torchvision
pip install diffusers transformers accelerate
pip install datasets ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git
```

---

## Datasets

You can choose between:

### [COCO 2017](https://cocodataset.org/#download)
- Download **train2017.zip** and **captions_train2017.json**.
- Organize as:
  ```
  coco_dataset/
  ├── images/
  │   └── train2017/
  └── annotations/
      └── captions_train2017.json
  ```

---

## How to Run

1. **Clone this repository** and open `GenieVision.ipynb` in a Jupyter environment like **Google Colab** or **JupyterLab**.
2. **Ensure GPU is enabled.**
   - Recommended GPU: `Tesla T4`, `A100`, or `V100` (tested).
3. **Download and unzip the dataset** (COCO).
4. **Run the notebook cells in order**, beginning with installation, dataset preparation, and then training/inference.

---

## Evaluation

- **CLIP Score** is used to measure image-text alignment.
- Planned metrics: FID, Inception Score (optional).
- Early stopping and image previews included to monitor training quality.

---

## Structure

| Section | Description |
|--------|-------------|
| `Data Loader` | Preprocess images and captions |
| `Embedder` | Generates CLIP text embeddings |
| `VAE` | Converts images to/from latent space |
| `UNet` | Denoising model used in diffusion |
| `Trainer` | Training loop, optimizer, scheduler |
| `Inference` | Generates images from prompts |

---

## Experimental Files  

Some notebooks in this repository contain raw experimentation and debugging logs from development. These are **not production-ready** but may help others understand our trial-and-error process.  

---

## Code Style  

This repository follows the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html) for naming conventions, docstrings, and formatting wherever possible. Jupyter notebooks follow PEP8 standards within the markdown and code cells.  

---  

## Authors

- Sri Murari Dachepalli    
- Sunkara Rohith  
- Sneha Uppu  
  
Developed as part of Spring 2025 course: **CSCI 6366 Neural Networks and Deep Learning**. 


