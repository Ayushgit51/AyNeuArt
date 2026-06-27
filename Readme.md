# AyNeuArt 🎨

**AyNeuArt** is a neural style transfer web app built on **AdaIN (Adaptive Instance Normalization)**. Upload a content image and a style image, and the app blends them together — repainting your photo in the texture, color, and brushwork of the style reference, in real time.

🔗 **Live demo:** [ayush51/AyNeuArt on Hugging Face Spaces](https://huggingface.co/spaces/ayush51/AyNeuArt)

---

## ✨ Features

- **AdaIN-based style transfer** — fast, arbitrary style transfer using a VGG encoder/decoder architecture with adaptive instance normalization, instead of slow per-image optimization.
- **Web interface** — simple Flask app where users upload a content image and a style image and get a stylized result.
- **Pretrained VGG encoder** (`vgg_normalised.pth`) for feature extraction.
- **Training script** (`train.py`) to retrain or fine-tune the decoder on your own dataset.
- **Deployment-ready** — includes a `Procfile` and `requirements.txt` for easy deployment to platforms like Heroku or Hugging Face Spaces.

---

## 🧠 How It Works

Neural style transfer with AdaIN works in three steps:

1. **Encode** — both the content image and style image are passed through a pretrained VGG encoder to extract feature maps.
2. **Align statistics** — the content features' mean and variance are aligned to match the style features' mean and variance (the "Adaptive Instance Normalization" step).
3. **Decode** — the normalized features are passed through a trained decoder network to reconstruct the final stylized image.

This approach (Huang & Belongie, 2017) enables arbitrary style transfer in a single forward pass, much faster than classic optimization-based methods (e.g. Gatys et al.).

---

## 📁 Project Structure

```
AyNeuArt/
├── app.py                  # Flask web app — handles uploads and stylization requests
├── train.py                # Script to train/fine-tune the AdaIN decoder
├── vgg_normalised.pth       # Pretrained VGG weights used for feature extraction
├── requirements.txt         # Python dependencies
├── procfile.txt             # Process file for deployment (e.g. Heroku/HF Spaces)
├── templates/               # HTML templates for the web UI
├── static/uploads/          # Uploaded content/style images and generated outputs
└── experiment/final_exp/    # Experimental runs / training artifacts
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip

### Installation

```bash
git clone https://github.com/Ayushgit51/AyNeuArt.git
cd AyNeuArt
pip install -r requirements.txt
```

### Running locally

```bash
python app.py
```

Then open your browser at `http://localhost:5000` (or the port shown in your terminal) and upload a content image and a style image to generate your stylized result.

### Training your own decoder

If you want to retrain the decoder network on a custom dataset of content/style images:

```bash
python train.py
```

> Make sure to update dataset paths and hyperparameters inside `train.py` before training.

---

## 🛠️ Tech Stack

- **Python** — core logic and model code
- **PyTorch** — AdaIN encoder/decoder implementation
- **Flask** — web server and routing
- **HTML/CSS** — frontend templates
- **Hugging Face Spaces** — deployment target

---

## 📸 Example

| Content Image | Style Image | Output |
|---|---|---|
| *your photo* | *artwork/painting* | *stylized result* |

*(Add sample images here once available — drop them in a `samples/` folder and reference them in this table.)*

---

## Acknowledgements

- Huang, X., & Belongie, S. (2017). *Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization.* [ICCV 2017]
- Pretrained VGG weights adapted from the original AdaIN implementation.

---

## 👤 Author

**Ayush** — AI/ML Engineer | Building practical, story-driven AI projects
- GitHub: [@Ayushgit51](https://github.com/Ayushgit51)
