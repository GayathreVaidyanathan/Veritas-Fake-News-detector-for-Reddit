# VERITAS — Visual Misinformation Detection Terminal

![Python](https://img.shields.io/badge/Python-3.11-blue) ![CLIP](https://img.shields.io/badge/CLIP-ViT--B%2F32-purple) ![Streamlit](https://img.shields.io/badge/Streamlit-1.32-red) ![Accuracy](https://img.shields.io/badge/Accuracy-88.7%25-green) ![ROC--AUC](https://img.shields.io/badge/ROC--AUC-0.956-brightgreen)

A multimodal AI system that detects visual misinformation by analyzing the semantic alignment between news images and headlines using OpenAI CLIP embeddings and a trained MLP classifier.

---

## Demo

> Upload a news image + paste a headline → VERITAS returns a verdict: **AUTHENTIC** or **MANIPULATED**

![Veritas Demo](assets/demo.png)

---

## How It Works
News Image + Headline
↓
CLIP ViT-B/32 Encoder
↓
Image Embedding (512d) + Text Embedding (512d) + Cosine Similarity (1d)
↓
Feature Vector (1025d) → StandardScaler → MLP Classifier
↓
AUTHENTIC / MANIPULATED + Confidence Score
CLIP encodes both the image and headline into a shared 512-dimensional embedding space. The cosine similarity between these embeddings serves as an image-text alignment signal. A lightweight MLP classifier trained on top of these features learns to distinguish real from manipulated news posts.

---

## Dataset

**Fakeddit** — A large-scale multimodal fake news detection dataset sourced from Reddit.

| Split | Samples | Real | Fake |
|-------|---------|------|------|
| Train | 34,147 | 24,170 | 9,977 |
| Val | 50,877 | 32,460 | 18,417 |
| Test | 56,458 | 34,530 | 21,928 |

- Source: [Fakeddit on Kaggle](https://www.kaggle.com/datasets/vanshikavmittal/fakeddit-dataset)
- Labels: 2-way classification (0 = Real, 1 = Fake)
- Images downloaded on-the-fly from Reddit CDN during embedding extraction (~31% URL failure rate due to expired links)

---

## Model Architecture

| Component | Details |
|-----------|---------|
| Visual Encoder | CLIP ViT-B/32 (frozen) |
| Text Encoder | CLIP ViT-B/32 (frozen) |
| Feature Vector | 512 (img) + 512 (txt) + 1 (cosine sim) = 1025 dims |
| Classifier | MLP (512 → 256 → 128) with early stopping |
| Scaler | StandardScaler |

CLIP weights are frozen — only the MLP classifier head is trained.

---

## Results

| Metric | Validation | Test |
|--------|-----------|------|
| Accuracy | 89.6% | 88.7% |
| F1 Score | 85.2% | 84.8% |
| ROC-AUC | 96.2% | 95.6% |

### Per-class Performance (Test Set)

| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| Real | 0.89 | 0.93 | 0.91 |
| Fake | 0.89 | 0.81 | 0.85 |

---

## Limitations & Observations

Qualitative testing revealed an important finding: **politically charged imagery** (e.g. world leaders, diplomatic meetings) consistently scores as MANIPULATED regardless of caption accuracy. This reflects dataset-level bias in the Fakeddit training distribution, where Reddit community labeling patterns associate certain political imagery with misinformation at higher rates.

This highlights a known challenge in multimodal misinformation detection: **models trained on social media data inherit the labeling biases of the platform**, and generalization beyond the training distribution remains an open research problem.

---

## Project Structure
visual-misinformation-detector/
├── app.py                  # Streamlit application
├── retrain.py              # Local retraining script
├── classifier.pkl          # Trained MLP classifier
├── scaler.pkl              # Feature scaler
├── config.json             # Model configuration
├── train.npz               # Train embeddings
├── val.npz                 # Val embeddings
├── test.npz                # Test embeddings
├── requirements.txt        # Dependencies
├── .streamlit/
│   └── config.toml         # Streamlit theme
└── assets/
└── demo.png            # Demo screenshot
---

## Installation

```bash
# Clone the repo
git clone https://github.com/GayathreVaidyanathan/visual-misinformation-detector.git
cd visual-misinformation-detector

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux

# Install dependencies
pip install -r requirements.txt
pip install git+https://github.com/openai/CLIP.git
```

---

## Usage

```bash
streamlit run app.py
```

1. Enter a news headline in the text box
2. Provide an image via URL or file upload
3. Click **RUN ANALYSIS**
4. View the verdict, confidence score, and image-text alignment

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Multimodal Encoder | OpenAI CLIP ViT-B/32 |
| Classifier | Scikit-learn MLPClassifier |
| Feature Engineering | NumPy, StandardScaler |
| Frontend | Streamlit + Custom CSS |
| Image Processing | Pillow, Requests |
| Experiment Tracking | Kaggle Notebooks (T4 x2 GPU) |

---

## Kaggle Notebook

The full training pipeline (embedding extraction + classifier training) is available as a Kaggle notebook:

🔗 *(add your Kaggle notebook link here)*

---

## Author

**Gayathre Vaidyanathan**  
Final Year Integrated M.Sc. Data Science  
Amrita Vishwa Vidyapeetham, Coimbatore

🔗 [GitHub](https://github.com/GayathreVaidyanathan) · [LinkedIn](https://www.linkedin.com/in/gayathre-vaidyanathan-567b4a30a/) *(add your LinkedIn)*