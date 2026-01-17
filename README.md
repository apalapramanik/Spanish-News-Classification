# Spanish News Classification

A multi-class text classification project for Spanish news articles using BERT-based deep learning.

## Overview

This project fine-tunes a pre-trained Spanish BERT model ([BETO](https://github.com/dccuchile/beto)) to classify Spanish-language news articles into 7 categories:

- **Macroeconomia** - Macroeconomic news
- **Alianzas** - Business alliances and partnerships
- **Innovacion** - Innovation and technology
- **Regulaciones** - Regulations and policy
- **Sostenibilidad** - Sustainability
- **Reputacion** - Reputation and brand news
- **Otra** - Other/miscellaneous

## Dataset

- **Source:** Spanish business/financial news articles (La República)
- **Size:** 4,570 articles
- **Format:** CSV with columns: `url`, `news`, `Type`

## Model Architecture

```
Input (Spanish text, max 512 tokens)
        ↓
Spanish BERT (dccuchile/bert-base-spanish-wwm-cased)
        ↓
CLS Token Extraction
        ↓
Dense Layer (7 units, softmax)
        ↓
Output (7 class probabilities)
```

## Requirements

```bash
pip install numpy pandas scikit-learn transformers tensorflow matplotlib
```

**Dependencies:**
- Python 3.7+
- NumPy
- Pandas
- Scikit-learn
- Transformers (HuggingFace)
- TensorFlow 2.x
- Matplotlib

## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/apalapramanik/Spanish-News-Classification.git
   cd Spanish-News-Classification
   ```

2. Install dependencies:
   ```bash
   pip install numpy pandas scikit-learn transformers tensorflow matplotlib
   ```

3. Run the training script:
   ```bash
   python beto3.py
   ```

## Training Details

- **Pre-trained Model:** `dccuchile/bert-base-spanish-wwm-cased`
- **Optimizer:** Adam (learning rate: 2e-5)
- **Loss Function:** Categorical Cross-Entropy
- **Epochs:** 50
- **Batch Size:** 16
- **Data Split:** 60% train / 20% validation / 20% test

## Output

The script generates three visualization files:

| File | Description |
|------|-------------|
| `accuracy.png` | Training vs validation accuracy over epochs |
| `loss.png` | Training vs validation loss over epochs |
| `confusion_matrix.png` | Confusion matrix showing classification performance |

## Project Structure

```
Spanish-News-Classification/
├── README.md           # Project documentation
├── beto3.py            # Main training script
└── df_total.csv        # Dataset (4,570 Spanish news articles)
```

## References

- [BETO: Spanish BERT](https://github.com/dccuchile/beto) - Pre-trained Spanish language model
- [HuggingFace Transformers](https://huggingface.co/docs/transformers/) - Model library
