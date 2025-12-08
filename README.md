<p align="center">
  <img src="logo.png" alt="ASML VLM Benchmark Logo" width="60%">
</p>

# ASML Vision-Language Benchmark with Probabilistic Scoring  
### Evaluating CLIP on Semiconductor-Style Diagnostic Tasks

This project benchmarks a general-purpose Vision-Language Model (OpenAI CLIP ViT-B/32) on a synthetic diagnostic dataset designed to mimic ASML-style wafer defect scenarios.  

The goal is to evaluate not only classification accuracy, but also uncertainty, entropy, and calibration quality which are critical for safety-critical semiconductor diagnostic workflows.

This repository demonstrates:
- How VLMs behave on domain-specific semiconductor-style imagery  
- How confidence and accuracy diverge in out-of-domain settings  
- Why calibration matters for industrial diagnostic systems  
- How to build a complete uncertainty-aware benchmarking pipeline  

---

## Project Overview

Modern ASML tools produce rich visual diagnostic data. While Vision-Language Models (VLMs) are powerful, they often lack **domain alignment** required for wafer defect analysis.

This project explores:

### **1. A synthetic wafer defect dataset**  
Three diagnostic classes:
- `none`
- `scratch`
- `particle`

### **2. A CLIP-based baseline model**
Evaluated using:
- Softmax probabilities  
- Entropy  
- Confidence  
- Brier score  
- Expected Calibration Error (ECE)  

### **3. Visual diagnostic analysis**
- Confidence histogram  
- Entropy distribution  
- Calibration / reliability diagram  
- Confidence vs correctness scatter

---

## Key Results

| Metric | Value |
|--------|--------|
| **Accuracy** | **34%** |
| **Expected Calibration Error (ECE)** | **0.52** |
| **Avg Confidence** | ~0.82 |
| **Avg Entropy** | ~0.55 |

### Interpretation

- CLIP is **highly overconfident** on semiconductor-style imagery  
- ECE > 0.5 indicates **severe miscalibration**  
- Accuracy remains low due to out-of-domain data  
- Highlights the need for **domain-specific VLMs or fine-tuning**

---

## Notebook: Uncertainty Analysis

The notebook `notebooks/uncertainty_analysis.ipynb` includes:

- Confidence histogram  
- Entropy histogram  
- Calibration curve  
- ECE computation  
- Scatter: confidence vs correctness  
- Summary metrics  

These plots reveal precisely where and why the model fails.

---

## Project Structure

```
asml-vlm-prob-benchmark/
│
├── data/
│   ├── images/                ← synthetic wafer images
│   └── annotations/
│       └── annotations.json   ← ground-truth labels
│
├── scripts/
│   ├── generate_dataset.py    ← dataset generation
│   └── evaluate_clip.py       ← main evaluation + uncertainty scoring
│
├── notebooks/
│   └── uncertainty_analysis.ipynb
│
├── reports/
│   ├── clip_results.json      ← raw model predictions and metrics
│   └── asml_vlm_benchmark_report.md
│
└── README.md
```

---

## Methods

### **Uncertainty Metrics Implemented**
- **Confidence:** `max(p)`  
- **Entropy:** `-Σ p log p`  
- **Brier Score:** squared probabilistic error  
- **ECE (Expected Calibration Error):** measures miscalibration  

This aligns with modern evaluation standards in high-risk AI systems.

### **Model**
- CLIP ViT-B/32  
- Loaded with `safetensors` only (avoids CVE-2025-32434 torch.load restrictions)

---

## Installation

```bash
git clone https://github.com/yourusername/asml-vlm-prob-benchmark
cd asml-vlm-prob-benchmark

python -m venv .venv
.\.venv\Scriptsctivate

pip install -r requirements.txt
```

Or install manually:

```bash
pip install torch torchvision transformers matplotlib numpy pillow ipykernel
```

---

## Running the Evaluation

```bash
python scripts/evaluate_clip.py
```

Results are saved to:

```
reports/clip_results.json
```

---

## Running the Notebook

Open VS Code → Jupyter Notebook → Select Kernel:

```
Python 3.11 (.venv)
```

Run:

```
notebooks/uncertainty_analysis.ipynb
```

---

## ASML Context

Semiconductor lithography diagnostics require:

- High model reliability  
- Robust uncertainty quantification  
- Calibration for trustworthy decision-making  

This project illustrates how general-purpose VLMs **fail silently** without calibration, and why specialized domain training or fine-tuning is required.

This benchmark serves as a **starting point** for building semiconductor-aware AI systems.

---

## Future Work

- Training a domain-tuned CLIP/SigLIP model  
- Adding more diagnostic classes (blobs, streaks, edge defects)  
- Integrating SEM-style textures  
- Realistic noise to simulate lithography optics  
- Temperature scaling or isotonic calibration  

---

## Acknowledgements

- OpenAI CLIP  
- HuggingFace Transformers  
- ASML for inspiring the diagnostic problem framing  

---

## Contact

If you’re interested in semiconductor AI, vision-language models, or uncertainty-aware ML:

**Nicholai Gay**  
AI Engineer  
📧 nicholai.gay@gmail.com 
🔗 LinkedIn: https://www.linkedin.com/in/nicholai-gay-201905148/
