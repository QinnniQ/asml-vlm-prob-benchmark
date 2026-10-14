<p align="center">
  <img src="logo.png" alt="ASML VLM Benchmark Logo" width="60%">
</p>

# ASML Vision-Language Benchmark with Probabilistic Scoring

**A small, reproducible study of model confidence under domain shift.** [Portfolio](https://profile.nickcloud.dev/) · [LinkedIn](https://www.linkedin.com/in/nicholai-gay-201905148/)

> Independent portfolio project using synthetic imagery. It is not affiliated with or evaluated by ASML, and the results do not represent performance on production wafer data.

## Overview

This project investigates how a general-purpose vision-language model (**OpenAI CLIP ViT-B/32**) behaves on synthetic wafer-defect images inspired by semiconductor inspection.

Real-world EUV lithography diagnostics depend not only on identifying wafer defects — but on **knowing when the model is uncertain**.  
This repository benchmarks CLIP on synthetic wafer-defect imagery and evaluates **calibration, entropy, reliability, and confidence** to identify gaps between model predictions and trustworthy engineering-grade performance.

---

## Project Goals

### **1. Evaluate CLIP on ASML-style diagnostic imagery**
- Synthetic wafer defects: *scratch*, *particle contamination*, *none*
- Zero-shot classification using natural language prompts  
- Measure accuracy & distribution of predictions

### **2. Quantify Uncertainty**
- Probability entropy  
- Max-prob (confidence)  
- Brier score  
- Expected Calibration Error (ECE)

### **3. Generate Reliability & Calibration Reports**
These measures reveal **how much the model overestimates itself**, a critical issue in semiconductor safety-critical environments.

---

## Key Results

| Metric | Value |
|--------|--------|
| **Samples** | **50** |
| **Accuracy** | **34%** |
| **Expected Calibration Error (ECE)** | **0.521** |
| **Maximum Calibration Error (MCE)** | **0.750** |
| **Mean confidence** | **0.817** |

Source: [archived calibration report](reports_archive/reports_20251208_171342/calibration_report.json). This is one small synthetic run; it does not establish performance on real semiconductor imagery. The current calibration code includes confidence values of exactly 1.0 in the final bin, so rerunning the pipeline may change archived metrics.

### Interpretation

- The observed mean confidence is much higher than the observed accuracy on this synthetic sample.
- This result motivates further evaluation with larger, representative datasets before any deployment claim.

---

## Notebook: Uncertainty Analysis

The notebook `notebooks/uncertainty_analysis.ipynb` includes:

- Confidence histogram  
- Entropy histogram  
- Calibration curve  
- ECE computation  
- Scatter: confidence vs correctness  
- Summary metrics  

These plots help inspect where confidence and correctness diverge in this sample.

---

## Project Structure

```
asml-vlm-prob-benchmark/
│
├── data/
│   ├── images/                  # Synthetic wafer images
│   └── annotations/             # JSON annotations (task, label, prompt)
│
├── scripts/
│   ├── run_all.py               # FULL pipeline runner
│   ├── evaluate_clip.py         # Model evaluation and JSON export
│   ├── eval_calibration.py      # Calibration metrics
│   └── generate_reliability_plot.py
│
├── src/
│   ├── dataset.py               # WaferDataset loader
│   ├── model.py                 # CLIP wrapper (safetensors enforced)
│   └── metrics.py               # Confidence, entropy, Brier, calibration utils
│
├── reports/                     # Generated locally by the pipeline
│   ├── clip_results.json
│   ├── calibration_report.json
│   └── reliability_diagram.png
│
├── reports_archive/             # Auto-archived previous runs
└── README.md
```

---

# What the result means

The 50-image synthetic run shows a gap between confidence and correctness. It is useful as an uncertainty-analysis exercise, but sample size, synthetic image generation, prompt wording, and class balance limit broader conclusions.

---

# Why This Project Matters for Semiconductor AI

Semiconductor manufacturing demands:
- **High diagnostic precision**
- **Interpretability**
- **Fail-safe uncertainty awareness**

This benchmark demonstrates:
- How a generalist VLM behaves under **domain shift**
- Where confidence diverges from truth
- How models can be evaluated for **trustworthiness** before deployment  
- A starting point for better-scoped VLM benchmarks and multimodal datasets

Next steps include evaluating on representative labeled data, reporting uncertainty intervals, and comparing with task-specific baselines.

---

# Installation

```powershell
git clone https://github.com/QinnniQ/asml-vlm-prob-benchmark
cd asml-vlm-prob-benchmark
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The pinned PyTorch packages in `requirements.txt` target NVIDIA CUDA 12.1 on Windows/Linux. The model can run on CPU, but this installation still downloads the CUDA build; for a smaller CPU-only or another accelerator setup, install the matching PyTorch build and adapt the requirements. The first model run downloads CLIP weights from Hugging Face.

---

# Run Full Pipeline

```powershell
python scripts/run_all.py
```

The runner resolves paths from the repository directory and can be launched from another working directory. To run the calibration unit checks without downloading model weights, install NumPy and run `python -m unittest discover -s tests -v`.

Outputs are automatically stored in:

```
/reports
/reports_archive
```
---

## Acknowledgements

- OpenAI CLIP  
- HuggingFace Transformers  
- Semiconductor inspection as inspiration for the diagnostic framing

---

## Contact

If you’re interested in semiconductor AI, vision-language models, or uncertainty-aware ML:

**Nicholai Gay**  
AI Engineer  
📧 nicholai.gay@gmail.com 
🔗 LinkedIn: https://www.linkedin.com/in/nicholai-gay-201905148/
