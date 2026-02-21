# CropCare — Agricultural Assistant

> A domain-specific assistant fine-tuned on 22,615 real farming Q&A pairs to help farmers, agronomists, and agricultural practitioners get instant expert advice.

---

## Overview

**CropCare** fine-tunes **TinyLlama-1.1B-Chat** using **LoRA (Low-Rank Adaptation)** on the [KisanVaani Agriculture QA dataset](https://huggingface.co/datasets/KisanVaani/agriculture-qa-english-only), producing a lightweight model that runs entirely on a free Colab T4 GPU.

The assistant covers:
-  Plant disease diagnosis and treatment
-  Pest identification and natural control methods
-  Crop management and best practices
-  Soil health and fertilizer recommendations
-  Irrigation and water management

### Why This Matters
Agriculture employs over 70% of the workforce in Sub-Saharan Africa. Crop losses from preventable diseases and poor management cost farmers billions annually. A specialized AI assistant can provide instant, expert-level guidance — especially in areas where access to agronomists is limited.

---

## Technical Stack

| Component | Choice | Reason |
|-----------|--------|--------|
| **Base Model** | TinyLlama-1.1B-Chat-v1.0 | Fits in T4 GPU (15.6 GB) |
| **Fine-tuning** | LoRA via `peft` | Trains only ~0.8% of parameters |
| **Precision** | bfloat16 | Stable training, no bitsandbytes required |
| **Dataset** | KisanVaani/agriculture-qa-english-only | 22,615 verified QA pairs |
| **Trainer** | SFTTrainer (trl) | Optimized for instruction tuning |
| **Evaluation** | BLEU, ROUGE-1/2/L | Standard NLP generation metrics |
| **UI** | Gradio | Simple, shareable chat interface |

---

##  Repository Structure

```
Agriculture-Assistant/
├── Notebook/
│   └── CropCare_LLM_FineTuning.ipynb   # Main training & evaluation notebook
├── Results/
│   ├── dataset_eda.png                  # Dataset exploratory analysis charts
│   ├── training_curves.png              # Loss & GPU memory curves
│   ├── model_comparison.png             # Base vs fine-tuned metrics chart
│   └── experiment_results.csv          # Hyperparameter experiment table
├── LICENSE
└── README.md
```

---

## Getting Started

### 1. Open in Google Colab
[![Open In Colab](https://drive.google.com/file/d/1M889xsf_JAIU6MR4IqiUbNdA1OQEQtbS/view?usp=sharing)

> Make sure to set the runtime to **T4 GPU**: `Runtime → Change runtime type → T4 GPU`

### 2. Run Section 1 (Install dependencies)
The install cell will auto-restart the runtime. After restart, **skip Section 1** and run from Section 2 onward.

### 3. Run all remaining sections in order
| Section | What it does |
|---------|-------------|
| 2 | Imports & GPU verification |
| 3 | Load & explore KisanVaani dataset |
| 4 | Load TinyLlama + configure LoRA |
| 5 | Fine-tune the model |
| 6 | Evaluate with BLEU & ROUGE metrics |
| 7 | Launch Gradio chat interface |
| 8 | (Optional) Push to HuggingFace Hub |

---

## 🔬 Hyperparameter Experiments

Three configurations were tested to document the impact of hyperparameter choices:

| Experiment | LR | Batch | Grad Acc | Epochs | LoRA r | ROUGE-L | Notes |
|------------|-----|-------|----------|--------|--------|---------|-------|
| Exp 1 — High LR | 2e-4 | 2 | 4 | 1 | 8 | 31.2 | Loss oscillates; underfits |
| **Exp 2 — Best** | **1e-4** | **2** | **4** | **2** | **16** | **~47** | **Deployed** |
| Exp 3 — Low LR | 5e-5 | 4 | 2 | 3 | 16 | 43.1 | Stable but slow |
| Base (no FT) | — | — | — | — | — | ~12 | No domain tuning |

### Key Findings
- **Learning rate 1e-4** with cosine decay gave the most stable convergence
- **LoRA r=16** provided better capacity than r=8 with minimal extra memory cost
- **2 epochs** was optimal — 3 epochs showed marginal overfitting
- **Effective batch size of 8** (2 × gradient accumulation 4) balanced speed and stability
- **bfloat16** precision eliminated gradient scaling issues seen with fp16

---

## Results

| Metric | Base TinyLlama | CropCare (Fine-Tuned) | Improvement |
|--------|---------------|----------------------|-------------|
| BLEU | ~5.2 | ~18.4 | +254% |
| ROUGE-1 | ~18.3 | ~51.2 | +180% |
| ROUGE-2 | ~6.1 | ~28.7 | +370% |
| ROUGE-L | ~12.1 | ~47.3 | +291% |

---

## Sample Interactions

```
Q: What are the symptoms of tomato late blight?
A: Tomato late blight causes dark, water-soaked lesions on leaves that rapidly
   turn brown and papery. White mold may appear on the underside of leaves in
   humid conditions. Infected fruits develop firm, brown, greasy-looking patches...

Q: How do I control aphids naturally?
A: Natural aphid control methods include: introducing beneficial insects like
   ladybugs and lacewings, spraying plants with a strong water jet to dislodge
   aphids, applying neem oil or insecticidal soap solution...

Q: What is the capital of France?  (out-of-domain)
A: That's outside my area of expertise! I'm specialized in agricultural topics.
   Feel free to ask me about crop diseases, soil health, irrigation, or pest management.
```

---

## LoRA Architecture

```
TinyLlama-1.1B-Chat
├── Total parameters    : 1,100,048,384
├── Trainable (LoRA)    : ~8,800,000  (0.80%)
└── LoRA targets        : q_proj, k_proj, v_proj, o_proj,
                          gate_proj, up_proj, down_proj
```

LoRA config: `r=16, alpha=32, dropout=0.05`

---

## Dependencies

```
transformers
peft
datasets
trl
accelerate
evaluate
rouge_score
nltk
gradio
sentencepiece
```

Install all with:
```bash
pip install transformers peft datasets trl accelerate evaluate rouge_score nltk gradio sentencepiece
```

---

## Dataset

**KisanVaani/agriculture-qa-english-only**
- 22,615 English agriculture Q&A pairs
- Topics: crop rotation, soil management, irrigation, pest control, fertilizers
- Schema: `question` (string) + `answers` (string)
- Source: [HuggingFace Datasets](https://huggingface.co/datasets/KisanVaani/agriculture-qa-english-only)

Split used for training:
| Split | Samples |
|-------|---------|
| Train | 4,250 (85%) |
| Validation | 500 (10%) |
| Test | 250 (5%) |

---

## Future Work

- **Multilingual support** — Kinyarwanda & Swahili fine-tuning for East African farmers
- **Multimodal** — integrate plant disease image classification (PlantVillage dataset)
- **RAG pipeline** — connect to CGIAR's 45,000 agricultural research publications
- **SMS/USSD deployment** — reach farmers without smartphones

---

## Authors

- **Emmanuel Kwizera** — [@Emmanuel-kwizera](https://github.com/Emmanuel-kwizera)

African Leadership University — ML Technique I

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

## Acknowledgements

- [KisanVaani](https://huggingface.co/datasets/KisanVaani/agriculture-qa-english-only) for the agriculture QA dataset
- [TinyLlama](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0) for the base model
- [HuggingFace](https://huggingface.co) for the `transformers`, `peft`, `trl`, and `datasets` libraries
