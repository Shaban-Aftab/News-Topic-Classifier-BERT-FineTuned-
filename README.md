# 📰 News Topic Classifier (BERT + PyTorch First-Principles)

A complete, high-performance **News Topic Classifier** fine-tuned on the AG News dataset from absolute first principles using **PyTorch** and **BERT** (no generic high-level Hugging Face trainers). Includes a fully customized sequence classification architecture, dynamic padding optimization, mixed-precision (FP16) training, metric profiling, Post-Training Dynamic INT8 Quantization, and an interactive Streamlit UI with dynamic remote weight loading.

🚀 **Live Model Repository on Hugging Face:** [shabanaftab01/news-topic-classifier-bert](https://huggingface.co/shabanaftab01/news-topic-classifier-bert)

---

## ✨ Features & Architecture

* **PyTorch First-Principles Pipeline**: Written completely in native PyTorch (`nn.Module`, manual training and backpropagation loop, gradient clipping, custom collators).
* **Multi-Layer Feature Pooling**: Extracted and concatenated the `[CLS]` token contextual representation from the **last 4 hidden layers of BERT** (`768 * 4 = 3072` features) to capture both deep semantic and local syntactic attributes.
* **High-Performance Preprocessing**: Custom `Dataset` with regular expression-based HTML tag cleaning and a custom **Dynamic Padding Collator** that dynamically pads sentences on a batch-by-batch basis, reducing training time by ~35%.
* **Robust Training Optimization**: Implemented **Label Smoothing** regularization ($\alpha=0.1$) to prevent overconfidence, **Automatic Mixed Precision (AMP)** for fast training on T4 GPUs, and a linear learning rate warmup + cosine decay scheduler.
* **Production-Grade CPU Optimization**: Post-Training **Dynamic INT8 Quantization** applied to all `Linear` modules. Speed benchmarks demonstrate a **~1.40x CPU inference speedup** while reducing the parameter memory footprint by 75% (~440MB to ~110MB).
* **Interactive Streamlit Web Dashboard**: Serve predictions with real-time inference latency logs, category confidence metrics, and probability distribution charts.
* **Dynamic Weight Fetching**: Select between loading local weights or pulling remote model weights directly from the [Hugging Face Hub](https://huggingface.co/shabanaftab01/news-topic-classifier-bert) on the fly using `hf_hub_download`.

---

## 📂 Project Structure

```
├── news-topic-classifier-bert-finetuned.ipynb  # Interactive Jupyter Notebook (with complete Kaggle run logs)
├── app.py                                     # Streamlit web server with Dynamic Quantization
├── preprocess.py                              # Custom NewsDataset & Dynamic Padding Batch Collator
├── model.py                                   # NewsClassifier network (BERT Last-4 Layer Concat head)
├── train.py                                   # Manual PyTorch optimization loop (AMP, Schedulers, SMOOTH)
├── evaluate.py                                # Performance reporter & confidence failure diagnostics
├── upload_to_hf.py                            # Programmatic weights & tokenizer HF uploader script
├── requirements.txt                           # Dependency file
└── README.md                                  # Documentation
```

---

## ⚡ Setup & Installation

Clone the repository and install all required scientific and web dependencies:

```bash
git clone https://github.com/Shaban-Aftab/News-Topic-Classifier-BERT-FineTuned-.git
cd News-Topic-Classifier-BERT-FineTuned-
pip install -r requirements.txt
```

---

## 🎯 Usage

### 1. Interactive Notebook Execution (GPU Recommended)
Open and run **`news-topic-classifier-bert-finetuned.ipynb`** in Kaggle or Google Colab. The notebook features step-by-step cells containing:
* Environment and GPU hardware diagnostics.
* Tokenization, custom dataset builder, and dynamic padders.
* Model construction and custom optimization loops.
* Diagnostic metric printouts profiling top confident errors.
* Secure Hugging Face Hub publishing using Kaggle Secrets (`UserSecretsClient`).

### 2. Run the Streamlit Servable Web App
Execute the app locally or run it directly on Streamlit Community Cloud:

```bash
streamlit run app.py
```

* **Local Mode**: Select "Local Checkpoint" in the left panel to load weights from your local `saved_models/` folder.
* **Cloud / Hub Mode**: Select "Hugging Face Hub" and enter your repository path (`shabanaftab01/news-topic-classifier-bert`) to stream and quantize model weights instantly from the cloud.

### 3. Evaluate Metrics
Run verification diagnostics on local evaluation checkpoints:

```bash
python evaluate.py
```

---

## 📊 Scientific Performance Report

| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **World** | 96.41% | 94.52% | 95.45% | 511 |
| **Sports** | 98.47% | 97.91% | 98.19% | 526 |
| **Business** | 91.03% | 88.20% | 89.59% | 449 |
| **Sci/Tech** | 89.83% | 94.55% | 92.13% | 514 |
| **Overall Accuracy** | | | **94.00%** | **2000** |

### 🚀 Optimization Benchmarks on CPU (Intel/AMD)
* **Standard FP32 Model Latency**: 141.82 ms
* **Dynamic INT8 Quantized Model Latency**: 101.38 ms
* **Inference Acceleration**: **1.40x Speedup** 🚀

---

## 🤝 Contributions & Educational Reference

This project was built from absolute first principles to demystify complex Transformer pipelines. You can find structural deep dives, mathematical rationales, and training mechanics inside [News_Topic_Classifier_Masterclass.md](News_Topic_Classifier_Masterclass.md). Feel free to fork, experiment with layer architectures, and submit pull requests!
