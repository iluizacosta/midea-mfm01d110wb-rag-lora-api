# 🧠 Midea Washer-Dryer QA System with LoRA Fine-Tuning, Evaluation and REST API

> End-to-end NLP pipeline for domain adaptation of Large Language Models using LoRA, built from the Midea MFM01D110WB washer-dryer user manual.

## 📖 Overview

This project was developed for the course **Advanced Topics in Artificial Intelligence (Tópicos Avançados em IA A)** at the **Federal University of Rio Grande do Norte (UFRN)**.

The goal was to build a complete AI workflow covering:

* Dataset generation from a real-world technical manual;
* Manual dataset curation and validation;
* Parameter-Efficient Fine-Tuning (PEFT) using LoRA;
* Evaluation of multiple language models;
* Deployment through a FastAPI REST API;
* Preparation for future Retrieval-Augmented Generation (RAG) integration.

The knowledge domain used throughout the project is the **Midea MFM01D110WB Washer-Dryer User Manual**.

---

# 🚀 Project Pipeline

```text
PDF Manual
    ↓
Text Extraction
    ↓
Chunk Generation
    ↓
Instruction–Response Dataset
    ↓
Manual Curation
    ↓
LoRA Fine-Tuning
    ↓
Model Evaluation
    ↓
REST API Deployment
```

---

# 🎯 Objectives

The project was divided into four stages.

## Stage 1 — Dataset Construction

* Extract text from the Midea user manual.
* Generate instruction–response pairs automatically.
* Remove duplicates and irrelevant samples.
* Perform manual curation.
* Build a supervised fine-tuning dataset.

## Stage 2 — LoRA Fine-Tuning

Train four different language models:

### Causal Models

* EleutherAI/Pythia-14M
* EleutherAI/Pythia-31M

### Seq2Seq Models

* Google/Flan-T5-Small
* Google/T5-v1.1-Small

## Stage 3 — Evaluation

Evaluate all models using:

* Perplexity (PPL)
* BLEU
* ROUGE
* Faithfulness
* Answer Relevance
* Plan Adherence

## Stage 4 — REST API

Deploy all fine-tuned models through FastAPI and provide a simple web interface for interaction.

---

# 📂 Project Structure

```text
.
├── api/
│   ├── main.py
│   └── requirements.txt
│
├── data/
│   ├── chunks/
│   ├── pdf/
│   ├── processed/
│   └── splits/
│
├── models/
│   ├── lora_causal_model_1/
│   ├── lora_causal_model_2/
│   ├── lora_seq2seq_model_1/
│   ├── lora_seq2seq_model_2/
│   ├── results_pythia14m/
│   ├── results_pythia31m/
│   ├── results_flan_t5_small/
│   └── results_t5_v1_1_small/
│
├── notebooks/
│   ├── 01_rag.ipynb
│   ├── 02_lora_causal_1.ipynb
│   ├── 02_lora_causal_2.ipynb
│   ├── 02_lora_seq2seq_1.ipynb
│   ├── 02_lora_seq2seq_2.ipynb
│   ├── 03_avaliacao_modelo_finetuned_*.ipynb
│   └── requirements.txt
│
├── reports/
├── static/
│   └── index.html
│
├── README.md
└── .gitignore
```

---

# 📊 Dataset

The final curated dataset contains:

| Split | Samples |
| ----- | ------: |
| Train |      99 |
| Test  |      25 |
| Total |     124 |

Dataset files:

```text
data/splits/train.jsonl
data/splits/test.jsonl
```

Example:

```json
{
  "Instruction": "How do you unlock the door during a wash cycle?",
  "Output": "Press START/PAUSE for 3 seconds."
}
```

---

# 🤖 Fine-Tuned Models

## Causal Model 1 — Pythia-14M

Base Model:

```text
EleutherAI/pythia-14m
```

LoRA Configuration:

```python
r = 16
lora_alpha = 32
lora_dropout = 0.05
target_modules = ["query_key_value"]
```

Trainable Parameters:

```text
49,152
```

---

## Causal Model 2 — Pythia-31M

Base Model:

```text
EleutherAI/pythia-31m
```

LoRA Configuration:

```python
r = 16
lora_alpha = 32
lora_dropout = 0.05
target_modules = ["query_key_value"]
```

Trainable Parameters:

```text
98,304
```

---

## Seq2Seq Model 1 — Flan-T5-Small

Base Model:

```text
google/flan-t5-small
```

LoRA Configuration:

```python
target_modules=["q", "v"]
task_type=TaskType.SEQ_2_SEQ_LM
```

---

## Seq2Seq Model 2 — T5-v1.1-Small

Base Model:

```text
google/t5-v1_1-small
```

Additional requirements:

```bash
pip install protobuf
pip install sentencepiece
```

---

# 📈 Evaluation Metrics

The models were evaluated using:

| Metric           | Purpose                    |
| ---------------- | -------------------------- |
| Perplexity (PPL) | Language modeling quality  |
| BLEU             | N-gram precision           |
| ROUGE            | Content overlap            |
| Faithfulness     | Consistency with reference |
| Answer Relevance | Question-answer alignment  |
| Plan Adherence   | Structural similarity      |

---

# 🏆 Best Performing Model

The best overall model was:

```text
Flan-T5-Small + LoRA
```

Main reasons:

* Lowest perplexity
* Highest BLEU score
* Highest ROUGE scores
* Highest faithfulness
* Highest answer relevance

Although the results were promising, the final performance was limited by the small dataset size and the complexity of adapting language models to a highly specific technical domain.

---

# ⚙️ API Installation

The REST API is located inside the `api` directory.

Install the API dependencies:

```bash
pip install -r api/requirements.txt
```

Alternatively:

```bash
cd api
pip install -r requirements.txt
cd ..
```

---

# ▶️ Running the API

From the project root:

```bash
python -m uvicorn api.main:app --reload
```

The application will be available at:

```text
http://localhost:8000
```

Swagger documentation:

```text
http://localhost:8000/docs
```

Health endpoint:

```text
http://localhost:8000/health
```

---

# 📡 API Endpoints

## GET /modelos

Returns all available models.

## POST /chat

Example request:

```json
{
  "modelo": "seq2seq-flan-t5-small",
  "mensagem": "How do you unlock the door during a wash cycle?"
}
```

Example response:

```json
{
  "resposta": "Press START/PAUSE for 3 seconds.",
  "modelo": "seq2seq-flan-t5-small",
  "tokens_gerados": 8
}
```

## GET /health

Returns API status and loaded models.

---

# 🛠️ Technologies

* Python
* PyTorch
* Hugging Face Transformers
* PEFT
* LoRA
* FastAPI
* Uvicorn
* PyMuPDF
* Pandas
* NumPy
* Matplotlib

---

# ⚠️ Limitations

The project successfully implemented the complete pipeline, but several limitations were identified:

* Small dataset (124 examples);
* Domain-specific technical vocabulary;
* Limited computational resources;
* Small model sizes;
* Restricted training time.

These limitations significantly affected the final performance of the models.

---

# 🔮 Future Work

Possible improvements include:

* Larger supervised datasets;
* More extensive manual curation;
* Additional LoRA experiments;
* Stronger foundation models;
* Retrieval-Augmented Generation (RAG);
* Semantic evaluation metrics based on embeddings;
* Production-ready deployment.

---

# 👩‍💻 Author

**Ana Luiza Oliveira**

Computer Engineering Undergraduate
Federal University of Rio Grande do Norte (UFRN)

Course: **Advanced Topics in Artificial Intelligence**
