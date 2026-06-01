# Midea MFM01D110WB RAG LoRA API

This repository contains the implementation of a Retrieval-Augmented Generation (RAG) pipeline with LoRA fine-tuning and RESTful API deployment.

The project was developed as part of the **Second Assessment** for the **Advanced Topics in Artificial Intelligence (Tópicos Avançados em IA A)** course in the **Computer Engineering Program** at the **Federal University of Rio Grande do Norte (UFRN)**.

## Project Overview

The system uses the user manual of the **Midea MFM01D110WB 11kg washing machine** as its knowledge base and implements a complete end-to-end workflow for domain-specific question answering.

The project includes:

- PDF document processing
- Dataset generation (instruction-response pairs)
- Retrieval-Augmented Generation (RAG)
- LoRA-based fine-tuning (PEFT)
- Quantitative and qualitative model evaluation
- RESTful API deployment

## Project Structure

```text
midea-mfm01d110wb-rag-lora-api/
├── 01_rag.ipynb
├── 02_lora.ipynb
├── 03_avaliacao_modelo_finetuned.ipynb
├── main.py
├── data/
├── models/
├── results/
├── requirements.txt
└── README.md
```

## Assessment Stages

### Stage 1 – RAG Dataset Generation

Generation of instruction-response pairs from the washing machine user manual.

### Stage 2 – LoRA Fine-Tuning

Parameter-efficient fine-tuning of language models using Low-Rank Adaptation (LoRA).

### Stage 3 – Model Evaluation

Evaluation of the fine-tuned models using standard Natural Language Processing metrics.

### Stage 4 – RESTful API Deployment

Integration and deployment of the trained models through a REST API.

## Knowledge Source

- Midea MFM01D110WB 11kg Washing Machine User Manual

## Technologies

- Python
- PyTorch
- Hugging Face Transformers
- PEFT (LoRA)
- LangChain
- ChromaDB / FAISS
- FastAPI
- Jupyter Notebook

## Author

**[Ana Luiza Costa de Oliveira]**  
Computer Engineering Undergraduate  
Federal University of Rio Grande do Norte (UFRN)

## Academic Information

**Course:** Advanced Topics in Artificial Intelligence (Tópicos Avançados em Inteligência Artificial A)  
**Assessment:** Second Assessment  
**Institution:** Federal University of Rio Grande do Norte (UFRN)