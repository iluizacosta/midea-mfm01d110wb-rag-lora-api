import os
import logging
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    pipeline
)

from peft import PeftModel


# =============================================================================
# PATHS
# =============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)

MODELS_DIR = os.path.join(PROJECT_DIR, "models")
STATIC_DIR = os.path.join(PROJECT_DIR, "static")

print("BASE_DIR:", BASE_DIR)
print("PROJECT_DIR:", PROJECT_DIR)
print("MODELS_DIR:", MODELS_DIR)
print("STATIC_DIR:", STATIC_DIR)


# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

logger = logging.getLogger(__name__)


# =============================================================================
# FASTAPI APP
# =============================================================================

app = FastAPI(
    title="LoRA Models API - Washer-Dryer Manual",
    description="REST API for interacting with four LoRA fine-tuned language models.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# GLOBAL MODELS
# =============================================================================

MODELS: dict = {}


MODEL_INFO = {
    "causal-pythia-14m": {
        "id": "causal-pythia-14m",
        "nome": "Pythia-14M Fine-Tuned with LoRA",
        "descricao": "Causal language model fine-tuned with LoRA on the Midea washer-dryer manual.",
        "base_model": "EleutherAI/pythia-14m",
        "lora_path": os.path.join(MODELS_DIR, "lora_causal_model_1"),
        "architecture": "causal"
    },
    "causal-pythia-31m": {
        "id": "causal-pythia-31m",
        "nome": "Pythia-31M Fine-Tuned with LoRA",
        "descricao": "Second causal language model fine-tuned with LoRA on the Midea washer-dryer manual.",
        "base_model": "EleutherAI/pythia-31m",
        "lora_path": os.path.join(MODELS_DIR, "lora_causal_model_2"),
        "architecture": "causal"
    },
    "seq2seq-flan-t5-small": {
        "id": "seq2seq-flan-t5-small",
        "nome": "Flan-T5-Small Fine-Tuned with LoRA",
        "descricao": "Seq2Seq instruction-tuned model adapted with LoRA for washer-dryer manual Q&A.",
        "base_model": "google/flan-t5-small",
        "lora_path": os.path.join(MODELS_DIR, "lora_seq2seq_model_1"),
        "architecture": "seq2seq"
    },
    "seq2seq-t5-v1-1-small": {
        "id": "seq2seq-t5-v1-1-small",
        "nome": "T5-v1.1-Small Fine-Tuned with LoRA",
        "descricao": "Seq2Seq model fine-tuned with LoRA using the Midea washer-dryer manual dataset.",
        "base_model": "google/t5-v1_1-small",
        "lora_path": os.path.join(MODELS_DIR, "lora_seq2seq_model_2"),
        "architecture": "seq2seq",
        "use_fast": False
    }
}


# =============================================================================
# LOAD MODELS
# =============================================================================

def carregar_modelo_causal(model_key: str) -> dict:
    config = MODEL_INFO[model_key]

    logger.info(f"Loading causal model: {model_key}")
    logger.info(f"LoRA path: {config['lora_path']}")

    if not os.path.exists(config["lora_path"]):
        raise FileNotFoundError(f"LoRA directory not found: {config['lora_path']}")

    tokenizer = AutoTokenizer.from_pretrained(config["base_model"])

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(config["base_model"])

    model = PeftModel.from_pretrained(
        base_model,
        config["lora_path"]
    )

    model.eval()

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=-1
    )

    logger.info(f"Model loaded successfully: {model_key}")

    return {
        "model": model,
        "tokenizer": tokenizer,
        "pipeline": pipe,
        "architecture": "causal"
    }


def carregar_modelo_seq2seq(model_key: str) -> dict:
    config = MODEL_INFO[model_key]

    logger.info(f"Loading Seq2Seq model: {model_key}")
    logger.info(f"LoRA path: {config['lora_path']}")

    if not os.path.exists(config["lora_path"]):
        raise FileNotFoundError(f"LoRA directory not found: {config['lora_path']}")

    tokenizer = AutoTokenizer.from_pretrained(
        config["base_model"],
        use_fast=config.get("use_fast", True)
    )

    base_model = AutoModelForSeq2SeqLM.from_pretrained(config["base_model"])

    model = PeftModel.from_pretrained(
        base_model,
        config["lora_path"]
    )

    model.eval()

    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        device=-1
    )

    logger.info(f"Model loaded successfully: {model_key}")

    return {
        "model": model,
        "tokenizer": tokenizer,
        "pipeline": pipe,
        "architecture": "seq2seq"
    }


@app.on_event("startup")
async def startup_event():
    global MODELS

    logger.info("=" * 60)
    logger.info("Starting server and loading LoRA fine-tuned models...")
    logger.info(f"Base directory: {BASE_DIR}")
    logger.info(f"Models directory: {MODELS_DIR}")
    logger.info("=" * 60)

    for model_key, config in MODEL_INFO.items():
        try:
            if config["architecture"] == "causal":
                MODELS[model_key] = carregar_modelo_causal(model_key)

            elif config["architecture"] == "seq2seq":
                MODELS[model_key] = carregar_modelo_seq2seq(model_key)

        except Exception as error:
            logger.error(f"Error loading model '{model_key}': {error}")

    logger.info("=" * 60)
    logger.info(f"Loaded models: {list(MODELS.keys())}")
    logger.info("=" * 60)


# =============================================================================
# SCHEMAS
# =============================================================================

class ChatRequest(BaseModel):
    modelo: str
    mensagem: str
    max_tokens: Optional[int] = 80
    temperatura: Optional[float] = 0.7


class ChatResponse(BaseModel):
    resposta: str
    modelo: str
    tokens_gerados: int


# =============================================================================
# ENDPOINTS
# =============================================================================

@app.get("/modelos", response_class=JSONResponse)
async def listar_modelos():
    disponiveis = [
        {
            "id": info["id"],
            "nome": info["nome"],
            "descricao": info["descricao"]
        }
        for key, info in MODEL_INFO.items()
        if key in MODELS
    ]

    return {"modelos": disponiveis}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if request.modelo not in MODELS:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{request.modelo}' not found. Available models: {list(MODELS.keys())}"
        )

    if not request.mensagem.strip():
        raise HTTPException(
            status_code=400,
            detail="Message cannot be empty."
        )

    model_data = MODELS[request.modelo]
    pipe = model_data["pipeline"]
    tokenizer = model_data["tokenizer"]
    architecture = model_data["architecture"]

    logger.info(
        f"[CHAT] Model='{request.modelo}' | Architecture='{architecture}' | Message='{request.mensagem[:50]}...'"
    )

    try:
        if architecture == "causal":
            prompt = f"Instruction: {request.mensagem}\nResponse:"

            result = pipe(
                prompt,
                max_new_tokens=request.max_tokens,
                temperature=request.temperatura,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
                num_return_sequences=1,
            )

            generated_text = result[0]["generated_text"]
            resposta = generated_text.split("Response:")[-1].strip()

        else:
            prompt = f"Answer the question: {request.mensagem}"

            result = pipe(
                prompt,
                max_new_tokens=request.max_tokens,
                temperature=request.temperatura,
                do_sample=True,
                top_p=0.9,
                num_return_sequences=1,
            )

            resposta = result[0]["generated_text"].strip()

        if not resposta:
            resposta = "[The model did not generate an answer. Try increasing max_tokens.]"

        tokens_gerados = len(tokenizer.encode(resposta))

        return ChatResponse(
            resposta=resposta,
            modelo=request.modelo,
            tokens_gerados=tokens_gerados
        )

    except Exception as error:
        logger.error(f"Generation error: {error}")

        raise HTTPException(
            status_code=500,
            detail=f"Error generating response: {str(error)}"
        )


@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "modelos_carregados": list(MODELS.keys()),
        "quantidade": len(MODELS)
    }


# =============================================================================
# FRONTEND
# =============================================================================

if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", response_class=HTMLResponse)
async def root():
    html_path = os.path.join(STATIC_DIR, "index.html")

    if not os.path.exists(html_path):
        return HTMLResponse(
            content="""
            <h1>LoRA Models API</h1>
            <p>API is running.</p>
            <p>Access <a href="/docs">/docs</a> to test the endpoints.</p>
            <p>Use <a href="/health">/health</a> to check loaded models.</p>
            """,
            status_code=200
        )

    with open(html_path, "r", encoding="utf-8") as file:
        return HTMLResponse(content=file.read())


# =============================================================================
# RUN
# =============================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )