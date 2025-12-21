# rag_engine.py
# Core LLM and embedding engine used by the RAG pipeline.

import pandas as pd
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

# Load the LLM (FLAN-T5 small)
model_name = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
llm = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Load embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

def embed_text(text: str):
    """Generate embeddings using MiniLM."""
    return embedder.encode([text])[0]

def run_llm(prompt: str) -> str:
    """Run FLAN-T5 on the prompt."""
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = llm.generate(**inputs, max_new_tokens=150)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
