# CC-0F4 Examen parcial - entorno: recuperacion (e5 + FAISS + BM25 + Cross-Encoder), RAG con Qwen,
# herramienta con contratos y anexo DeepSeek. Mismo patron que los Dockerfile de la PC1 y la PC2.

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/home/user/.cache/huggingface

WORKDIR /workspace/parcial

COPY requirements.txt /tmp/requirements.txt


# 1. PyTorch (CPU): Qwen2.5-0.5B, e5-small y el Cross-Encoder corren sin GPU

RUN python -m pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu


# 2. Dependencias del parcial

RUN python -m pip install --no-cache-dir -r /tmp/requirements.txt


# 3. Verificar librerias principales

RUN python -c "import torch, transformers, jsonschema, pandas, sklearn, sentence_transformers, faiss, rank_bm25, pydantic, httpx; \
print('torch=', torch.__version__); \
print('transformers=', transformers.__version__); \
print('sentence-transformers=', sentence_transformers.__version__); \
print('faiss=', faiss.__version__); \
print('pydantic=', pydantic.__version__); \
print('Import check: OK')"


# 4. Verificar consistencia de dependencias

RUN python -m pip check


# 5. Usuario no privilegiado

RUN useradd --create-home --uid 1000 user \
    && mkdir -p /workspace/parcial /home/user/.cache/huggingface \
    && chown -R 1000:1000 /workspace/parcial /home/user

USER 1000:1000

EXPOSE 8888

CMD ["jupyter", "lab", \
     "--ip=0.0.0.0", \
     "--port=8888", \
     "--no-browser", \
     "--ServerApp.root_dir=/workspace/parcial"]
