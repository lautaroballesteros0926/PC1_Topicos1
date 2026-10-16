# CC-0F4 PC2 - entorno minimo: consulta estructurada (Qwen) + recuperacion densa (e5 + FAISS).
# Mismo patron que el Dockerfile de la PC1; agrega sentence-transformers y faiss-cpu.

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/home/user/.cache/huggingface

WORKDIR /workspace/pc2

COPY requirements.txt /tmp/requirements.txt


# 1. PyTorch (CPU): Qwen2.5-0.5B y multilingual-e5-small corren sin GPU

RUN python -m pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu


# 2. Dependencias de la PC2

RUN python -m pip install --no-cache-dir -r /tmp/requirements.txt


# 3. Verificar librerias principales

RUN python -c "import torch, transformers, jsonschema, pandas, sklearn, sentence_transformers, faiss; \
print('torch=', torch.__version__); \
print('transformers=', transformers.__version__); \
print('sentence-transformers=', sentence_transformers.__version__); \
print('faiss=', faiss.__version__); \
print('Import check: OK')"


# 4. Verificar consistencia de dependencias

RUN python -m pip check


# 5. Usuario no privilegiado

RUN useradd --create-home --uid 1000 user \
    && mkdir -p /workspace/pc2 /home/user/.cache/huggingface \
    && chown -R 1000:1000 /workspace/pc2 /home/user

USER 1000:1000

EXPOSE 8888

CMD ["jupyter", "lab", \
     "--ip=0.0.0.0", \
     "--port=8888", \
     "--no-browser", \
     "--ServerApp.root_dir=/workspace/pc2"]
