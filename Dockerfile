# CC-0F4 PC1 - entorno minimo para el prototipo de riesgo de mercado.
# Adaptado del Dockerfile global del curso: mismas verificaciones por etapas,
# pero sin CUDA ni las librerias que la PC1 no usa (open_clip, faiss, peft, etc.).

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/home/user/.cache/huggingface

WORKDIR /workspace/pc1

COPY requirements.txt /tmp/requirements.txt


# 1. PyTorch (CPU): Qwen2.5-0.5B corre sin GPU

RUN python -m pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu


# 2. Dependencias de la PC1

RUN python -m pip install --no-cache-dir -r /tmp/requirements.txt


# 3. Verificar librerias principales

RUN python -c "import torch, transformers, jsonschema, pandas, sklearn, yfinance; \
print('torch=', torch.__version__); \
print('transformers=', transformers.__version__); \
print('jsonschema=', jsonschema.__version__); \
print('Import check: OK')"


# 4. Verificar consistencia de dependencias

RUN python -m pip check


# 5. Usuario no privilegiado

RUN useradd --create-home --uid 1000 user \
    && mkdir -p /workspace/pc1 /home/user/.cache/huggingface \
    && chown -R 1000:1000 /workspace/pc1 /home/user

USER 1000:1000

EXPOSE 8888

CMD ["jupyter", "lab", \
     "--ip=0.0.0.0", \
     "--port=8888", \
     "--no-browser", \
     "--ServerApp.root_dir=/workspace/pc1"]
