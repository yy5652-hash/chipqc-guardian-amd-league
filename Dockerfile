# ChipQC Guardian — AMD AI League Match 3 (citation-exact RAG)
# Final stage MUST start from the mandated ROCm base (checked by layer identity).
FROM rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0

ENV PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TOKENIZERS_PARALLELISM=false \
    PIP_NO_CACHE_DIR=1 \
    MC3_MODELS_DIR=/models

WORKDIR /app

# Dependencies: --no-deps so the ROCm torch wheel is never replaced; verified right after.
COPY app/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --no-deps -r /app/requirements.txt \
 && python3 - <<'PY'
import sys, torch
v = torch.__version__
print("torch", v)
assert "+rocm" in v, f"torch is not the ROCm build: {v}"
import transformers, fitz, docx, openpyxl, PIL, numpy, regex, yaml, safetensors, tokenizers
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, AutoModel, AutoTokenizer
print("transformers", transformers.__version__, "pymupdf", fitz.__doc__)
PY

# Model weights shipped in the image (no network at evaluation). One layer per shard
# keeps every layer well under registry limits.
COPY models/Qwen3-Embedding-0.6B /models/Qwen3-Embedding-0.6B
COPY models/Qwen3-VL-8B-Instruct/config.json \
     models/Qwen3-VL-8B-Instruct/generation_config.json \
     models/Qwen3-VL-8B-Instruct/chat_template.json \
     models/Qwen3-VL-8B-Instruct/merges.txt \
     models/Qwen3-VL-8B-Instruct/vocab.json \
     models/Qwen3-VL-8B-Instruct/tokenizer.json \
     models/Qwen3-VL-8B-Instruct/tokenizer_config.json \
     models/Qwen3-VL-8B-Instruct/preprocessor_config.json \
     models/Qwen3-VL-8B-Instruct/video_preprocessor_config.json \
     models/Qwen3-VL-8B-Instruct/model.safetensors.index.json \
     /models/Qwen3-VL-8B-Instruct/
COPY models/Qwen3-VL-8B-Instruct/model-00001-of-00004.safetensors /models/Qwen3-VL-8B-Instruct/
COPY models/Qwen3-VL-8B-Instruct/model-00002-of-00004.safetensors /models/Qwen3-VL-8B-Instruct/
COPY models/Qwen3-VL-8B-Instruct/model-00003-of-00004.safetensors /models/Qwen3-VL-8B-Instruct/
COPY models/Qwen3-VL-8B-Instruct/model-00004-of-00004.safetensors /models/Qwen3-VL-8B-Instruct/

# Application code last so that code changes never invalidate the weight layers.
COPY app/ /app/
RUN chmod +x /app/supervise.sh /app/app.py /app/server.py \
 && mkdir -p /app/corpus /app/output /app/index \
 && python3 -c "import ast,sys; [ast.parse(open(f).read()) for f in ['/app/app.py','/app/server.py','/app/mc3/pipeline.py']]; print('syntax ok')"

# The harness execs into the running container; the CMD keeps the model server
# resident (it is started here so model load is charged to the startup budget).
CMD ["/bin/bash", "/app/supervise.sh"]
