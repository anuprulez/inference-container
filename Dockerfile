ARG BASE_IMAGE=python:3.11-slim-bookworm
FROM ${BASE_IMAGE}

ARG ACCELERATOR=cpu
ARG INSTALL_TENSORFLOW=1

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models/.cache/huggingface \
    TORCH_HOME=/models/.cache/torch

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl git libgomp1 libgl1 libglib2.0-0 python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/inference
COPY requirements.txt install-runtime.sh ./
RUN chmod +x install-runtime.sh && ./install-runtime.sh "${ACCELERATOR}" "${INSTALL_TENSORFLOW}"

COPY inspect_model.py healthcheck.py ./
RUN mkdir -p /models /data && chmod -R a+rwX /models /data

VOLUME ["/models", "/data"]
ENTRYPOINT ["python3", "/opt/inference/inspect_model.py"]
CMD ["/models"]

