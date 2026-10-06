# Base: NGC vLLM container for aarch64 with CUDA 13.1, Python 3.12
FROM nvcr.io/nvidia/vllm:26.02-py3

ARG DEBIAN_FRONTEND=noninteractive

# Unset ROCR_VISIBLE_DEVICES to prevent conflict with CUDA_VISIBLE_DEVICES
# (cluster sets both, but verl requires only one)
ENV ROCR_VISIBLE_DEVICES=""
ENV NCCL_NET=Socket

# Install system dependencies (ADDED rsync HERE)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    patch \
    rsync \
    build-essential \
    libsndfile1 \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /tmp/requirements.txt

RUN pip install --no-cache-dir -r /tmp/requirements.txt && \
    rm -f /tmp/requirements.txt

# Default command
CMD ["/bin/bash"]
