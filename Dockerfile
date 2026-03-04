FROM nvcr.io/nvidia/pytorch:24.10-py3

ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    wget \
    vim \
    ca-certificates \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade --no-cache-dir \
    pip \
    setuptools \
    wheel

COPY requirements.txt .

RUN python -m pip install -r requirements.txt
RUN python -m pip install torch==2.10.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

WORKDIR /workspace

RUN mkdir -p /workspace/notebooks \
    && mkdir -p /workspace/data

RUN git clone https://github.com/psytechlab/kitoboy-ml.git
RUN git clone https://github.com/psytechlab/empathy_dataset_transfer.git
RUN jupyter lab --generate-config

RUN echo "c.ServerApp.password = 'sha1:2e471d7efc16:054bb90793eb5c1cdf64d427c4538a710469e2fd'" \
    >> /root/.jupyter/jupyter_server_config.py

EXPOSE 8888

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
