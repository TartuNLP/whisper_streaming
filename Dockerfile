FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies and CUDA libraries
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    libsndfile1 \
    portaudio19-dev \
    python3-dev \
    ffmpeg \
    cuda-cudart-11-8 \
    cuda-compat-11-8 \
    cuda-libraries-11-8 \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch with CUDA support
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Set up CUDA library paths
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/cuda-11.8/lib64:${LD_LIBRARY_PATH}"
RUN echo 'export LD_LIBRARY_PATH=$(python3 -c "import os; import nvidia.cublas.lib; import nvidia.cudnn.lib; print(os.path.dirname(nvidia.cublas.lib.__file__) + \":\" + os.path.dirname(nvidia.cudnn.lib.__file__))")' >> /etc/bash.bashrc

WORKDIR /app

COPY requirements.txt .

RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p model

EXPOSE 8000

CMD ["/bin/bash", "-c", "export LD_LIBRARY_PATH=$(python3 -c 'import os; import nvidia.cublas.lib; import nvidia.cudnn.lib; print(os.path.dirname(nvidia.cublas.lib.__file__) + \":\" + os.path.dirname(nvidia.cudnn.lib.__file__))') && python3 web_server.py"]