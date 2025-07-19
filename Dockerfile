# Use NVIDIA DeepStream base image (development container with build tools)
FROM nvcr.io/nvidia/deepstream:7.1-gc-triton-devel

# Set working directory
WORKDIR /workspace

# Update system packages and install dependencies
RUN apt-get update -q && \
    apt-get install -y -q \
    python3-pip \
    python3-dev \
    python3-gi \
    python3-gi-cairo \
    gir1.2-gstreamer-1.0 \
    build-essential \
    git \
    curl \
    wget \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    pkg-config && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install fastapi>=0.110.0 uvicorn>=0.27.1 python-multipart>=0.0.9 websockets>=11.0.0 && \
    pip3 install opencv-python>=4.11.0.86 numpy>=1.24.0 pillow>=10.1.0 && \
    pip3 install torch>=2.0.0 torchvision>=0.15.0 torchaudio --index-url https://download.pytorch.org/whl/cu121 && \
    pip3 install ultralytics>=8.0.0 transformers==4.44.0 sentence-transformers>=2.2.2 huggingface_hub==0.24.0 && \
    pip3 install deep-sort-realtime>=1.3.2 filterpy scikit-learn scipy matplotlib lap && \
    pip3 install python-dotenv>=1.0.0 groq>=0.8.0 pydantic>=2.0.0 tqdm>=4.66.1 && \
    pip3 install faiss-cpu>=1.7.4 spacy>=3.7.2 && \
    pip3 install segmentation-models-pytorch>=0.3.0 scikit-image>=0.21.0 && \
    pip3 install httpx>=0.28.1 pymongo>=4.0.0 && \
    pip3 install pygobject>=3.42.0

# Copy the application code
COPY . .

# Create necessary directories
RUN mkdir -p frames uploads outputs inputs logs

# Set environment variables
ENV PYTHONPATH=/workspace:$PYTHONPATH

# Expose the port
EXPOSE 8000

# Start the application
CMD ["python3", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"] 