FROM pytorch/pytorch:2.14.1-cuda12.6-cudnn9-runtime@sha256:1262abc856a428933c0477b4be87079173a0b087b7691c47fcb396b7b5775c56
LABEL org.opencontainers.image.title="DeepFilterNet CUDA development runtime" \
      org.opencontainers.image.source="https://github.com/juliancoy/DeepFilterNet"
RUN apt-get update && apt-get install -y --no-install-recommends libsndfile1 && rm -rf /var/lib/apt/lists/*
RUN python -m pip install --no-cache-dir torchaudio==2.14.1 --index-url https://download.pytorch.org/whl/cu126
RUN python -m pip install --no-cache-dir numpy==1.26.4 deepfilternet==0.5.6 soundfile==0.12.1 scipy
ENV PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 PYTHONPATH=/code/DeepFilterNet XDG_CACHE_HOME=/cache TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
WORKDIR /output
ENTRYPOINT ["python", "/code/docker/process.py"]
