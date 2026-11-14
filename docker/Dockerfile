# syntax=docker/dockerfile:1
FROM pytorch/pytorch:2.14.1-cuda12.6-cudnn9-runtime@sha256:1262abc856a428933c0477b4be87079173a0b087b7691c47fcb396b7b5775c56 AS runtime
LABEL org.opencontainers.image.title="DeepFilterNet CUDA development runtime" \
      org.opencontainers.image.source="https://github.com/juliancoy/DeepFilterNet"
RUN apt-get update && apt-get install -y --no-install-recommends libsndfile1 && rm -rf /var/lib/apt/lists/*
RUN python -c "import sys,torch; print('BASE_PYTHON',sys.version); print('BASE_TORCH',torch.__version__,torch.__file__); print('BASE_CUDA',torch.version.cuda)"
RUN python -m pip install --no-cache-dir --target /tmp/venv-bootstrap virtualenv==20.35.4 && PYTHONPATH=/tmp/venv-bootstrap python -m virtualenv --system-site-packages /opt/global && rm -rf /tmp/venv-bootstrap
ENV VIRTUAL_ENV=/opt/global
ENV PATH="/opt/global/bin:$PATH"
RUN printf '\nsource /opt/global/bin/activate\n' >> /etc/bash.bashrc
RUN python -c "import sys,torch; assert sys.prefix == '/opt/global'; print('GLOBAL_TORCH',torch.__version__)"
RUN python -m pip install --no-cache-dir numpy==1.26.4 appdirs==1.4.4 loguru packaging==23.2 requests scipy soundfile==0.12.1

FROM runtime AS native-builder
RUN apt-get update && apt-get install -y --no-install-recommends cargo rustc build-essential pkg-config && rm -rf /var/lib/apt/lists/*
RUN python -m pip install --no-cache-dir maturin==1.4.0
FROM native-builder AS wheel-builder
RUN --mount=type=bind,source=.,target=/src,readonly \
    --mount=type=cache,target=/root/.cargo/registry \
    --mount=type=cache,target=/root/.cargo/git \
    mkdir -p /tmp/df-source && cp -a /src/libDF /src/pyDF /tmp/df-source/ && \
    cp /src/docker/native/Cargo.toml /src/docker/native/Cargo.lock /tmp/df-source/ && \
    maturin build --release --locked --manifest-path /tmp/df-source/pyDF/Cargo.toml \
      --interpreter python --target-dir /tmp/df-target --out /wheels

FROM runtime
COPY --from=wheel-builder /wheels/ /tmp/df-wheels/
RUN python -m pip install --no-cache-dir --no-deps /tmp/df-wheels/*.whl && python -m pip install --no-cache-dir packaging==25.0 && rm -rf /tmp/df-wheels
ENV PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 PYTHONPATH=/code/DeepFilterNet XDG_CACHE_HOME=/cache TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
WORKDIR /output
ENTRYPOINT ["python", "/code/docker/process.py"]
