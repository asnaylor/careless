FROM nvcr.io/nvidia/pytorch:26.01-py3

LABEL org.opencontainers.image.title="careless"
LABEL org.opencontainers.image.description="GPU-enabled careless crystallographic merging for NERSC Perlmutter"
LABEL org.opencontainers.image.source="https://github.com/rs-station/careless"
LABEL org.opencontainers.image.licenses="MIT"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg

WORKDIR /opt/careless

COPY pyproject.toml README.md LICENSE MANIFEST.in .coveragerc ./
COPY careless ./careless
COPY tests ./tests

# The dev extra supplies pytest and its plugins so the same GPU tests used
# during development can be run against the finished image.
RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install ".[dev]" && \
    mkdir -p tests/data/output && \
    careless --version && \
    python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}')"

# Runtime data should be bind-mounted here. Do not set an ENTRYPOINT so all
# careless.* console commands and normal debugging shells remain accessible.
WORKDIR /work

CMD ["careless", "--help"]
