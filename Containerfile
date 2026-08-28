ARG NVIDIA_PYTORCH_IMAGE=nvcr.io/nvidia/pytorch:26.01-py3

# Install DIALS's Python runtime dependencies directly into NVIDIA's Python.
# This shared base is inherited by both the native builder and final image, so
# the final application never switches to a conda-managed Python or PyTorch.
FROM ${NVIDIA_PYTORCH_IMAGE} AS dials-runtime

ARG CCTBX_VERSION=2025.11
ARG RAY_VERSION=2.54.0

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg

RUN python -m pip install --no-cache-dir \
      "cctbx-base==${CCTBX_VERSION}" \
      "gemmi==0.7.5" \
      "h5py==3.16.0" \
      "hdf5plugin==7.0.0" \
      "jinja2==3.1.6" \
      "natsort==8.4.0" \
      "nxmx==0.0.7" \
      "optuna==4.9.0" \
      "ordered-set==4.1.0" \
      "pandas==2.3.3" \
      "pint==0.25.3" \
      "procrunner==2.3.3" \
      "pycbf==0.9.6.7" \
      "ray==${RAY_VERSION}" \
      "safetensors==0.7.0" \
      "scikit-image==0.26.0" \
      "scikit-learn==1.8.0" \
      "tabulate==0.9.0" \
      "tqdm==4.67.1"


# Compile only dxtbx and DIALS. cctbx itself comes from its CPython 3.12
# manylinux wheel; Boost headers are matched to the Boost 1.86 libraries
# bundled in that wheel so all Boost.Python extensions share one ABI.
FROM dials-runtime AS dials-builder

ARG BOOST_VERSION=1.86.0
ARG BOOST_VERSION_U=1_86_0
ARG DIALS_SHA=dda5bf8949d1d3d5f11afd105ed23bcb568c96ed
ARG DXTBX_SHA=f881b1dd531f5b53add3c49841d54b1368bf77bb

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install --yes --no-install-recommends \
      ca-certificates \
      curl \
      libeigen3-dev \
      libhdf5-dev \
      libmsgpack-cxx-dev && \
    rm -rf /var/lib/apt/lists/* && \
    python -m pip install --no-cache-dir "pybind11==3.0.1"

RUN curl -fsSL \
      "https://archives.boost.io/release/${BOOST_VERSION}/source/boost_${BOOST_VERSION_U}.tar.bz2" \
      | tar -xj -C /opt && \
    mkdir -p "/opt/boost_${BOOST_VERSION_U}/lib" && \
    ln -s \
      /usr/local/lib/python3.12/dist-packages/cctbx_base.libs/libboost_python312-14523df8.so.1.86.0 \
      "/opt/boost_${BOOST_VERSION_U}/lib/libboost_python312.so" && \
    ln -s \
      /usr/local/lib/python3.12/dist-packages/cctbx_base.libs/libboost_thread-36d5d84c.so.1.86.0 \
      "/opt/boost_${BOOST_VERSION_U}/lib/libboost_thread.so"

RUN mkdir -p /opt/src && \
    curl -fsSL "https://github.com/cctbx/dxtbx/archive/${DXTBX_SHA}.tar.gz" \
      | tar -xz -C /opt/src && \
    mv "/opt/src/dxtbx-${DXTBX_SHA}" /opt/src/dxtbx && \
    curl -fsSL "https://github.com/dials/dials/archive/${DIALS_SHA}.tar.gz" \
      | tar -xz -C /opt/src && \
    mv "/opt/src/dials-${DIALS_SHA}" /opt/src/dials

COPY container/generate_cctbx_headers.py /opt/generate_cctbx_headers.py
COPY container/boost-policy.cmake /opt/boost-policy.cmake

# The cctbx wheel contains the generators but omits headers needed to compile
# downstream extensions. It also retains its conda-build include prefix in
# libtbx_env, so make that metadata resolve to the wheel's real pip location.
RUN cctbx_build="$(python -c 'import libtbx.load_env; print(abs(libtbx.env.build_path))')" && \
    cctbx_prefix="$(python /opt/src/dxtbx/cmake/read_env.py \
      "${cctbx_build}/libtbx_env" --build-path "${cctbx_build}" --sys-prefix /usr \
      | python -c 'import json, sys; from pathlib import Path; print(Path(json.load(sys.stdin)["python_exe"]).parent.parent)')" && \
    mkdir -p "${cctbx_prefix}/lib/python3.12" && \
    ln -s /usr/local/lib/python3.12/dist-packages \
      "${cctbx_prefix}/lib/python3.12/site-packages" && \
    python /opt/generate_cctbx_headers.py

RUN cmake -S /opt/src/dxtbx -B /opt/build/dxtbx -GNinja \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=/usr \
      "-DCMAKE_INSTALL_RPATH=/usr/local/lib/python3.12/dist-packages;/usr/local/lib/python3.12/dist-packages/cctbx_base.libs" \
      -DCMAKE_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages \
      -DBOOST_ROOT="/opt/boost_${BOOST_VERSION_U}" \
      -DBoost_ROOT="/opt/boost_${BOOST_VERSION_U}" \
      -DBoost_INCLUDE_DIR="/opt/boost_${BOOST_VERSION_U}" \
      -DBoost_LIBRARY_DIR="/opt/boost_${BOOST_VERSION_U}/lib" \
      -DBoost_NO_BOOST_CMAKE=ON \
      -DBoost_NO_SYSTEM_PATHS=ON \
      -DCMAKE_PROJECT_INCLUDE_BEFORE=/opt/boost-policy.cmake \
      -DPython_EXECUTABLE=/usr/bin/python \
      -Dpybind11_DIR=/usr/local/lib/python3.12/dist-packages/pybind11/share/cmake/pybind11 && \
    cmake --build /opt/build/dxtbx --parallel 4 && \
    DESTDIR=/opt/dials-stage cmake --install /opt/build/dxtbx && \
    mkdir -p /opt/wheels && \
    python -m pip wheel --no-cache-dir --no-deps \
      --wheel-dir /opt/wheels /opt/src/dxtbx && \
    python -m pip install --no-cache-dir --no-deps \
      /opt/wheels/dxtbx-3.30.0-py3-none-any.whl

RUN cmake -S /opt/src/dials -B /opt/build/dials -GNinja \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=/usr \
      "-DCMAKE_INSTALL_RPATH=/usr/local/lib/python3.12/dist-packages;/usr/local/lib/python3.12/dist-packages/cctbx_base.libs" \
      -DCMAKE_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages \
      -DBOOST_ROOT="/opt/boost_${BOOST_VERSION_U}" \
      -DBoost_ROOT="/opt/boost_${BOOST_VERSION_U}" \
      -DBoost_INCLUDE_DIR="/opt/boost_${BOOST_VERSION_U}" \
      -DBoost_LIBRARY_DIR="/opt/boost_${BOOST_VERSION_U}/lib" \
      -DBoost_NO_BOOST_CMAKE=ON \
      -DBoost_NO_SYSTEM_PATHS=ON \
      -DCMAKE_PROJECT_INCLUDE_BEFORE=/opt/boost-policy.cmake \
      -DPython_EXECUTABLE=/usr/bin/python && \
    cmake --build /opt/build/dials --parallel 4 && \
    DESTDIR=/opt/dials-stage cmake --install /opt/build/dials && \
    python -m pip wheel --no-cache-dir --no-deps \
      --wheel-dir /opt/wheels /opt/src/dials && \
    python -m pip install --no-cache-dir --no-deps \
      /opt/wheels/dials-3.30-py3-none-any.whl

RUN PYTHONPATH=/opt/dials-stage/usr/local/lib/python3.12/dist-packages \
    python - <<'PY'
import torch
from dials.array_family import flex
from dxtbx.model.experiment_list import ExperimentListFactory

table = flex.reflection_table()
table["id"] = flex.int([0, 1])
assert list(table["id"]) == [0, 1]
assert ExperimentListFactory is not None
assert ".nv26.01." in torch.__version__
assert torch.version.cuda == "13.1"
PY


FROM dials-runtime AS careless

LABEL org.opencontainers.image.title="careless"
LABEL org.opencontainers.image.description="GPU-enabled careless crystallographic merging for NERSC Perlmutter"
LABEL org.opencontainers.image.source="https://github.com/rs-station/careless"
LABEL org.opencontainers.image.licenses="MIT"

COPY --from=dials-builder \
  /opt/dials-stage/usr/local/lib/python3.12/dist-packages/ \
  /usr/local/lib/python3.12/dist-packages/
COPY --from=dials-builder /opt/wheels/ /opt/dials-wheels/

RUN python -m pip install --no-cache-dir --no-deps \
      /opt/dials-wheels/dxtbx-3.30.0-py3-none-any.whl \
      /opt/dials-wheels/dials-3.30-py3-none-any.whl && \
    rm -rf /opt/dials-wheels

WORKDIR /opt/careless

COPY pyproject.toml README.md LICENSE MANIFEST.in .coveragerc ./
COPY careless ./careless
COPY tests ./tests
COPY container/nvidia-constraints.txt /opt/nvidia-constraints.txt

# The dev extra supplies pytest and its plugins so the same GPU tests used
# during development can be run against the finished image.
RUN python -m pip install --no-build-isolation \
      --constraint /opt/nvidia-constraints.txt ".[dev,distributed]" && \
    python -m pip check && \
    mkdir -p tests/data/output && \
    careless --version && \
    CARELESS_CONTAINER_TESTS=1 python -c \
      'from cctbx import sgtbx; from dials.array_family import flex; from dxtbx.model.experiment_list import ExperimentListFactory; import pytest; raise SystemExit(pytest.main(["-q", "-o", "addopts=", "tests/test_dials.py", "tests/io/test_prepared.py"]))'

# Runtime data should be bind-mounted here. Do not set an ENTRYPOINT so all
# careless.* console commands and normal debugging shells remain accessible.
WORKDIR /work

CMD ["careless", "--help"]
