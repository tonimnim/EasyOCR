# syntax=docker/dockerfile:1.7
# The screenshot reader service. Build from the repository root (the context needs easyocr/ and
# tonits_vision/):
#
#   docker build -f tonits_vision/Dockerfile -t tonits-vision .
#   docker run --rm -e VISION_TOKEN=... -p 8000:8000 --cpus 2 tonits-vision
#
# tonits_vision/Dockerfile.dockerignore keeps the context to what the image needs.
#
# The image carries its models: the CRAFT detector and BASE_RECOGNIZER are downloaded at build time, and
# a fine-tuned recognizer exported to tonits_vision/models (training/export_model.py) is copied in. It
# never downloads at runtime. RECOGNIZER picks the one it serves; the build fails if it can't load it.
#
#   docker build -f tonits_vision/Dockerfile --build-arg RECOGNIZER=efootball_v1 -t tonits-vision .

ARG PYTHON_IMAGE=python:3.14-slim

FROM ${PYTHON_IMAGE} AS easyocr-wheel
WORKDIR /src
COPY setup.py setup.cfg MANIFEST.in README.md requirements.txt ./
COPY easyocr ./easyocr
RUN pip wheel --no-deps --wheel-dir /wheels .

FROM ${PYTHON_IMAGE}

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore

# CPU-only torch (the "+cpu" builds exist only on the PyTorch index): the CUDA build is ~4 GB larger
# and the reader never uses a GPU. Its own layer, so changing another pin doesn't download it again;
# its dependencies are pinned in requirements-server.txt and installed next. The pip cache lives in a
# cache mount, outside the image.
ARG TORCH_VERSION=2.14.1+cpu
ARG TORCHVISION_VERSION=0.29.1+cpu
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-deps --index-url https://download.pytorch.org/whl/cpu \
        "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}"
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=tonits_vision/requirements-server.txt,target=/tmp/requirements-server.txt \
    pip install -r /tmp/requirements-server.txt

# easyocr from this repository (the context carries only the English word list; the reader reads ["en"]).
RUN --mount=type=bind,from=easyocr-wheel,source=/wheels,target=/wheels \
    pip install --no-deps /wheels/*.whl \
    && pip check

# Detector and base recognizer weights. Kept in a cache mount so rebuilding after a code change doesn't
# download them again.
ARG BASE_RECOGNIZER=english_g2
RUN --mount=type=cache,target=/var/cache/easyocr \
    mkdir -p /models \
    && python -c "import easyocr; easyocr.Reader(['en'], gpu=False, model_storage_directory='/var/cache/easyocr', recog_network='${BASE_RECOGNIZER}', download_enabled=True, verbose=False)" \
    && python -c "from easyocr.config import detection_models, recognition_models as r; import shutil; \
names = [detection_models['craft']['filename'], r['gen2']['${BASE_RECOGNIZER}']['filename']]; \
[shutil.copy('/var/cache/easyocr/' + n, '/models/' + n) for n in names]"

# The fine-tuned recognizer this image serves, when RECOGNIZER names one exported to tonits_vision/models
# (<name>.pth, .yaml, .py). Only its files are copied: the stock weights that folder may also hold are
# already in /models, verified at download.
ARG RECOGNIZER=english_g2
RUN --mount=type=bind,source=tonits_vision,target=/ctx \
    if [ -f "/ctx/models/${RECOGNIZER}.yaml" ]; then \
        cp "/ctx/models/${RECOGNIZER}.pth" "/ctx/models/${RECOGNIZER}.yaml" "/ctx/models/${RECOGNIZER}.py" /models/; \
    fi

# The service is the package's top-level modules; training, synth and tests stay out of the image.
WORKDIR /app
COPY tonits_vision/*.py ./tonits_vision/

ENV VISION_RECOGNIZER=${RECOGNIZER} \
    VISION_MODEL_DIR=/models \
    VISION_USER_NETWORK_DIR=/models \
    VISION_DOWNLOAD_MODELS=0 \
    VISION_PORT=8000 \
    HOME=/tmp

# Fail the build, not the deploy, when the recognizer can't load without a download.
RUN python -c "from tonits_vision.reader import ScreenshotReader; print(ScreenshotReader().model_info)"

# The root filesystem can be read-only: nothing is written outside /tmp.
USER 10001:10001
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=60s --retries=5 \
    CMD ["python", "-c", "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=3).status == 200 else 1)"]
CMD ["python", "-m", "tonits_vision.server"]
