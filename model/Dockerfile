# Training image for the gaze model.
FROM ghcr.io/astral-sh/uv:python3.14-bookworm-slim

# MediaPipe needs OpenGL
RUN apt-get update && apt-get install -y --no-install-recommends \
        libegl1 libgles2 libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies before source, so a code edit reuses the cached install layer.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project

# Get landmarker weights
ADD https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task \
    /app/models/landmarker.task
RUN chmod 644 /app/models/landmarker.task

COPY src/ ./src/
COPY train.py ./

# To enable GPU on the Docker container.
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

ENV PYTHONUNBUFFERED=1

VOLUME ["/data", "/app/checkpoints"]

ENTRYPOINT ["uv", "run", "train.py"]
CMD ["--help"]
