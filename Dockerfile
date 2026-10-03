FROM python:3.12-slim-trixie

ARG NEMO_SPEECH_VERSION=0.2.0
# vulkan runs on CPU and on any GPU with a Vulkan driver. cuda needs an NVIDIA driver.
ARG NEMO_BACKEND=vulkan

RUN apt-get update \
  && apt-get install -y --no-install-recommends ca-certificates curl libvulkan1 mesa-vulkan-drivers \
  && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /opt/nemo-speech \
  && curl -fsSL "https://github.com/NVIDIA/NeMo-Speech.cpp/releases/download/v${NEMO_SPEECH_VERSION}/nemo-speech-${NEMO_SPEECH_VERSION}-linux-x86_64-${NEMO_BACKEND}.tar.gz" \
  | tar -xz -C /opt/nemo-speech --strip-components=1

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:/opt/nemo-speech/bin:${PATH}" \
    NEMO_SPEECH_MODEL_DIR=/models

ENTRYPOINT ["wyoming-nemo-speech"]
