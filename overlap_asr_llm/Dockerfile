FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src

WORKDIR /app

# ffmpeg is needed by Whisper/audio tooling; libsndfile supports soundfile.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ffmpeg \
        git \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt pyproject.toml ./
COPY src ./src

# Set INSTALL_FULL_DEPS=0 for a smaller image that can run unit tests and mock
# workflows, or keep the default for real ASR/audio experiments.
ARG INSTALL_FULL_DEPS=1
RUN if [ "$INSTALL_FULL_DEPS" = "1" ]; then \
        python -m pip install -r requirements.txt; \
    fi \
    && python -m pip install -e .

COPY configs ./configs
COPY data ./data
COPY docs ./docs
COPY outputs ./outputs
COPY scripts ./scripts
COPY tests ./tests
COPY CONTRIBUTIONS.md REPOSITORY.md ./

EXPOSE 7861

CMD ["python", "-m", "unittest", "discover", "-s", "tests", "-q"]
