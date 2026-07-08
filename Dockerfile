FROM python:3.11-slim

WORKDIR /workspace
COPY . /workspace

ARG INSTALL_ANALYSIS_DEPS=false
RUN python -m pip install --no-cache-dir --upgrade pip && \
    python -m pip install --no-cache-dir nbformat pyyaml && \
    if [ "$INSTALL_ANALYSIS_DEPS" = "true" ] && [ -f requirements.txt ]; then \
      python -m pip install --no-cache-dir -r requirements.txt; \
    fi

CMD ["python", "scripts/validate_repository.py"]
