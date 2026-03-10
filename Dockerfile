FROM python:3.11-slim

WORKDIR /app

# System deps (curl, ca-certificates, openssh-client) in case you want to debug,
# fetch configs, or use scp from the orchestrator container.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl openssh-client && \
    rm -rf /var/lib/apt/lists/*

# Copy project files into the image (config will be provided at runtime from ./volume)
COPY requirements.txt ./requirements.txt
COPY manage_runpod_training.py ./manage_runpod_training.py
COPY runpod ./runpod

# Install Python dependencies for the orchestration script
RUN pip install --no-cache-dir -r requirements.txt

ENV PYTHONUNBUFFERED=1

# Example usage (from host), using the pre-made volume folder:
#   docker build -t runpod-yolo-orchestrator .
#   docker run --rm \
#     -v "$PWD/volume:/app/volume" \
#     runpod-yolo-orchestrator \
#     --config volume/config.yaml \
#     --run-name my_first_run

ENTRYPOINT ["python", "manage_runpod_training.py"]

