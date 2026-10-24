# Photoreal job server: gsplat (CUDA) + splat-transform + the job API.
# Build from the repository root:
#   docker build -f apps/pipeline/Dockerfile -t model-anything-pipeline .
# Runs the same locally (Docker Desktop + WSL2 + NVIDIA driver) and on a cloud GPU.
FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    # Ampere (A100, RTX 30xx), Ada (L4, RTX 40xx), Hopper (H100): compiled ahead, no GPU needed to build
    TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"

RUN apt-get update \
 && apt-get install -y --no-install-recommends git curl ca-certificates build-essential libgl1 libglib2.0-0 \
 && curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
 && apt-get install -y --no-install-recommends nodejs \
 && rm -rf /var/lib/apt/lists/*

# gsplat from its tagged source, so the CUDA kernels and examples/simple_trainer.py match.
ARG GSPLAT_VERSION=1.5.3
RUN git clone --depth 1 --branch v${GSPLAT_VERSION} https://github.com/nerfstudio-project/gsplat.git /opt/gsplat \
 && pip install --no-build-isolation /opt/gsplat \
 && pip install --no-build-isolation -r /opt/gsplat/examples/requirements.txt

ARG SPLAT_TRANSFORM_VERSION=3.10.1
RUN npm install -g @playcanvas/splat-transform@${SPLAT_TRANSFORM_VERSION}

COPY apps/pipeline/requirements.txt /app/apps/pipeline/requirements.txt
RUN pip install -r /app/apps/pipeline/requirements.txt
COPY .agents/skills/scan-to-threejs/scripts /app/.agents/skills/scan-to-threejs/scripts
COPY apps/pipeline /app/apps/pipeline
RUN python /app/apps/pipeline/patch_trainer.py /opt/gsplat/examples

WORKDIR /app/apps/pipeline
ENV GSPLAT_DIR=/opt/gsplat \
    MA_JOBS_DIR=/data/jobs \
    MA_VIEWER_DIR=/app/apps/room-viewer/dist
EXPOSE 8787
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8787"]
