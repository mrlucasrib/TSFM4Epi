FROM nvidia/cuda:12.3.2-runtime-ubuntu22.04

WORKDIR /workspaces

# Install dependencies required
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get --no-install-recommends install -y git \
    python3.10 python3-pip

ARG EXPERIMENT_PATH
ENV EXPERIMENT_PATH_ENV=$EXPERIMENT_PATH
COPY $EXPERIMENT_PATH/requirements.txt $EXPERIMENT_PATH/setup.s[h] /workspaces/$EXPERIMENT_PATH/

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip==24.3.1 && \
    pip install -r /workspaces/$EXPERIMENT_PATH/requirements.txt

RUN --mount=type=cache,target=/root/.cache/huggingface \ 
    if [ -f /workspaces/$EXPERIMENT_PATH/setup.sh ]; then \
        chmod +x /workspaces/$EXPERIMENT_PATH/setup.sh && \
        /workspaces/$EXPERIMENT_PATH/setup.sh; \
    fi
    
COPY utils_exp/ /workspaces/utils_exp

COPY $EXPERIMENT_PATH /workspaces/$EXPERIMENT_PATH

COPY entrypoint.sh /entrypoint.sh

ENTRYPOINT ["sh", "/entrypoint.sh"]