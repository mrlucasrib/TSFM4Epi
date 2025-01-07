FROM nvidia/cuda:12.3.2-runtime-ubuntu22.04

ARG DEFAULT_USER=vscode

# Install dependencies required to install Poetry and clean up
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt update && apt-get --no-install-recommends install -y git \
    python3.10 \
    python3-virtualenv \
    sudo

RUN useradd -m -s /bin/bash $DEFAULT_USER

RUN echo $DEFAULT_USER ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$DEFAULT_USER \
    && chmod 0440 /etc/sudoers.d/$DEFAULT_USER \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspaces
