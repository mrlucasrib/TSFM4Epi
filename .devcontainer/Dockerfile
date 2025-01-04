FROM nvidia/cuda:12.3.2-runtime-ubuntu22.04

# Set environment variables to avoid interactive prompts and ensure Poetry installation works correctly
# ENV VIRTUALENV_APP_DATA=$HOME/.cache/virtualenv
ENV PATH="$POETRY_HOME/bin:$PATH"

ARG DEFAULT_USER=vscode

# Install dependencies required to install Poetry and clean up
RUN apt-get update && apt-get install -y \
    curl \
    git=1:2.34.1-1ubuntu1.11 \
    python3-virtualenv=20.13.0+ds-2 \
    python3.10=3.10.12-1~22.04.7 \
    && apt-get remove -y curl \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -s /bin/bash $DEFAULT_USER

RUN apt-get update \
    && apt-get install -y sudo \
    && echo $DEFAULT_USER ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$DEFAULT_USER \
    && chmod 0440 /etc/sudoers.d/$DEFAULT_USER \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspaces
