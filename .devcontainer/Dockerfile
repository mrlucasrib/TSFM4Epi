FROM nvidia/cuda:12.3.2-runtime-ubuntu22.04

# Set environment variables to avoid interactive prompts and ensure Poetry installation works correctly
ENV POETRY_VERSION=1.8.4
ENV POETRY_HOME=/opt/poetry
ENV PATH="$POETRY_HOME/bin:$PATH"

ARG DEFAULT_USER=vscode

# Install dependencies required to install Poetry and clean up
RUN apt-get update && apt-get install -y \
    curl=7.81.0-1ubuntu1.19 \
    git=1:2.34.1-1ubuntu1.11 \
    python3.10-venv=3.10.12-1~22.04.7 \
    python3.10=3.10.12-1~22.04.7 \
    && curl -sSL https://install.python-poetry.org | python3 - \
    && apt-get remove -y curl \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
RUN poetry config virtualenvs.in-project true

RUN useradd -m -s /bin/bash $DEFAULT_USER

RUN apt-get update \
    && apt-get install -y sudo \
    && echo $DEFAULT_USER ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$DEFAULT_USER \
    && chmod 0440 /etc/sudoers.d/$DEFAULT_USER \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspaces
