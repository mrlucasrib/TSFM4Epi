#!/bin/bash
set -e
python3 -m pip install -U "huggingface_hub[cli]==0.27.1"
mkdir /models
mkdir /models/timesfm-1.0
mkdir /models/timesfm-2.0
huggingface-cli download --repo-type model --cache-dir /root/.cache/huggingface --local-dir /models/timesfm-1.0 google/timesfm-1.0-200m | tail -n 1 > /etc/timesfm-1.0-200m
huggingface-cli download --repo-type model --cache-dir /root/.cache/huggingface --local-dir /models/timesfm-2.0 google/timesfm-2.0-500m-jax | tail -n 1 > /etc/timesfm-2.0-500m-jax