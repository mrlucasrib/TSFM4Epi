#!/bin/bash
set -e
huggingface-cli download time-series-foundation-models/Lag-Llama lag-llama.ckpt --local-dir /lag-llama