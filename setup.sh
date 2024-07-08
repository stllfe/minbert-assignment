#!/usr/bin/env bash

set -e
conda create -n bert_hw python=3.10
conda activate bert_hw

conda install pytorch torchvision torchaudio -c pytorch
pip install uv &&
uv pip install tqdm \
requests \
importlib-metadata \
filelock \
scikit-learn \
tokenizers \
explainaboard_client
