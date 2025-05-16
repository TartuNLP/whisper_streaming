#!/bin/bash

source /raid/rauno/miniconda3/bin/activate
conda activate subs_env
export LD_LIBRARY_PATH=$(python3 -c "import os; import nvidia.cublas.lib; import nvidia.cudnn.lib; print(os.path.dirname(nvidia.cublas.lib.__file__) + \":\" + os.path.dirname(nvidia.cudnn.lib.__file__))")
export MKL_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=1
# python3 web_server.py
uvicorn web_server:app --host 0.0.0.0 --port 8000
