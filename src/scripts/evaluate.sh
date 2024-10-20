#!/bin/bash
source ~/.bashrc
export PYTHONPATH=/mnt/e9c13c7c-f8d4-48da-ab7e-f214096b7777/zhangyang/sft_practice:$PYTHONPATH
python3 evaluate.py --model_path ./outputs/model/DSzero3 --dataset_path ./outputs/dataset --output_path ./outputs/evaluation
