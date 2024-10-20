#!/bin/bash
source ~/.bashrc
export PYTHONPATH=/mnt/e9c13c7c-f8d4-48da-ab7e-f214096b7777/zhangyang/sft_practice:$PYTHONPATH
deepspeed main.py --deepspeed src/configs/ds_s3.json
