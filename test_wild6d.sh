#!/bin/bash

set -x
set -e

export CUDA_VISIBLE_DEVICES=3

# python evaluate_SPD_on_Wild6D.py --use_nocs_map --implicit
python test_wild6d.py --use_nocs_map --implicit --model '/mnt/HDD2/lj/Diff9D/log1/diffusion_pose' --select_class 'bottle' --result_dir '/mnt/HDD2/lj/Diff9D/Wild6D_results/sys/bottle'