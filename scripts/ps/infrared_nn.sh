#!/bin/bash

action=infrared

CUDA_VISIBLE_DEVICES=$1 python hdrnet/bin/train.py \
        --learning_rate 1e-4 \
        --batch_size 4 \
        --model_name HDRNetPointwiseNNGuide \
        --nobatch_norm \
        --output_resolution 512 512 \
        --luma_bins 8 \
        --spatial_bin 16 \
        --channel_multiplier 1 \
        --data_dir data/photoshop_actions/${action}_1024/train/filelist.txt \
        --eval_data_dir data/photoshop_actions/${action}_1024/test/filelist.txt \
        --checkpoint_dir output/checkpoints/ps_${action}_1024
