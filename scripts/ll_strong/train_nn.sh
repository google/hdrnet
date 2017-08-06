#!/bin/bash

CUDA_VISIBLE_DEVICES=$1 python hdrnet/bin/train.py \
        --learning_rate 1e-4 \
        --batch_size 4 \
        --model_name HDRNetPointwiseNNGuide \
        --nobatch_norm \
        --output_resolution 1024 1024 \
        --luma_bins 8 \
        --spatial_bin 16 \
        --channel_multiplier 1 \
        --data_dir data/local_laplacian_2048/train/filelist.txt \
        --eval_data_dir data/local_laplacian_2048/test/filelist.txt \
        --checkpoint_dir output/checkpoints/ll_strong_1024_nn
