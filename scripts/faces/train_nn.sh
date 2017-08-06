#!/bin/bash

CUDA_VISIBLE_DEVICES=$1 python hdrnet/bin/train.py \
        --learning_rate 1e-4 \
        --batch_size 4 \
        --model_name HDRNetPointwiseNNGuide \
        --nobatch_norm \
        --output_resolution 256 256 \
        --luma_bins 8 \
        --spatial_bin 16 \
        --channel_multiplier 2 \
        --data_dir data/faces/train/filelist.txt \
        --eval_data_dir data/faces/test/filelist.txt \
        --checkpoint_dir output/checkpoints/faces_nn_cm2
