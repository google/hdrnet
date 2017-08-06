#!/bin/bash

cm=2

CUDA_VISIBLE_DEVICES=$1 python hdrnet/bin/train.py \
        --learning_rate 1e-4 \
        --batch_size 16 \
        --model_name HDRNetPointwiseNNGuide \
        --nobatch_norm \
        --output_resolution 512 512 \
        --channel_multiplier $cm \
        --data_dir data/style_transfer_1024/train/filelist.txt \
        --eval_data_dir data/style_transfer_1024/test/filelist.txt \
        --checkpoint_dir output/checkpoints/st_1024_nn_cm$cm
