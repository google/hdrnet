#!/bin/bash

cm=1
expert=C

CUDA_VISIBLE_DEVICES=$1 python hdrnet/bin/train.py \
        --learning_rate 1e-4 \
        --batch_size 16 \
        --model_name HDRNetPointwiseNNGuide \
        --nobatch_norm \
        --output_resolution 256 256 \
        --channel_multiplier $cm \
        --data_dir data/expert${expert}/train/filelist.txt \
        --eval_data_dir data/expert${expert}/test/filelist.txt \
        --checkpoint_dir output/checkpoints/expert${expert}_256_nn_cm$cm
