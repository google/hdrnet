#!/bin/bash

cm=1

CUDA_VISIBLE_DEVICES=$1 python hdrnet/bin/train.py \
        --learning_rate 1e-4 \
        --batch_size 4 \
        --data_pipeline HDRpDataPipeline \
        --model_name HDRNetPointwiseNNGuide \
        --nobatch_norm \
        --output_resolution 256 256 \
        --channel_multiplier $cm \
        --data_dir data/hdrp/filelist.txt \
        --eval_data_dir data/hdrp/filelist.txt \
        --checkpoint_dir output/checkpoints/hdrp_256_nn_cm$cm
