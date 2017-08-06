#!/bin/bash

blur=$2
sharpen=$3

CUDA_VISIBLE_DEVICES=$1 python hdrnet/bin/train.py \
        --learning_rate 1e-4 \
        --batch_size 1 \
        --model_name HDRNetCurves \
        --data_pipeline UnsharpMaskDataPipeline \
        --blur_sigma $blur \
        --sharpen $sharpen \
        --nobatch_norm \
        --output_resolution 2048 2048 \
        --data_dir data/local_laplacian_hl_2048/train/filelist.txt \
        --eval_data_dir data/local_laplacian_hl_2048/test/filelist.txt \
        --checkpoint_dir output/checkpoints/usm_std_radius${blur}_sharpen${sharpen}
