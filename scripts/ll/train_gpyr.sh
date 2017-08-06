#!/bin/bash

luma=8
spatial=16
cm=1

CUDA_VISIBLE_DEVICES=$1 python hdrnet/bin/train.py \
        --learning_rate 1e-4 \
        --batch_size 1 \
        --model_name HDRNetGaussianPyr \
        --nobatch_norm \
        --output_resolution 2048 2048 \
        --luma_bins $luma \
        --spatial_bin $spatial \
        --channel_multiplier $cm \
        --data_dir data/local_laplacian_hl_2048/train/filelist.txt \
        --eval_data_dir data/local_laplacian_hl_2048/test/filelist.txt \
        --checkpoint_dir output/checkpoints/ll_2048_gpyr__l${luma}_s${spatial}_cm$cm
