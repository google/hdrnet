#!/bin/bash

luma=8
spatial=16
cm=2

CUDA_VISIBLE_DEVICES=$1 python hdrnet/bin/train.py \
        --learning_rate 1e-4 \
        --batch_size 16 \
        --model_name StyleTransferCurves \
        --data_pipeline StyleTransferDataPipeline \
        --nobatch_norm \
        --output_resolution 256 256 \
        --luma_bins $luma \
        --spatial_bin $spatial \
        --channel_multiplier $cm \
        --data_dir data/style_transfer_n/train \
        --eval_data_dir data/style_transfer_n/test \
        --checkpoint_dir output/checkpoints/nst_256_std_l${luma}_s${spatial}_cm$cm
