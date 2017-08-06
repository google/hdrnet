#!/bin/bash

depth=$2
width=$3
CUDA_VISIBLE_DEVICES=$1 python hdrnet/bin/train.py \
        --learning_rate 1e-4 \
        --batch_size 1 \
        --model_name DilatedConvolutions \
        --nobatch_norm \
        --output_resolution 2048 2048 \
        --depth $depth \
        --width $width \
        --data_dir data/local_laplacian_hl_2048/train/filelist.txt \
        --eval_data_dir data/local_laplacian_hl_2048/test/filelist.txt \
        --checkpoint_dir output/checkpoints/ll_2048_dilated_d${depth}_w${width}
