#!/bin/bash

model=$1
# cp -R output/checkpoints/clean output/upgraded/$model
./hdrnet/bin/upgrade.py --src output/chkpts_siggraph2016_submission/$model --dst output/upgraded/$model
# ./hdrnet/bin/run.py --checkpoint_dir output/upgraded/$model \
#   --input data/overfit/train/filelist.txt --output output/runs/upgraded/_jiawen/$model
