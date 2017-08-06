#!/bin/bash

chkpts=output/checkpoints
out=output/bench_new_16mp
input=benchmark/data/square_16mpix.png
size=2048

mkdir -p $out
mkdir -p $out/gpu
mkdir -p $out/cpu

for net in unet
do
  for d in 6 9 11
  do
    for w in 16 32 64
    do
      f=ll_${size}_${net}_d${d}_w${w}
      echo $f
      ./hdrnet/bin/freeze_graph.py $chkpts/$f
      ./scripts/optimize_graph.sh $chkpts/$f
      ./benchmark/bin/benchmark --output_directory $out/gpu --use_gpu --input_path $input --mode 1 --checkpoint_path $chkpts/$f
      ./benchmark/bin/benchmark --output_directory $out/cpu --input_path $input --mode 1 --checkpoint_path $chkpts/$f
    done
  done
done

# cm=1
# for l in 4 8 16
# do
#   for s in 8 16 32
#   do
#     f=ll_${size}_std_l${l}_s${s}_cm${cm}
#     echo $f
#     # ./hdrnet/bin/freeze_graph.py $chkpts/$f
#     # ./scripts/optimize_graph.sh $chkpts/$f
#     ./benchmark/bin/benchmark --output_directory $out/gpu --use_gpu --input_path $input --mode 0 --checkpoint_path $chkpts/$f --burn_iters 50 --iters 50
#     ./benchmark/bin/benchmark --output_directory $out/cpu --input_path $input --mode 0 --checkpoint_path $chkpts/$f --burn_iters 50 --iters 50
#   done
# done
