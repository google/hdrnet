#!/usr/bin/env python
# encoding: utf-8

import matplotlib as mpl
mpl.use('Agg')

import argparse
import os
import re
import json

import matplotlib.pyplot as plt
import numpy as np

print plt.style.available
plt.style.use('seaborn-paper')

psnr = {
    "ll_2048_unet_d11_w16": 34.4,
    "ll_2048_unet_d11_w32": 34.6,
    "ll_2048_unet_d11_w64": 35.7,
    "ll_2048_unet_d3_w16": 22.6,
    "ll_2048_unet_d3_w32": 23.0,
    "ll_2048_unet_d3_w64": 24.2,
    "ll_2048_unet_d6_w16": 24.0,
    "ll_2048_unet_d6_w32": 24.1,
    "ll_2048_unet_d6_w64": 26.2,
    "ll_2048_unet_d9_w16": 32.5,
    "ll_2048_unet_d9_w32": 32.9,
    "ll_2048_unet_d9_w64": 32.7,

    "ll_2048_dilated_d11_w16": 0,
    "ll_2048_dilated_d11_w32": 0,
    "ll_2048_dilated_d11_w64": 0,
    "ll_2048_dilated_d3_w16": 22.3,
    "ll_2048_dilated_d3_w32": 24.3,
    "ll_2048_dilated_d3_w64": 24.5,
    "ll_2048_dilated_d6_w16": 23.5,
    "ll_2048_dilated_d6_w32": 24.2,
    "ll_2048_dilated_d6_w64": 24.5,
    "ll_2048_dilated_d9_w16": 0,
    "ll_2048_dilated_d9_w32": 0,
    "ll_2048_dilated_d9_w64": 0,

    "ll_2048_std_l16_s16_cm1": 32.2,
    "ll_2048_std_l16_s32_cm1": 32.7,
    "ll_2048_std_l16_s8_cm1": 32.3,
    "ll_2048_std_l4_s16_cm1": 31.0,
    "ll_2048_std_l4_s32_cm1": 31.2,
    "ll_2048_std_l4_s8_cm1": 30.8,
    "ll_2048_std_l8_s16_cm1": 31.8,
    "ll_2048_std_l8_s32_cm1": 32.5,
    "ll_2048_std_l8_s8_cm1": 31.7,
}

ref_ll_runtime = 383584/1000.0 

def make_array(l):
  arr = np.zeros((len(l), 5))
  for i, li in enumerate(l):
    arr[i, 0] = li['downsampling']
    arr[i, 1] = li['convert_to_float']
    arr[i, 2] = li['forward_pass']
    arr[i, 3] = li['rendering']
    arr[i, 4] = li['psnr']
  return arr


def main(args):
  unets = []
  dilateds = []
  ours = []
  e = re.compile(r".*\.json")
  for f in os.listdir(args.data_dir):
    if not e.match(f):
      continue
    name = os.path.splitext(f)[0]
    with open(os.path.join(args.data_dir, f)) as fid:
      bench = json.load(fid)
    bench['psnr'] = psnr[name]
    bench['name'] = name
    if "unet" in name:
      unets.append(bench)
    elif "dilated" in name:
      dilateds.append(bench)
    elif "std" in name:
      ours.append(bench)

  unets_arr = make_array(unets)
  dilateds_arr = make_array(dilateds)
  ours_arr = make_array(ours)

  ax = plt.subplot(1,1,1)
  ax.set_xscale("log")
  ax.scatter(unets_arr[:, 2], unets_arr[:, 4], label='U-Net (CPU)')
  ax.scatter(np.sum(ours_arr[:, 0:4], axis=1), ours_arr[:, 4], label='Ours (CPU)')
  ax.scatter(dilateds_arr[:, 2], dilateds_arr[:, 4], label='Dilated (CPU)')
  ax.axvline(x=ref_ll_runtime, color='r', linestyle='--')
  ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())

  exp = re.compile("ll_2048_unet_d(\d+)_w(\d+)")
  for i in range(len(unets)):
    n = unets[i]["name"]
    match = exp.match(n)
    d = match.group(1)
    w = match.group(2)
    plt.annotate("d{}w{}".format(d, w), (unets_arr[i, 2], unets_arr[i, 4]+0.5))
  exp = re.compile("ll_2048_dilated_d(\d+)_w(\d+)")
  for i in range(len(dilateds)):
    n = dilateds[i]["name"]
    match = exp.match(n)
    d = match.group(1)
    w = match.group(2)
    plt.annotate("d{}w{}".format(d, w), (dilateds_arr[i, 2], dilateds_arr[i, 4]+0.5))

  # for i in range(len(ours)):
  #   plt.annotate(ours[i]["name"], (np.sum(ours_arr[i, 0:4]), ours_arr[i, 4]))
  plt.annotate("Ours (CPU)", (np.mean(np.sum(ours_arr[:, 0:4], axis=1))-5, np.amax(ours_arr[:, 4])+1))
  plt.annotate("Reference filter (CPU)" , (ref_ll_runtime-30, 21), color='r', horizontalalignment="right")

  plt.xlabel('running time (ms)')
  plt.ylabel('PSNR (dB)')
  plt.ylim([20,38])
  plt.title('Performance comparison on the Local Laplacian filter (4 megapixels)')
  # plt.legend()

  plt.savefig("extra_figures/output/comparison_to_unet.pdf")

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('data_dir')
  args = parser.parse_args()

  main(args)
  
