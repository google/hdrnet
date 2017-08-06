# Deep Bilateral Learning for Real-Time Image Enhancements
Siggraph 2017

Visit our [Project Page](https://groups.csail.mit.edu/graphics/hdrnet/).

[Michael Gharbi](https://mgharbi.com)
Jiawen Chen
Jonathan T. Barron
Samuel W. Hasinoff
Fredo Durand

Maintained by Michael Gharbi (<gharbi@mit.edu>)

Tested on Python 2.7, Ubuntu 14.04.

## Disclaimer

This is not an official Google product.

## Setup

### Dependencies

To install the Python dependencies, run:

    cd hdrnet
    pip install -r requirements.txt

### Build

Our network requires a custom Tensorflow operator to "slice" in the bilateral grid.
To build it, run:

    cd hdrnet
    make

To build the benchmarking code, run:

    cd benchmark
    make

Note that the benchmarking code requires a frozen and optimized model. Use
`hdrnet/bin/scripts/optimize_graph.py` and `hdrnet/bin/freeze.py to produce these`.

To build the Android demo, see dedicated section below.

### Test

Run the test suite to make sure the BilateralSlice operator works correctly:

    cd hdrnet
    py.test test

### Download pretrained models

We provide a set of pretrained models. One of these is included in the repo
(see `pretrained_models/local_laplacian_sample`). To download the rest of them
run:

    cd pretrained_models
    ./download.py

## Usage

To train a model, run the following command:

    ./hdrnet/bin/train.py <checkpoint_dir> <path/to_training_data/filelist.txt>

Look at `sample_data/identity/` for a typical structure of the training data folder.

You can monitor the training process using Tensorboard:

    tensorboard --logdir <checkpoint_dir>

To run a trained model on a novel image (or set of images), use:

    ./hdrnet/bin/run.py <checkpoint_dir> <path/to_eval_data> <output_dir>

To prepare a model for use on mobile, freeze the graph, and optimize the network:

    ./hdrnet/bin/freeze_graph.py <checkpoint_dir>
    ./hdrnet/bin/scripts/optimize_graph.sh <checkpoint_dir>

You will need to change the `${TF_BASE}` environment variable in `./hdrnet/bin/scripts/optimize_graph.sh`
and compile the necessary tensorflow command line tools for this (automated in the script).


## Android prototype

We will add it to this repo soon.

### Build

Make sure to use Android-NDK 12b. Version 13 and above cause known issues with Tensorflow for Android.
We tested on Android SDK 25.0.1.

Change the path to tensorflow, android-sdk and android-ndk in `WORKSPACE` to match
your environment.

Change the `target_features` and `generators_args` parameters in `android/BUILD` to
match your device/emulator.

Then build:

    bazel build //android:hdrnet_demo

Plug your smartphone in, or launch an emulator, then install the package on device:

    bazel mobile-install //android:hdrnet_demo

To add a new operator, freeze and optimize a TF model:

    ./hdrnet/bin/freeze_graph.py <checkpoint_dir>
    ./hdrnet/bin/scripts/optimize_graph.sh <checkpoint_dir>

Then add a folder in `android/assets` containing the `optimized_graph.pb` file. Also add the `*.bin` weights 
for the guidemap.

Finally, add the name of this folder to the `filters_array` in `android/res/values/arrays.xml`.

The rendering shader that implements the slice and apply operation is in `android/assets/camera_preview.frag`. 

* Camera2 API yields YUV images, we convert to RGB to feed the TF input in
`android/jni/convert_image_hl.cxx`. The processing could be sped up by training a model on YUV images directly.

* OpenGL float textures are in the range [0,1] so we normalize the affine weights.
(see `android/jni/convert_output_hl.cxx`). The shader needs to undo this normalization.

## Known issues and limitations

* The BilateralSliceApply operation is GPU only at this point. We do not plan to release a CPU implementation.
* The provided pre-trained models were updated from an older version and might slightly differ from the models
  used for evaluation in the paper.
* The HDR+ pretrained model has a different input format (16-bits linear, custom YUV). It will produce
uncanny colors if run on standard RGB images. We will release and updated version.
