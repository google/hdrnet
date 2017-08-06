// Copyright 2016 Google Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

// -- OPS REGISTRAION ---------------------------------------------------------
REGISTER_OP("BilateralSlice")
  .Input("in: float")
  .Input("guide: float")
  .Output("out: float")
  .Doc(R"doc(
Slices input in in the location defined by guide, to produce output.
)doc");

REGISTER_OP("BilateralSliceGrad")
  .Input("in: float")
  .Input("guide: float")
  .Input("backprop: float")
  .Output("grid_grad: float")
  .Output("guide_grad: float");

REGISTER_OP("BilateralSliceApply")
  .Input("grid: float")
  .Input("guide: float")
  .Input("input: float")
  .Attr("has_offset: bool")
  .Output("out: float")
  .Doc(R"doc(
Slices input in in the location defined by guide and apply it, to produce output.
)doc");

REGISTER_OP("BilateralSliceApplyGrad")
  .Input("grid: float")
  .Input("guide: float")
  .Input("input: float")
  .Input("backprop: float")
  .Attr("has_offset: bool")
  .Output("grid_grad: float")
  .Output("guide_grad: float")
  .Output("input_grad: float");
// ----------------------------------------------------------------------------

// -- KERNEL LAUNCHERS --------------------------------------------------------
bool BilateralSliceKernelLauncher(
    const GPUDevice& d,
    int bs, int gh, int gw, int gd, int chans,
    int h, int w,
    const float* const grid, const float* const guide, float* const out);

bool BilateralSliceGradKernelLauncher(
    const GPUDevice& d,
    const float* const grid, const int64* grid_size,
    const float* const guide, const int64* guide_size,
    const float* const backprop,
    float* const grid_grad, float* const guide_grad);

bool BilateralSliceApplyKernelLauncher(
    const GPUDevice& d,
    int bs, int gh, int gw, int gd, 
    int input_chans, int output_chans, bool has_offset,
    int h, int w,
    const float* const grid, const float* const guide, const float* const input,
    float* const out);

bool BilateralSliceApplyGradKernelLauncher(
    const GPUDevice& d,
    const float* const grid, const int64* grid_size,
    const float* const guide, const int64* guide_size,
    const float* const input, const int64* input_size,
    const float* const backprop,
    bool has_offset,
    float* const grid_grad, float* const guide_grad, float* const input_grad);
// ----------------------------------------------------------------------------


// ----------------------------------------------------------------------------
class BilateralSliceOp : public OpKernel {
 public:
  explicit BilateralSliceOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the inputs
    const Tensor& bilateral_grid = context->input(0);
    const Tensor& guide = context->input(1);

    OP_REQUIRES(
        context, bilateral_grid.dims() == 5,
        errors::InvalidArgument(
        R"msg(Input grid should be 5D (batch, height, width, depth, nchannels))msg"));
    OP_REQUIRES(
        context, guide.dims() == 3,
        errors::InvalidArgument(
        R"msg(Guide image should be 3D (batch, height, width))msg"));

    // Get shape of output tensor
    TensorShape shape;
    shape.AddDim(guide.dim_size(0));  // Batch size
    shape.AddDim(guide.dim_size(1));  // height
    shape.AddDim(guide.dim_size(2));  // width
    shape.AddDim(bilateral_grid.dim_size(4));  // channels

    // Allocate output tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, shape, &output_tensor));

    auto output = output_tensor->flat<float>();

    const int64 *grid_size = bilateral_grid.shape().dim_sizes().data();
    const int64 *guide_size = guide.shape().dim_sizes().data();

    int h = guide.dim_size(1);
    int w = guide.dim_size(2);
    int bs = bilateral_grid.dim_size(0);
    int gh = bilateral_grid.dim_size(1);
    int gw = bilateral_grid.dim_size(2);
    int gd = bilateral_grid.dim_size(3);
    int chans = bilateral_grid.dim_size(4);

    // Call the cuda kernel launcher
    if (!context->status().ok()) {
      return;
    }

    bool status = BilateralSliceKernelLauncher(
        context->eigen_device<GPUDevice>(),
        bs, gh, gw, gd, chans,
        h, w,
        bilateral_grid.flat<float>().data(), guide.flat<float>().data(), 
        output.data());

    if (!status) {
      context->SetStatus(
          errors::Internal("Failed launch BilateralSliceKernel."));
    }
  }
};


class BilateralSliceGradOp : public OpKernel {
 public:
  explicit BilateralSliceGradOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the inputs
    const Tensor& bilateral_grid = context->input(0);
    const Tensor& guide = context->input(1);
    const Tensor& backprop = context->input(2);

    OP_REQUIRES(
        context, bilateral_grid.dims() == 5,
        errors::InvalidArgument(
        R"msg(Input grid should be 5D (batch, height, width, depth, nchannels))msg"));
    OP_REQUIRES(
        context, guide.dims() == 3,
        errors::InvalidArgument(
        R"msg(Guide image should be 3D (batch, height, width))msg"));
    OP_REQUIRES(
        context, backprop.dims() == 4,
        errors::InvalidArgument(
        R"msg(Backprop should be 4D (batch, height, width, nchannels))msg"));

    // Get shape of output tensor
    TensorShape grid_shape = bilateral_grid.shape();
    TensorShape guide_shape = guide.shape();

    // Allocate output tensor
    Tensor* grid_grad = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grid_shape,
                                                     &grid_grad));
    Tensor* guide_grad = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, guide_shape,
                                                     &guide_grad));

    const int64 *grid_size = bilateral_grid.shape().dim_sizes().data();
    const int64 *guide_size = guide.shape().dim_sizes().data();

    auto grid_grad_array = grid_grad->template flat<float>();
    auto guide_grad_array = guide_grad->template flat<float>();

    // Call the cuda kernel launcher
    bool status = BilateralSliceGradKernelLauncher(
        context->eigen_device<GPUDevice>(),
        bilateral_grid.flat<float>().data(), grid_size,
        guide.flat<float>().data(), guide_size,
        backprop.flat<float>().data(),
        grid_grad_array.data(), guide_grad_array.data());

    if (!status) {
      context->SetStatus(
          errors::Internal("Failed launch BilateralSliceGradKernel."));
    }
  }
};


class BilateralSliceApplyOp : public OpKernel {
  private:
    bool has_offset;

  public:
    explicit BilateralSliceApplyOp(OpKernelConstruction* context) : OpKernel(context) {
      OP_REQUIRES_OK(context, context->GetAttr("has_offset", &has_offset));
    }

    void Compute(OpKernelContext* context) override {
      // Grab the inputs
      const Tensor& bilateral_grid = context->input(0);
      const Tensor& guide = context->input(1);
      const Tensor& input = context->input(2);

      // Check tensor dims
      OP_REQUIRES(
          context, bilateral_grid.dims() == 5,
          errors::InvalidArgument(
            R"msg(Input grid should be 5D (batch, height, width, depth, nchannels))msg"));
      OP_REQUIRES(
          context, guide.dims() == 3,
          errors::InvalidArgument(
            R"msg(Guide image should be 3D (batch, height, width))msg"));
      OP_REQUIRES(
          context, input.dims() == 4,
          errors::InvalidArgument(
            R"msg(Guide image should be 4D (batch, height, width, nchannels))msg"));

      // Sizes
      const int64 *grid_size = bilateral_grid.shape().dim_sizes().data();
      const int64 *guide_size = guide.shape().dim_sizes().data();
      int h = guide.dim_size(1);
      int w = guide.dim_size(2);
      int bs = bilateral_grid.dim_size(0);
      int gh = bilateral_grid.dim_size(1);
      int gw = bilateral_grid.dim_size(2);
      int gd = bilateral_grid.dim_size(3);
      int coeffs_chans = bilateral_grid.dim_size(4);
      int input_chans = input.dim_size(3);

      OP_REQUIRES(
          context, input.dim_size(0) == guide.dim_size(0) && input.dim_size(1) == h && input.dim_size(2) == w,
          errors::InvalidArgument(
            R"msg(Input and guide size should match.)msg"));
      OP_REQUIRES(
          context, guide.dim_size(0) == bs,
          errors::InvalidArgument(
            R"msg(Batch sizes should match.)msg"));

      int output_chans = 0;
      if (has_offset) {
        OP_REQUIRES(
            context, coeffs_chans % (input_chans+1) == 0,
            errors::InvalidArgument(
              R"msg(Slicing with affine offset, coefficients grid should have n_out*(n_in+1) channels.)msg"));
        output_chans = coeffs_chans / (input_chans+1);
      } else {
        OP_REQUIRES(
            context, coeffs_chans % input_chans == 0,
            errors::InvalidArgument(
              R"msg(Slicing without affine offset, coefficients grid should have n_out*n_in channels.)msg"));
        output_chans = coeffs_chans / input_chans;
      }

      // Allocate output tensor
      TensorShape out_shape;
      out_shape.AddDim(bs);
      out_shape.AddDim(h);
      out_shape.AddDim(w);
      out_shape.AddDim(output_chans);
      Tensor* output_tensor = NULL;
      OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &output_tensor));

      // Call the cuda kernel launcher
      auto output = output_tensor->flat<float>();
      bool status = BilateralSliceApplyKernelLauncher(
          context->eigen_device<GPUDevice>(),
          bs, gh, gw, gd, 
          input_chans, output_chans, has_offset,
          h, w,
          bilateral_grid.flat<float>().data(), guide.flat<float>().data(), input.flat<float>().data(),
          output.data());

      if (!status) {
        context->SetStatus(
            errors::Internal("Failed to launch BilateralSliceApplyKernel."));
      }
    }
};

class BilateralSliceApplyGradOp : public OpKernel {
  private:
    bool has_offset;

  public:
    explicit BilateralSliceApplyGradOp(OpKernelConstruction* context) : OpKernel(context) {
      OP_REQUIRES_OK(context, context->GetAttr("has_offset", &has_offset));
    }

    void Compute(OpKernelContext* context) override {
      // Grab the inputs
      const Tensor& bilateral_grid = context->input(0);
      const Tensor& guide = context->input(1);
      const Tensor& input = context->input(2);
      const Tensor& backprop = context->input(3);

      OP_REQUIRES(
          context, bilateral_grid.dims() == 5,
          errors::InvalidArgument(
            R"msg(Input grid should be 5D (batch, height, width, depth, nchannels))msg"));
      OP_REQUIRES(
          context, guide.dims() == 3,
          errors::InvalidArgument(
            R"msg(Guide image should be 3D (batch, height, width))msg"));
      OP_REQUIRES(
          context, input.dims() == 4,
          errors::InvalidArgument(
            R"msg(Input image should be 4D (batch, height, width, nchannels))msg"));
      OP_REQUIRES(
          context, backprop.dims() == 4,
          errors::InvalidArgument(
            R"msg(Backprop should be 4D (batch, height, width, nchannels))msg"));

      // Get shape of output tensor
      TensorShape grid_shape = bilateral_grid.shape();
      TensorShape guide_shape = guide.shape();
      TensorShape input_shape = input.shape();

      // Allocate output tensor
      Tensor* grid_grad = NULL;
      OP_REQUIRES_OK(context, context->allocate_output(0, grid_shape,
            &grid_grad));
      Tensor* guide_grad = NULL;
      OP_REQUIRES_OK(context, context->allocate_output(1, guide_shape,
            &guide_grad));
      Tensor* input_grad = NULL;
      OP_REQUIRES_OK(context, context->allocate_output(2, input_shape,
            &input_grad));

      const int64 *grid_size = bilateral_grid.shape().dim_sizes().data();
      const int64 *guide_size = guide.shape().dim_sizes().data();
      const int64 *input_size = input.shape().dim_sizes().data();

      auto grid_grad_array = grid_grad->template flat<float>();
      auto guide_grad_array = guide_grad->template flat<float>();
      auto input_grad_array = input_grad->template flat<float>();

      // Call the cuda kernel launcher
      bool status = BilateralSliceApplyGradKernelLauncher(
          context->eigen_device<GPUDevice>(),
          bilateral_grid.flat<float>().data(), grid_size,
          guide.flat<float>().data(), guide_size,
          input.flat<float>().data(), input_size,
          backprop.flat<float>().data(), has_offset,
          grid_grad_array.data(), guide_grad_array.data(), input_grad_array.data());

      if (!status) {
        context->SetStatus(
            errors::Internal("Failed launch BilateralSliceApplyGradKernel."));
      }
    }
};
// ----------------------------------------------------------------------------

// -- KERNEL REGISTRATION -----------------------------------------------------
REGISTER_KERNEL_BUILDER(Name("BilateralSlice").Device(DEVICE_GPU), BilateralSliceOp);
REGISTER_KERNEL_BUILDER(Name("BilateralSliceGrad").Device(DEVICE_GPU), BilateralSliceGradOp);
REGISTER_KERNEL_BUILDER(Name("BilateralSliceApply").Device(DEVICE_GPU), BilateralSliceApplyOp);
REGISTER_KERNEL_BUILDER(Name("BilateralSliceApplyGrad").Device(DEVICE_GPU), BilateralSliceApplyGradOp);
// ----------------------------------------------------------------------------
