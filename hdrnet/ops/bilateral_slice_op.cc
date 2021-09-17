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

#include "bilateral_slice.h"
#include "third_party/array/array.h"
#include "third_party/tensorflow/core/framework/op.h"
#include "third_party/tensorflow/core/framework/op_kernel.h"
#include "third_party/tensorflow/core/framework/shape_inference.h"
#include "third_party/tensorflow/core/framework/tensor.h"
#include "third_party/tensorflow/core/framework/tensor_shape.h"
#include "third_party/tensorflow/core/framework/tensor_types.h"

using CpuDevice = ::Eigen::ThreadPoolDevice;
using GpuDevice = ::Eigen::GpuDevice;

using ::tensorflow::OpKernel;
using ::tensorflow::OpKernelConstruction;
using ::tensorflow::OpKernelContext;
using ::tensorflow::Status;
using ::tensorflow::Tensor;
using ::tensorflow::TensorShape;
using ::tensorflow::shape_inference::DimensionHandle;
using ::tensorflow::shape_inference::InferenceContext;
using ::tensorflow::shape_inference::ShapeHandle;

namespace hdrnet {

// Declare BilateralSlice and BilateralSliceGrad templated on the device. They
// will be specialized for each device.
template <typename Device>
bool BilateralSlice(const Device& device,
                    nda::array_ref_of_rank<const float, 5> grid,
                    nda::array_ref_of_rank<const float, 3> guide,
                    nda::array_ref_of_rank<float, 4> out);

template <typename Device>
bool BilateralSliceGrad(const Device& device,
                        nda::array_ref_of_rank<const float, 5> grid,
                        nda::array_ref_of_rank<const float, 3> guide,
                        nda::array_ref_of_rank<const float, 4> codomain_tangent,
                        nda::array_ref_of_rank<float, 5> grid_vjp_out,
                        nda::array_ref_of_rank<float, 3> guide_vjp_out);

// Specialize for the CPU (ignoring the device).
template <>
bool BilateralSlice<CpuDevice>(const CpuDevice& device,
                               nda::array_ref_of_rank<const float, 5> grid,
                               nda::array_ref_of_rank<const float, 3> guide,
                               nda::array_ref_of_rank<float, 4> out) {
  BilateralSlice(grid, guide, out);
  return true;
}

template <>
bool BilateralSliceGrad<CpuDevice>(
    const CpuDevice& device, nda::array_ref_of_rank<const float, 5> grid,
    nda::array_ref_of_rank<const float, 3> guide,
    nda::array_ref_of_rank<const float, 4> codomain_tangent,
    nda::array_ref_of_rank<float, 5> grid_vjp_out,
    nda::array_ref_of_rank<float, 3> guide_vjp_out) {
  BilateralSliceGridGrad(guide, codomain_tangent, grid_vjp_out);
  BilateralSliceGuideGrad(grid, guide, codomain_tangent, guide_vjp_out);
  return true;
}

// Specialize for the GPU.
#if GOOGLE_CUDA

// Forward declare CUDA launchers since #includes are messy.
bool BilateralSliceCudaLauncher(const GpuDevice& device,
                                nda::array_ref_of_rank<const float, 5> grid,
                                nda::array_ref_of_rank<const float, 3> guide,
                                nda::array_ref_of_rank<float, 4> out);

bool BilateralSliceGradCudaLauncher(
    const GpuDevice& device, nda::array_ref_of_rank<const float, 5> grid,
    nda::array_ref_of_rank<const float, 3> guide,
    nda::array_ref_of_rank<const float, 4> codomain_tangent,
    nda::array_ref_of_rank<float, 5> grid_vjp_out,
    nda::array_ref_of_rank<float, 3> guide_vjp_out);

template <>
bool BilateralSlice<GpuDevice>(const GpuDevice& device,
                               nda::array_ref_of_rank<const float, 5> grid,
                               nda::array_ref_of_rank<const float, 3> guide,
                               nda::array_ref_of_rank<float, 4> out) {
  return BilateralSliceCudaLauncher(device, grid, guide, out);
}

template <>
bool BilateralSliceGrad<GpuDevice>(
    const GpuDevice& device, nda::array_ref_of_rank<const float, 5> grid,
    nda::array_ref_of_rank<const float, 3> guide,
    nda::array_ref_of_rank<const float, 4> codomain_tangent,
    nda::array_ref_of_rank<float, 5> grid_vjp_out,
    nda::array_ref_of_rank<float, 3> guide_vjp_out) {
  return BilateralSliceGradCudaLauncher(device, grid, guide, codomain_tangent,
                                        grid_vjp_out, guide_vjp_out);
}

#endif  // GOOGLE_CUDA

template <typename Device>
class BilateralSliceOp : public OpKernel {
 public:
  explicit BilateralSliceOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grabs the inputs.
    const Tensor& grid = context->input(0);
    const Tensor& guide = context->input(1);

    // Check tensor dims.
    OP_REQUIRES(context, grid.dims() == 5,
                tensorflow::errors::InvalidArgument(
                    "Grid should be 5D (batch_size, grid_height, grid_width, "
                    "grid_depth, grid_channels)."));
    OP_REQUIRES(context, guide.dims() == 3,
                tensorflow::errors::InvalidArgument(
                    "Guide image should be 3D (batch_size, height, width)."));

    // Input shapes.
    const int batch_size = grid.dim_size(0);
    const int grid_height = grid.dim_size(1);
    const int grid_width = grid.dim_size(2);
    const int grid_depth = grid.dim_size(3);
    const int grid_channels = grid.dim_size(4);
    const int guide_height = guide.dim_size(1);
    const int guide_width = guide.dim_size(2);

    // Allocate output tensor.
    const TensorShape output_shape(
        {batch_size, guide_height, guide_width, grid_channels});
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));

    // TF: (b, h, w, d, c), c changes fastest.
    // nda: (c, d, w, h, b), c changes fastest.
    auto grid_ref = nda::make_array_ref(
        grid.flat<float>().data(),
        nda::shape_of_rank<5>(grid_channels, grid_depth, grid_width,
                              grid_height, batch_size));
    // TF: (b, h, w), w changes fastest.
    // nda: (w, h, b), w changes fastest.
    auto guide_ref = nda::make_array_ref(
        guide.flat<float>().data(),
        nda::shape_of_rank<3>(guide_width, guide_height, batch_size));

    // TF: (b, h, w, c), c changes fastest.
    // nda: (c, w, h, b), c changes fastest.
    auto output_ref =
        nda::make_array_ref(output->flat<float>().data(),
                            nda::shape_of_rank<4>(grid_channels, guide_width,
                                                  guide_height, batch_size));

    const bool status = BilateralSlice(context->eigen_device<Device>(),
                                       grid_ref, guide_ref, output_ref);
    if (!status) {
      context->SetStatus(
          tensorflow::errors::Internal("BilateralSlice kernel failed."));
    }
  }
};

template <typename Device>
class BilateralSliceGradOp : public OpKernel {
 public:
  explicit BilateralSliceGradOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the inputs.
    const Tensor& grid = context->input(0);
    const Tensor& guide = context->input(1);
    const Tensor& codomain_tangent = context->input(2);

    // Check tensor dims.
    OP_REQUIRES(context, grid.dims() == 5,
                tensorflow::errors::InvalidArgument(
                    "Grid should be 5D with shape (batch_size, grid_height, "
                    "grid_width, grid_depth, grid_channels)."));
    OP_REQUIRES(context, guide.dims() == 3,
                tensorflow::errors::InvalidArgument(
                    "Guide image should be 3D (batch_size, height, width)."));
    OP_REQUIRES(context, codomain_tangent.dims() == 4,
                tensorflow::errors::InvalidArgument(
                    "Codomain tangent should be 4D (batch, height, width, "
                    "nchannels))."));

    // Allocate vjp buffers, which have the same shape as the primals.
    Tensor* grid_vjp = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, grid.shape(), &grid_vjp));
    Tensor* guide_vjp = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(1, guide.shape(), &guide_vjp));

    // Input shapes.
    const int batch_size = grid.dim_size(0);
    const int grid_height = grid.dim_size(1);
    const int grid_width = grid.dim_size(2);
    const int grid_depth = grid.dim_size(3);
    const int grid_channels = grid.dim_size(4);
    const int guide_height = guide.dim_size(1);
    const int guide_width = guide.dim_size(2);

    // `grid` and `grid_vjp`:
    //
    // TF: (b, h, w, d, c), c changes fastest.
    // nda: (c, d, w, h, b), c changes fastest.
    auto grid_ref = nda::make_array_ref(
        grid.flat<float>().data(),
        nda::shape_of_rank<5>(grid_channels, grid_depth, grid_width,
                              grid_height, batch_size));
    auto grid_vjp_ref =
        nda::make_array_ref(grid_vjp->flat<float>().data(), grid_ref.shape());

    // `guide` and `guide_vjp`:
    //
    // TF: (b, h, w), w changes fastest.
    // nda: (w, h, b), w changes fastest.
    auto guide_ref = nda::make_array_ref(
        guide.flat<float>().data(),
        nda::shape_of_rank<3>(guide_width, guide_height, batch_size));
    auto guide_vjp_ref =
        nda::make_array_ref(guide_vjp->flat<float>().data(), guide_ref.shape());

    // `codomain_tangent`:
    //
    // TF: (b, h, w, c), c changes fastest.
    // nda: (c, w, h, b), c changes fastest.
    auto codomain_tangent_ref =
        nda::make_array_ref(codomain_tangent.flat<float>().data(),
                            nda::shape_of_rank<4>(grid_channels, guide_width,
                                                  guide_height, batch_size));

    const bool status =
        BilateralSliceGrad(context->eigen_device<Device>(), grid_ref, guide_ref,
                           codomain_tangent_ref, grid_vjp_ref, guide_vjp_ref);
    if (!status) {
      context->SetStatus(
          tensorflow::errors::Internal("BilateralSliceGrad kernel failed."));
    }
  }
};

}  // namespace hdrnet

REGISTER_KERNEL_BUILDER(Name("BilateralSlice").Device(tensorflow::DEVICE_CPU),
                        hdrnet::BilateralSliceOp<CpuDevice>);
REGISTER_KERNEL_BUILDER(
    Name("BilateralSliceGrad").Device(tensorflow::DEVICE_CPU),
    hdrnet::BilateralSliceGradOp<CpuDevice>);
#if GOOGLE_CUDA
REGISTER_KERNEL_BUILDER(Name("BilateralSlice").Device(tensorflow::DEVICE_GPU),
                        hdrnet::BilateralSliceOp<GpuDevice>);
REGISTER_KERNEL_BUILDER(
    Name("BilateralSliceGrad").Device(tensorflow::DEVICE_GPU),
    hdrnet::BilateralSliceGradOp<GpuDevice>);
#endif  // GOOGLE_CUDA

REGISTER_OP("BilateralSlice")
    .Input("grid: float")
    .Input("guide: float")
    .Output("out: float")
    .Doc("Slices grid at the location defined by guide to produce output.")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle grid;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 5, &grid));
      ShapeHandle guide;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 3, &guide));
      const DimensionHandle batch_size = c->Dim(grid, 0);
      const DimensionHandle h = c->Dim(guide, 1);
      const DimensionHandle w = c->Dim(guide, 2);
      const DimensionHandle grid_channels = c->Dim(grid, 4);
      c->set_output(0, c->MakeShape({batch_size, h, w, grid_channels}));
      return Status::OK();
    });

REGISTER_OP("BilateralSliceGrad")
    .Input("grid: float")
    .Input("guide: float")
    .Input("backprop: float")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle grid;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 5, &grid));
      ShapeHandle guide;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 3, &guide));
      c->set_output(0, grid);
      c->set_output(1, guide);
      return Status::OK();
    })
    .Output("grid_grad: float")
    .Output("guide_grad: float");
