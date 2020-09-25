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

#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include <algorithm>
#include <cstdint>

#include "numerics.h"
#include "third_party/array/array.h"
#include "third_party/tensorflow/core/util/gpu_kernel_helper.h"

namespace {

using GpuDevice = Eigen::GpuDevice;
using ::tensorflow::GetGpuLaunchConfig;
using ::tensorflow::GpuLaunchConfig;

}  // namespace

// TODO(jiawen): `grid_data` should not be necessary but is needed to work
// around a compiler bug in -O2 mode.
__global__ void BilateralSliceApplyKernel(
    const int nthreads, nda::array_ref_of_rank<const float, 6> grid,
    const float* grid_data, nda::array_ref_of_rank<const float, 3> guide,
    nda::array_ref_of_rank<const float, 4> input,
    nda::array_ref_of_rank<float, 4> out) {
  // - Samples centered at 0.5.
  // - Repeating boundary conditions.
  const int grid_input_channels = grid.dim<0>().extent();
  const int output_channels = grid.dim<1>().extent();
  const int grid_depth = grid.dim<2>().extent();
  const int grid_width = grid.dim<3>().extent();
  const int grid_height = grid.dim<4>().extent();
  const int input_channels = input.dim<0>().extent();
  const int input_width = input.dim<1>().extent();
  const int input_height = input.dim<2>().extent();
  const float scale_x = static_cast<float>(grid_width) / input_width;
  const float scale_y = static_cast<float>(grid_height) / input_height;

  // TODO(jiawen): Workaround for nda::array_ref -O2 compiler bug.
  const int grid_i_stride = grid.dim<1>().stride();
  const int grid_z_stride = grid.dim<2>().stride();
  const int grid_x_stride = grid.dim<3>().stride();
  const int grid_y_stride = grid.dim<4>().stride();
  const int grid_b_stride = grid.dim<5>().stride();

  // Factor the 1D index `idx` back into a 4D index.
  // TODO(jiawen): Remove the factorization by launching a 3D grid and using a
  // for loop over the remaining axis instead.
  const int output_x_stride = output_channels;
  const int output_y_stride = output_x_stride * input_width;
  const int output_b_stride = output_y_stride * input_height;
  CUDA_1D_KERNEL_LOOP(idx, nthreads) {
    const int i = idx % output_channels;
    const int x = (idx / output_x_stride) % input_width;
    const int y = (idx / output_y_stride) % input_height;
    const int b = idx / output_b_stride;

    const float gxf = (x + 0.5f) * scale_x;
    const float gyf = (y + 0.5f) * scale_y;
    // TODO(jiawen): Offset gz by 0.5 as well.
    const float gzf = guide(x, y, b) * grid_depth;

    const int gx0 = static_cast<int>(std::floor(gxf - 0.5f));
    const int gy0 = static_cast<int>(std::floor(gyf - 0.5f));
    const int gz0 = static_cast<int>(std::floor(gzf - 0.5f));

    float value = 0.0f;
    for (int j = 0; j < grid_input_channels; ++j) {
      // Grid trilinear interpolation to retrieve grid(gxf, gyf, gzf, i, j).
      float grid_sample = 0.0f;

      for (int gy = gy0; gy < gy0 + 2; ++gy) {
        const int gyc = std::clamp(gy, 0, grid_height - 1);
        const float wy = LerpWeight(gy + 0.5f, gyf);

        for (int gx = gx0; gx < gx0 + 2; ++gx) {
          const int gxc = std::clamp(gx, 0, grid_width - 1);
          const float wx = LerpWeight(gx + 0.5f, gxf);

          for (int gz = gz0; gz < gz0 + 2; ++gz) {
            const int gzc = std::clamp(gz, 0, grid_depth - 1);
            const float wz = SmoothedLerpWeight(gz + 0.5f, gzf);

            const int grid_idx = j + grid_i_stride * i + grid_z_stride * gzc +
                                 grid_x_stride * gxc + grid_y_stride * gyc +
                                 grid_b_stride * b;
            // TODO(jiawen): The workaround used for input grad below doesn't
            // even work here. Using replacing `grid_data` with `grid.base()`
            // breaks tests.
            //
            // Even `grid_data[grid.shape()(j, i, gzc, gxc, gyc, b)]` is
            // broken.
            //
            // It *should* be just `grid(j, i, gzc, gxc, gyc, b)`.
            grid_sample += wx * wy * wz * grid_data[grid_idx];
          }  // gz
        }    // gy
      }      // gx
      // Grid trilinear interpolation.

      // Matrix multiply.
      if (j < input_channels) {
        value += grid_sample * input(j, x, y, b);
      } else {  // Offset term
        value += grid_sample;
      }
    }  // j

    out(i, x, y, b) = value;
  }
}

__global__ void BilateralSliceApplyGridGradKernel(
    const int nthreads, nda::array_ref_of_rank<const float, 3> guide,
    nda::array_ref_of_rank<const float, 4> input,
    nda::array_ref_of_rank<const float, 4> codomain_tangent,
    nda::array_ref_of_rank<float, 6> vjp_out) {
  const int grid_depth = vjp_out.dim<2>().extent();
  const int grid_width = vjp_out.dim<3>().extent();
  const int grid_height = vjp_out.dim<4>().extent();
  const int input_channels = input.dim<0>().extent();
  const int input_width = input.dim<1>().extent();
  const int input_height = input.dim<2>().extent();
  const float scale_x = static_cast<float>(input_width) / grid_width;
  const float scale_y = static_cast<float>(input_height) / grid_height;

  // Factor the 1D index `idx` back into a 6D index.
  // TODO(jiawen): Remove the factorization by launching a 3D grid and using a
  // for loop over the remaining three axes instead.
  const int grid_input_channels = vjp_out.dim<0>().extent();
  const int output_channels = vjp_out.dim<1>().extent();
  const int grid_i_stride = output_channels;
  const int grid_z_stride = grid_i_stride * grid_input_channels;
  const int grid_x_stride = grid_z_stride * grid_depth;
  const int grid_y_stride = grid_x_stride * grid_width;
  const int grid_b_stride = grid_y_stride * grid_height;
  CUDA_1D_KERNEL_LOOP(idx, nthreads) {
    const int j = idx % grid_input_channels;
    const int i = (idx / grid_i_stride) % output_channels;
    const int gz = (idx / grid_z_stride) % grid_depth;
    const int gx = (idx / grid_x_stride) % grid_width;
    const int gy = (idx / grid_y_stride) % grid_height;
    const int b = idx / grid_b_stride;

    const int x0 = static_cast<int>(std::floor(scale_x * (gx + 0.5f - 1.0f)));
    const int x1_exclusive =
        static_cast<int>(std::ceil(scale_x * (gx + 0.5f + 1.0f)));
    const int y0 = static_cast<int>(std::floor(scale_y * (gy + 0.5f - 1.0f)));
    const int y1_exclusive =
        static_cast<int>(std::ceil(scale_y * (gy + 0.5f + 1.0f)));

    float vjp_value = 0.0f;
    for (int y = y0; y < y1_exclusive; ++y) {
      const int y_mirror = MirrorBoundary(y, input_height);
      const float gyf = (y + 0.5f) / scale_y;
      const float wy = LerpWeight(gy + 0.5f, gyf);

      for (int x = x0; x < x1_exclusive; ++x) {
        // TODO(jiawen): Consider using clamp boundary.
        const int x_mirror = MirrorBoundary(x, input_width);
        const float gxf = (x + 0.5f) / scale_x;
        const float wx = LerpWeight(gx + 0.5f, gxf);

        // TODO(jiawen): Offset gz by 0.5 as well.
        const float gzf = guide(x_mirror, y_mirror, b) * grid_depth;
        float wz = SmoothedLerpWeight(gz + 0.5f, gzf);
        if ((gz == 0 && gzf < 0.5f) ||
            (gz == grid_depth - 1 && gzf > grid_depth - 0.5f)) {
          wz = 1.0f;
        }

        // Index `input` accounting for optional offset.
        const float input_value =
            (j < input_channels) ? input(j, x_mirror, y_mirror, b) : 1.0f;
        const float grad_value = wx * wy * wz * input_value;

        vjp_value += grad_value * codomain_tangent(i, x_mirror, y_mirror, b);
      }  // y
    }    // x

    vjp_out(j, i, gz, gx, gy, b) = vjp_value;
  }
}

// TODO(jiawen): `grid_data` and `codomain_tangent_data` should not be necessary
// but is needed to work around a compiler bug in -O2 mode.
__global__ void BilateralSliceApplyGuideGradKernel(
    const int nthreads, nda::array_ref_of_rank<const float, 6> grid,
    const float* grid_data, nda::array_ref_of_rank<const float, 3> guide,
    nda::array_ref_of_rank<const float, 4> input,
    nda::array_ref_of_rank<const float, 4> codomain_tangent,
    const float* codomain_tangent_data,
    nda::array_ref_of_rank<float, 3> vjp_out) {
  const int grid_input_channels = grid.dim<0>().extent();
  const int output_channels = grid.dim<1>().extent();
  const int grid_depth = grid.dim<2>().extent();
  const int grid_width = grid.dim<3>().extent();
  const int grid_height = grid.dim<4>().extent();
  const int input_channels = input.dim<0>().extent();
  const int input_width = input.dim<1>().extent();
  const int input_height = input.dim<2>().extent();
  const float scale_x = static_cast<float>(grid_width) / input_width;
  const float scale_y = static_cast<float>(grid_height) / input_height;

  // TODO(jiawen): Workaround for nda::array_ref -O2 compiler bug.
  const int grid_i_stride = grid.dim<1>().stride();
  const int grid_z_stride = grid.dim<2>().stride();
  const int grid_x_stride = grid.dim<3>().stride();
  const int grid_y_stride = grid.dim<4>().stride();
  const int grid_b_stride = grid.dim<5>().stride();

  // Factor the 1D index `idx` back into a 3D index.
  // TODO(jiawen): Remove the factorization by launching a 3D grid.
  const int input_y_stride = input_width;
  const int input_b_stride = input_y_stride * input_height;
  CUDA_1D_KERNEL_LOOP(idx, nthreads) {
    const int x = idx % input_width;
    const int y = (idx / input_y_stride) % input_height;
    const int b = idx / input_b_stride;

    const float gxf = (x + 0.5f) * scale_x;
    const float gyf = (y + 0.5f) * scale_y;
    // TODO(jiawen): Offset gz by 0.5 as well.
    const float gzf = guide(x, y, b) * grid_depth;

    const int gx0 = static_cast<int>(std::floor(gxf - 0.5f));
    const int gy0 = static_cast<int>(std::floor(gyf - 0.5f));
    const int gz0 = static_cast<int>(std::floor(gzf - 0.5f));

    float vjp_value = 0.0f;
    for (int i = 0; i < output_channels; ++i) {
      float grad_value = 0.0f;

      for (int j = 0; j < grid_input_channels; ++j) {
        float grid_sample = 0.0f;

        for (int gy = gy0; gy < gy0 + 2; ++gy) {
          const int gyc = std::clamp(gy, 0, grid_height - 1);
          const float wy = LerpWeight(gy + 0.5f, gyf);

          // Grid trilinear interpolation to retrieve grid(gxf, gyf, gzf, i, j).
          for (int gx = gx0; gx < gx0 + 2; ++gx) {
            const int gxc = std::clamp(gx, 0, grid_width - 1);
            const float wx = LerpWeight(gx + 0.5f, gxf);

            for (int gz = gz0; gz < gz0 + 2; ++gz) {
              const int gzc = std::clamp(gz, 0, grid_depth - 1);
              // TODO(jiawen): Offset gz by 0.5 as well?
              const float dwz =
                  grid_depth * SmoothedLerpWeightGrad(gz + 0.5f, gzf);

              const int grid_idx = j + grid_i_stride * i + grid_z_stride * gzc +
                                   grid_x_stride * gxc + grid_y_stride * gyc +
                                   grid_b_stride * b;
              // TODO(jiawen): The workaround used for input grad below doesn't
              // even work here. Using replacing `grid_data` with `grid.base()`
              // breaks tests.
              //
              // Even `grid_data[grid.shape()(j, i, gzc, gxc, gyc, b)]` is
              // broken.
              //
              // It *should* be just `grid(j, i, gzc, gxc, gyc, b)`.
              grid_sample += wx * wy * dwz * grid_data[grid_idx];
            }  // gz
          }    // gy
        }      // gx
        // Grid trilinear interpolation.

        // Index `input` accounting for optional offset.
        const float input_value =
            (j < input_channels) ? input(j, x, y, b) : 1.0f;
        grad_value += grid_sample * input_value;
      }  // Sum over j.

      const int codomain_tangent_idx =
          i + output_channels * (x + input_width * (y + input_height * b));
      // TODO(jiawen): The workaround used for input grad below doesn't
      // even work here. Using replacing `codomain_tangent_data` with
      // `codomain_tangent.base()` breaks tests.
      //
      // Even `codomain_tangent_data[codomain_tangent.shape()(i, x, y, b)]` is
      // broken.
      //
      // It *should* be just `codomain_tangent(i, x, y, b)`.
      vjp_value += grad_value * codomain_tangent_data[codomain_tangent_idx];
    }  // Sum over i.

    vjp_out(x, y, b) = vjp_value;
  }
}

__global__ void BilateralSliceApplyInputGradKernel(
    const int nthreads, nda::array_ref_of_rank<const float, 6> grid,
    nda::array_ref_of_rank<const float, 3> guide,
    nda::array_ref_of_rank<const float, 4> codomain_tangent,
    nda::array_ref_of_rank<float, 4> vjp_out) {
  const int output_channels = grid.dim<1>().extent();
  const int grid_depth = grid.dim<2>().extent();
  const int grid_width = grid.dim<3>().extent();
  const int grid_height = grid.dim<4>().extent();
  const int guide_width = guide.dim<0>().extent();
  const int guide_height = guide.dim<1>().extent();
  const float scale_x = static_cast<float>(grid_width) / guide_width;
  const float scale_y = static_cast<float>(grid_height) / guide_height;

  // Factor the 1D index `idx` back into a 4D index.
  // TODO(jiawen): Remove the factorization by launching a 3D grid and using a
  // for loop over the remaining axis instead.
  const int input_x_stride = vjp_out.dim<1>().stride();
  const int input_y_stride = vjp_out.dim<2>().stride();
  const int input_b_stride = vjp_out.dim<3>().stride();
  CUDA_1D_KERNEL_LOOP(idx, nthreads) {
    const int j = idx % input_x_stride;
    const int x = (idx / input_x_stride) % guide_width;
    const int y = (idx / input_y_stride) % guide_height;
    const int b = idx / input_b_stride;

    const float gxf = (x + 0.5f) * scale_x;
    const float gyf = (y + 0.5f) * scale_y;
    // TODO(jiawen): Offset gz by 0.5 as well.
    const float gzf = guide(x, y, b) * grid_depth;

    const int gx0 = static_cast<int>(std::floor(gxf - 0.5f));
    const int gy0 = static_cast<int>(std::floor(gyf - 0.5f));
    const int gz0 = static_cast<int>(std::floor(gzf - 0.5f));

    float vjp_value = 0.0f;
    for (int i = 0; i < output_channels; ++i) {
      float grad_value = 0.0f;

      // Grid trilinear interpolation to retrieve grid(gxf, gyf, gzf, i, j).
      for (int gy = gy0; gy < gy0 + 2; ++gy) {
        const int gyc = std::clamp(gy, 0, grid_height - 1);
        const float wy = LerpWeight(gy + 0.5f, gyf);

        for (int gx = gx0; gx < gx0 + 2; ++gx) {
          const int gxc = std::clamp(gx, 0, grid_width - 1);
          const float wx = LerpWeight(gx + 0.5f, gxf);

          for (int gz = gz0; gz < gz0 + 2; ++gz) {
            const int gzc = std::clamp(gz, 0, grid_depth - 1);
            const float wz = SmoothedLerpWeight(gz + 0.5f, gzf);

            // TODO(jiawen): This should be grid(j, i, gzc, gxc, gyc, b), but
            // tests fail due to a compiler error.
            grad_value += wx * wy * wz *
                          grid.base()[grid.shape()(j, i, gzc, gxc, gyc, b)];
          }  // gz
        }    // gy
      }      // gx
      // Grid trilinear interpolation.

      vjp_value += grad_value * codomain_tangent(i, x, y, b);
    }  // Sum over i.

    vjp_out(j, x, y, b) = vjp_value;
  }
}

namespace hdrnet {

bool BilateralSliceApplyCudaLauncher(
    const GpuDevice& device, nda::array_ref_of_rank<const float, 6> grid,
    nda::array_ref_of_rank<const float, 3> guide,
    nda::array_ref_of_rank<const float, 4> input,
    nda::array_ref_of_rank<float, 4> out) {
  const int out_count = out.size();
  if (out_count > 0) {
    const GpuLaunchConfig config = GetGpuLaunchConfig(out_count, device);
    BilateralSliceApplyKernel<<<config.block_count, config.thread_per_block, 0,
                                device.stream()>>>(out_count, grid, grid.data(),
                                                   guide, input, out);
  }

  return device.ok();
}

bool BilateralSliceApplyGradCudaLauncher(
    const GpuDevice& device, nda::array_ref_of_rank<const float, 6> grid,
    nda::array_ref_of_rank<const float, 3> guide,
    nda::array_ref_of_rank<const float, 4> input,
    nda::array_ref_of_rank<const float, 4> codomain_tangent,
    nda::array_ref_of_rank<float, 6> grid_vjp_out,
    nda::array_ref_of_rank<float, 3> guide_vjp_out,
    nda::array_ref_of_rank<float, 4> input_vjp_out) {
  const int grid_vjp_count = grid_vjp_out.size();
  if (grid_vjp_count > 0) {
    const GpuLaunchConfig config = GetGpuLaunchConfig(grid_vjp_count, device);
    BilateralSliceApplyGridGradKernel<<<
        config.block_count, config.thread_per_block, 0, device.stream()>>>(
        grid_vjp_count, guide, input, codomain_tangent, grid_vjp_out);
  }

  const int guide_vjp_count = guide_vjp_out.size();
  if (guide_vjp_count > 0) {
    const GpuLaunchConfig config = GetGpuLaunchConfig(guide_vjp_count, device);
    BilateralSliceApplyGuideGradKernel<<<
        config.block_count, config.thread_per_block, 0, device.stream()>>>(
        guide_vjp_count, grid, grid.data(), guide, input, codomain_tangent,
        codomain_tangent.data(), guide_vjp_out);
  }

  const int input_vjp_count = input_vjp_out.size();
  if (input_vjp_count > 0) {
    const GpuLaunchConfig config = GetGpuLaunchConfig(input_vjp_count, device);
    BilateralSliceApplyInputGradKernel<<<
        config.block_count, config.thread_per_block, 0, device.stream()>>>(
        input_vjp_count, grid, guide, codomain_tangent, input_vjp_out);
  }

  return device.ok();
}

}  // namespace hdrnet

#endif
