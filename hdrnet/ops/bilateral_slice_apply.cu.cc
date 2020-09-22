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

__global__ void BilateralSliceApplyKernel(
    const int nthreads, const float* grid, const float* guide,
    const float* input, const int batch_size, const int input_height,
    const int input_width, const int grid_height, const int grid_width,
    const int grid_depth, const int grid_input_channels,
    const int input_channels, const int output_channels, float* out) {
  // - Samples centered at 0.5.
  // - Repeating boundary conditions.
  const float scale_x = static_cast<float>(grid_width) / input_width;
  const float scale_y = static_cast<float>(grid_height) / input_height;

  // TODO(jiawen): These extra strides can be removed once we have a CUDA
  // compatible ND array abstraction.
  const int grid_i_stride = grid_input_channels;
  const int grid_z_stride = grid_i_stride * output_channels;
  const int grid_x_stride = grid_z_stride * grid_depth;
  const int grid_y_stride = grid_x_stride * grid_width;
  const int grid_b_stride = grid_y_stride * grid_height;

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
    const int guide_idx = x + input_width * (y + input_height * b);
    // TODO(jiawen): Offset gz by 0.5 as well.
    const float gzf = guide[guide_idx] * grid_depth;

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

            const int grid_idx = (grid_i_stride * i + j) + grid_z_stride * gzc +
                                 grid_x_stride * gxc + grid_y_stride * gyc +
                                 grid_b_stride * b;
            grid_sample += wx * wy * wz * grid[grid_idx];
          }  // gz
        }    // gy
      }      // gx
      // Grid trilinear interpolation.

      // Matrix multiply.
      if (j < input_channels) {
        const int input_idx =
            j + input_channels * (x + input_width * (y + input_height * b));
        value += grid_sample * input[input_idx];
      } else {  // Offset term
        value += grid_sample;
      }
    }
    out[idx] = value;
  }
}

__global__ void BilateralSliceApplyGridGradKernel(
    const int nthreads, const float* grid, const float* guide,
    const float* input, const float* codomain_tangent, const int batch_size,
    const int input_height, const int input_width, const int grid_height,
    const int grid_width, const int grid_depth, const int grid_input_channels,
    const int input_channels, const int output_channels, float* vjp) {
  const int input_x_stride = input_channels;
  const int input_y_stride = input_x_stride * input_width;
  const int input_b_stride = input_y_stride * input_height;

  const int codomain_tangent_x_stride = output_channels;
  const int codomain_tangent_y_stride = codomain_tangent_x_stride * input_width;
  const int codomain_tangent_b_stride =
      codomain_tangent_y_stride * input_height;

  const int grid_i_stride = output_channels;
  const int grid_z_stride = grid_i_stride * grid_input_channels;
  const int grid_x_stride = grid_z_stride * grid_depth;
  const int grid_y_stride = grid_x_stride * grid_width;
  const int grid_b_stride = grid_y_stride * grid_height;

  const float scale_x = static_cast<float>(input_width) / grid_width;
  const float scale_y = static_cast<float>(input_height) / grid_height;

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
        const int guide_idx =
            x_mirror + input_width * y_mirror + input_height * input_width * b;
        const float gzf = guide[guide_idx] * grid_depth;
        float wz = SmoothedLerpWeight(gz + 0.5f, gzf);
        if ((gz == 0 && gzf < 0.5f) ||
            (gz == grid_depth - 1 && gzf > grid_depth - 0.5f)) {
          wz = 1.0f;
        }

        // Index input accounting for optional offset.
        const int input_idx = j + input_x_stride * x_mirror +
                              input_y_stride * y_mirror + input_b_stride * b;
        const float input_value =
            (j < input_channels) ? input[input_idx] : 1.0f;
        const float grad_value = wx * wy * wz * input_value;

        const int codomain_tangent_idx = i +
                                         codomain_tangent_x_stride * x_mirror +
                                         codomain_tangent_y_stride * y_mirror +
                                         codomain_tangent_b_stride * b;
        vjp_value += grad_value * codomain_tangent[codomain_tangent_idx];
      }  // y
    }    // x
    vjp[idx] = vjp_value;
  }
}

__global__ void BilateralSliceApplyGuideGradKernel(
    const int nthreads, const float* grid, const float* guide,
    const float* input, const float* codomain_tangent, const int batch_size,
    const int input_height, const int input_width, const int grid_height,
    const int grid_width, const int grid_depth, const int grid_input_channels,
    const int input_channels, const int output_channels, float* vjp) {
  const int grid_i_stride = grid_input_channels;
  const int grid_z_stride = grid_i_stride * output_channels;
  const int grid_x_stride = grid_z_stride * grid_depth;
  const int grid_y_stride = grid_x_stride * grid_width;
  const int grid_b_stride = grid_y_stride * grid_height;

  const float scale_x = static_cast<float>(grid_width) / input_width;
  const float scale_y = static_cast<float>(grid_height) / input_height;

  const int input_y_stride = input_width;
  const int input_b_stride = input_y_stride * input_height;

  CUDA_1D_KERNEL_LOOP(idx, nthreads) {
    const int x = idx % input_width;
    const int y = (idx / input_y_stride) % input_height;
    const int b = idx / input_b_stride;

    const float gxf = (x + 0.5f) * scale_x;
    const float gyf = (y + 0.5f) * scale_y;
    // TODO(jiawen): Offset gz by 0.5 as well.
    const int guide_idx = x + input_width * (y + input_height * b);
    const float gzf = guide[guide_idx] * grid_depth;

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
              grid_sample += wx * wy * dwz * grid[grid_idx];
            }  // gz
          }    // gy
        }      // gx
        // Grid trilinear interpolation.

        const int input_idx = x + input_width * (y + input_height * b);
        const float input_value =
            (j < input_channels) ? input[input_idx] : 1.0f;
        grad_value += grid_sample * input_value;
      }  // Sum over j.

      const int codomain_tangent_idx =
          i + output_channels * (x + input_width * (y + input_height * b));
      vjp_value += grad_value * codomain_tangent[codomain_tangent_idx];
    }  // Sum over i.
    vjp[idx] = vjp_value;
  }
}

__global__ void BilateralSliceApplyInputGradKernel(
    const int nthreads, const float* grid, const float* guide,
    const float* codomain_tangent, const int batch_size, const int guide_height,
    const int guide_width, const int grid_height, const int grid_width,
    const int grid_depth, const int grid_input_channels,
    const int input_channels, const int output_channels, float* vjp) {
  const float scale_x = static_cast<float>(grid_width) / guide_width;
  const float scale_y = static_cast<float>(grid_height) / guide_height;

  const int grid_i_stride = grid_input_channels;
  const int grid_z_stride = grid_i_stride * output_channels;
  const int grid_x_stride = grid_z_stride * grid_depth;
  const int grid_y_stride = grid_x_stride * grid_width;
  const int grid_b_stride = grid_y_stride * grid_height;

  const int input_x_stride = input_channels;
  const int input_y_stride = input_x_stride * guide_width;
  const int input_b_stride = input_y_stride * guide_height;

  CUDA_1D_KERNEL_LOOP(idx, nthreads) {
    const int j = idx % input_channels;
    const int x = (idx / input_x_stride) % guide_width;
    const int y = (idx / input_y_stride) % guide_height;
    const int b = idx / input_b_stride;

    const float gxf = (x + 0.5f) * scale_x;
    const float gyf = (y + 0.5f) * scale_y;
    // TODO(jiawen): Offset gz by 0.5 as well.
    const int guide_idx = x + guide_width * (y + guide_height * b);
    const float gzf = guide[guide_idx] * grid_depth;

    const int gx0 = static_cast<int>(std::floor(gxf - 0.5f));
    const int gy0 = static_cast<int>(std::floor(gyf - 0.5f));
    const int gz0 = static_cast<int>(std::floor(gzf - 0.5f));

    float vjp_value = 0.0f;
    for (int i = 0; i < output_channels; ++i) {
      float grad_val = 0.0f;

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

            const int grid_idx = j + grid_i_stride * i + grid_z_stride * gzc +
                                 grid_x_stride * gxc + grid_y_stride * gyc +
                                 grid_b_stride * b;
            grad_val += wx * wy * wz * grid[grid_idx];
          }  // gz
        }    // gy
      }      // gx
      // Grid trilinear interpolation.

      const int codomain_tangent_idx =
          i + output_channels * (x + guide_width * (y + guide_height * b));
      vjp_value += grad_val * codomain_tangent[codomain_tangent_idx];
    }  // sum over i.
    vjp[idx] = vjp_value;
  }
}

namespace hdrnet {

bool BilateralSliceApplyCudaLauncher(
    const GpuDevice& device, nda::array_ref_of_rank<const float, 6> grid,
    nda::array_ref_of_rank<const float, 3> guide,
    nda::array_ref_of_rank<const float, 4> input,
    nda::array_ref_of_rank<float, 4> out) {
  const int out_count = out.size();
  const auto [grid_input_channels, output_channels, grid_depth, grid_width,
              grid_height, batch_size] = grid.shape().extent();
  const auto [input_channels, input_width, input_height, input_batch_size] =
      input.shape().extent();

  if (out_count > 0) {
    GpuLaunchConfig config = GetGpuLaunchConfig(out_count, device);
    BilateralSliceApplyKernel<<<config.block_count, config.thread_per_block, 0,
                                device.stream()>>>(
        out_count, grid.data(), guide.data(), input.data(), batch_size,
        input_height, input_width, grid_height, grid_width, grid_depth,
        grid_input_channels, input_channels, output_channels, out.data());
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
  const auto [grid_input_channels, output_channels, grid_depth, grid_width,
              grid_height, batch_size] = grid.shape().extent();
  const auto [input_channels, input_width, input_height, input_batch_size] =
      input.shape().extent();

  const int grid_vjp_count = grid_vjp_out.size();
  if (grid_vjp_count > 0) {
    const GpuLaunchConfig config = GetGpuLaunchConfig(grid_vjp_count, device);
    BilateralSliceApplyGridGradKernel<<<
        config.block_count, config.thread_per_block, 0, device.stream()>>>(
        grid_vjp_count, grid.data(), guide.data(), input.data(),
        codomain_tangent.data(), batch_size, input_height, input_width,
        grid_height, grid_width, grid_depth, grid_input_channels,
        input_channels, output_channels, grid_vjp_out.data());
  }

  const int guide_vjp_count = guide_vjp_out.size();
  if (guide_vjp_count > 0) {
    const GpuLaunchConfig config = GetGpuLaunchConfig(guide_vjp_count, device);
    BilateralSliceApplyGuideGradKernel<<<
        config.block_count, config.thread_per_block, 0, device.stream()>>>(
        guide_vjp_count, grid.data(), guide.data(), input.data(),
        codomain_tangent.data(), batch_size, input_height, input_width,
        grid_height, grid_width, grid_depth, grid_input_channels,
        input_channels, output_channels, guide_vjp_out.data());
  }

  const int input_vjp_count = input_vjp_out.size();
  if (input_vjp_count > 0) {
    const GpuLaunchConfig config = GetGpuLaunchConfig(input_vjp_count, device);
    BilateralSliceApplyInputGradKernel<<<
        config.block_count, config.thread_per_block, 0, device.stream()>>>(
        input_vjp_count, grid.data(), guide.data(), codomain_tangent.data(),
        batch_size, input_height, input_width, grid_height, grid_width,
        grid_depth, grid_input_channels, input_channels, output_channels,
        input_vjp_out.data());
  }

  return device.ok();
}

}  // namespace hdrnet

#endif
