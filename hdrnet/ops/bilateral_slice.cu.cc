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

using GpuDevice = ::Eigen::GpuDevice;
using ::tensorflow::GetGpuLaunchConfig;
using ::tensorflow::GpuLaunchConfig;

}  // namespace

__global__ void BilateralSliceKernel(
    const int nthreads, nda::array_ref_of_rank<const float, 5> grid,
    nda::array_ref_of_rank<const float, 3> guide,
    nda::array_ref_of_rank<float, 4> out) {
  // - Samples centered at 0.5.
  // - Repeating boundary conditions.
  const int grid_channels = grid.dim<0>().extent();
  const int grid_depth = grid.dim<1>().extent();
  const int grid_width = grid.dim<2>().extent();
  const int grid_height = grid.dim<3>().extent();
  const int guide_width = guide.width();
  const int guide_height = guide.height();

  const float scale_x = static_cast<float>(grid_width) / guide_width;
  const float scale_y = static_cast<float>(grid_height) / guide_height;

  // Factor the 1D index `idx` back into a 4D index.
  // TODO(jiawen): Remove the factorization by launching a 3D grid and using a
  // for loop over the remaining axis instead.
  const int output_x_stride = grid_channels;
  const int output_y_stride = output_x_stride * guide_width;
  const int output_b_stride = output_y_stride * guide_height;
  CUDA_1D_KERNEL_LOOP(idx, nthreads) {
    const int c = idx % grid_channels;
    const int x = (idx / output_x_stride) % guide_width;
    const int y = (idx / output_y_stride) % guide_height;
    const int b = idx / output_b_stride;

    const float gxf = (x + 0.5f) * scale_x;
    const float gyf = (y + 0.5f) * scale_y;
    // TODO(jiawen): Offset gz by 0.5f as well.
    const float gzf = guide(x, y, b) * grid_depth;

    const int gx0 = static_cast<int>(std::floor(gxf - 0.5f));
    const int gy0 = static_cast<int>(std::floor(gyf - 0.5f));
    const int gz0 = static_cast<int>(std::floor(gzf - 0.5f));

    // Grid trilinear interpolation to retrieve grid(gxf, gyf, gzf, i, j).
    float value = 0.0f;
    for (int gy = gy0; gy < gy0 + 2; ++gy) {
      const int gyc = std::clamp(gy, 0, grid_height - 1);
      const float wy = LerpWeight(gy + 0.5f, gyf);
      for (int gx = gx0; gx < gx0 + 2; ++gx) {
        const int gxc = std::clamp(gx, 0, grid_width - 1);
        const float wx = LerpWeight(gx + 0.5f, gxf);
        for (int gz = gz0; gz < gz0 + 2; ++gz) {
          const int gzc = std::clamp(gz, 0, grid_depth - 1);
          const float wz = SmoothedLerpWeight(gz + 0.5f, gzf);

          value += wx * wy * wz * grid(c, gzc, gxc, gyc, b);
        }
      }
    }
    // Grid trilinear interpolation.

    out(c, x, y, b) = value;
  }
}

__global__ void BilateralSliceGridGradKernel(
    const int nthreads, nda::array_ref_of_rank<const float, 3> guide,
    nda::array_ref_of_rank<const float, 4> codomain_tangent,
    nda::array_ref_of_rank<float, 5> grid_vjp_out) {
  const int grid_channels = grid_vjp_out.dim<0>().extent();
  const int grid_depth = grid_vjp_out.dim<1>().extent();
  const int grid_width = grid_vjp_out.dim<2>().extent();
  const int grid_height = grid_vjp_out.dim<3>().extent();
  const int guide_width = guide.width();
  const int guide_height = guide.height();
  const float scale_x = static_cast<float>(guide_width) / grid_width;
  const float scale_y = static_cast<float>(guide_height) / grid_height;

  // Factor the 1D index `idx` back into a 5D index.
  // TODO(jiawen): Remove the factorization by launching a 3D grid and using a
  // for loop over the remaining two axes instead.
  const int grid_z_stride = grid_channels;
  const int grid_x_stride = grid_z_stride * grid_depth;
  const int grid_y_stride = grid_x_stride * grid_width;
  const int grid_b_stride = grid_y_stride * grid_height;
  CUDA_1D_KERNEL_LOOP(idx, nthreads) {
    const int gc = idx % grid_channels;
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
      const int y_mirror = MirrorBoundary(y, guide_height);
      const float gyf = (y + 0.5f) / scale_y;
      const float wy = LerpWeight(gy + 0.5f, gyf);

      for (int x = x0; x < x1_exclusive; ++x) {
        // TODO(jiawen): Consider using clamp boundary.
        const int x_mirror = MirrorBoundary(x, guide_width);
        const float gxf = (x + 0.5f) / scale_x;
        const float wx = LerpWeight(gx + 0.5f, gxf);

        // TODO(jiawen): Offset gz by 0.5 as well.
        const float gzf = guide(x_mirror, y_mirror, b) * grid_depth;
        float wz = SmoothedLerpWeight(gz + 0.5f, gzf);
        if ((gz == 0 && gzf < 0.5f) ||
            (gz == grid_depth - 1 && gzf > grid_depth - 0.5f)) {
          wz = 1.0f;
        }

        vjp_value += wz * wx * wy * codomain_tangent(gc, x_mirror, y_mirror, b);
      }  // y
    }    // x

    grid_vjp_out(gc, gz, gx, gy, b) = vjp_value;
  }
}

__global__ void BilateralSliceGuideGradKernel(
    const int nthreads, nda::array_ref_of_rank<const float, 5> grid,
    nda::array_ref_of_rank<const float, 3> guide,
    nda::array_ref_of_rank<const float, 4> codomain_tangent,
    nda::array_ref_of_rank<float, 3> guide_vjp_out) {
  const int grid_channels = grid.dim<0>().extent();
  const int grid_depth = grid.dim<1>().extent();
  const int grid_width = grid.dim<2>().extent();
  const int grid_height = grid.dim<3>().extent();
  const int guide_width = guide.width();
  const int guide_height = guide.height();
  const float scale_x = static_cast<float>(grid_width) / guide_width;
  const float scale_y = static_cast<float>(grid_height) / guide_height;

  // Factor the 1D index `idx` back into a 3D index.
  // TODO(jiawen): Remove the factorization by launching a 3D grid and using a
  // for loop over the remaining two axes instead.
  const int guide_y_stride = guide_width;
  const int guide_b_stride = guide_y_stride * guide_height;
  CUDA_1D_KERNEL_LOOP(idx, nthreads) {
    const int x = idx % guide_width;
    const int y = (idx / guide_y_stride) % guide_height;
    const int b = idx / guide_b_stride;

    const float gxf = (x + 0.5f) * scale_x;
    const float gyf = (y + 0.5f) * scale_y;
    // TODO(jiawen): Offset gz by 0.5f as well.
    const float gzf = guide(x, y, b) * grid_depth;

    const int gx0 = static_cast<int>(std::floor(gxf - 0.5f));
    const int gy0 = static_cast<int>(std::floor(gyf - 0.5f));
    const int gz0 = static_cast<int>(std::floor(gzf - 0.5f));

    float vjp_value = 0.0f;
    for (int c = 0; c < grid_channels; ++c) {
      float grid_sample = 0.0f;
      // Grid trilinear interpolation to retrieve grid(gxf, gyf, gzf, c).
      for (int gy = gy0; gy < gy0 + 2; ++gy) {
        const int gyc = std::clamp(gy, 0, grid_height - 1);
        const float wy = LerpWeight(gy + 0.5f, gyf);
        for (int gx = gx0; gx < gx0 + 2; ++gx) {
          const int gxc = std::clamp(gx, 0, grid_width - 1);
          const float wx = LerpWeight(gx + 0.5f, gxf);
          for (int gz = gz0; gz < gz0 + 2; ++gz) {
            const int gzc = std::clamp(gz, 0, grid_depth - 1);
            // TODO(jiawen): Offset gz by 0.5 as well?
            const float dwz =
                grid_depth * SmoothedLerpWeightGrad(gz + 0.5f, gzf);

            grid_sample += wx * wy * dwz * grid(c, gzc, gxc, gyc, b);
          }
        }
      }
      vjp_value += grid_sample * codomain_tangent(c, x, y, b);
    }  // Sum over c.

    guide_vjp_out(x, y, b) = vjp_value;
  }
}

namespace hdrnet {

bool BilateralSliceCudaLauncher(const GpuDevice& device,
                                nda::array_ref_of_rank<const float, 5> grid,
                                nda::array_ref_of_rank<const float, 3> guide,
                                nda::array_ref_of_rank<float, 4> out) {
  const int out_count = out.size();
  if (out_count > 0) {
    // TODO(jiawen): Use GetGpu3DLaunchConfig() to launch a 3D kernel then do a
    // 1D loop over the inner axis.
    const GpuLaunchConfig config = GetGpuLaunchConfig(out_count, device);
    BilateralSliceKernel<<<config.block_count, config.thread_per_block, 0,
                           device.stream()>>>(out_count, grid, guide, out);
  }

  return device.ok();
}

bool BilateralSliceGradCudaLauncher(
    const GpuDevice& device, nda::array_ref_of_rank<const float, 5> grid,
    nda::array_ref_of_rank<const float, 3> guide,
    nda::array_ref_of_rank<const float, 4> codomain_tangent,
    nda::array_ref_of_rank<float, 5> grid_vjp_out,
    nda::array_ref_of_rank<float, 3> guide_vjp_out) {
  const int grid_vjp_count = grid_vjp_out.size();
  if (grid_vjp_count > 0) {
    // TODO(jiawen): Use GetGpu3DLaunchConfig() to launch a 3D kernel then do a
    // 1D loop over the two inner axes.
    const GpuLaunchConfig config = GetGpuLaunchConfig(grid_vjp_count, device);
    BilateralSliceGridGradKernel<<<config.block_count, config.thread_per_block,
                                   0, device.stream()>>>(
        grid_vjp_count, guide, codomain_tangent, grid_vjp_out);
  }

  const int guide_vjp_count = guide_vjp_out.size();
  if (guide_vjp_count > 0) {
    // TODO(jiawen): Use GetGpu3DLaunchConfig() to launch a 3D kernel.
    const GpuLaunchConfig config = GetGpuLaunchConfig(guide_vjp_count, device);
    BilateralSliceGuideGradKernel<<<config.block_count, config.thread_per_block,
                                    0, device.stream()>>>(
        guide_vjp_count, grid, guide, codomain_tangent, guide_vjp_out);
  }

  return device.ok();
}

}  // namespace hdrnet

#endif
