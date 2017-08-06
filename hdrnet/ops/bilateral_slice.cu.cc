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

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/util/cuda_kernel_helper.h"

#include "math.h"

#include <iostream>

using namespace tensorflow;

typedef Eigen::GpuDevice GPUDevice;

__device__ float diff_abs(float x) {
  float eps = 1e-8;
  return sqrt(x*x+eps);
}

__device__ float d_diff_abs(float x) {
  float eps = 1e-8;
  return x/sqrt(x*x+eps);
}

__device__ float weight_z(float x) {
  float abx = diff_abs(x);
  return max(1.0f-abx, 0.0f);
}

__device__ float d_weight_z(float x) {
  float abx = diff_abs(x);
  if(abx > 1.0f) {
    return 0.0f;
    // return abx;
  } else {
    return d_diff_abs(x);
  }
}

__global__ void BilateralSliceKernel(
    int64 nthreads,
    const float* grid, const float* guide, 
    const int bs, const int h, const int w, const int chans,
    const int gh, const int gw, const int gd,
    float* out)
{
  // - Samples centered at 0.5.
  // - Repeating boundary conditions

  CUDA_1D_KERNEL_LOOP(idx, nthreads) {
    int c = idx % chans;
    int x = (idx / chans) % w;
    int y = (idx / (chans*w)) % h;
    int b = (idx / (chans*w*h));

    float gx = (x+0.5f)*gw/(1.0f*w);
    float gy = (y+0.5f)*gh/(1.0f*h);
    float gz = guide[x + w*(y + h*b)]*gd;

    int fx = static_cast<int>(floor(gx-0.5f));
    int fy = static_cast<int>(floor(gy-0.5f));
    int fz = static_cast<int>(floor(gz-0.5f));

    int sz = chans;
    int sx = chans*gd;
    int sy = chans*gd*gw;
    int sb = chans*gd*gw*gh;

    float value = 0.0f;
    for (int xx = fx; xx < fx+2; ++xx) {
      int x_ = max(min(xx, gw-1), 0);
      float wx = max(1.0f-abs(xx+0.5-gx), 0.0f);
      for (int yy = fy; yy < fy+2; ++yy)
      {
        int y_ = max(min(yy, gh-1), 0);
        float wy = max(1.0f-abs(yy+0.5-gy), 0.0f);
        for (int zz = fz; zz < fz+2; ++zz)
        {
          int z_ = max(min(zz, gd-1), 0);
          float wz = weight_z(zz+0.5-gz);
          int grid_idx = c + sz*z_ + sx*x_ + sy*y_ + sb*b;
          value += grid[grid_idx]*wx*wy*wz;
        }
      }
    }
    out[idx] = value;
  }
}

__global__ void BilateralSliceGridGradKernel(
    int64 nthreads,
    const float* grid, const float* guide, const float* backprop, 
    const int bs, const int h, const int w, const int chans,
    const int gh, const int gw, const int gd,
    float* out)
{
  CUDA_1D_KERNEL_LOOP(idx, nthreads) {
    int c = idx % chans;
    int gz = (idx / chans) % gd;
    int gx = (idx / (chans*gd)) % gw;
    int gy = (idx / (chans*gd*gw)) % gh;
    int b = (idx / (chans*gd*gw*gh));

    float scale_w = w*1.0/gw;
    float scale_h = h*1.0/gh;

    int left_x = static_cast<int>(floor(scale_w*(gx+0.5-1)));
    int right_x = static_cast<int>(ceil(scale_w*(gx+0.5+1)));
    int left_y = static_cast<int>(floor(scale_h*(gy+0.5-1)));
    int right_y = static_cast<int>(ceil(scale_h*(gy+0.5+1)));

    int sx = chans;
    int sy = chans*w;
    int sb = chans*w*h;

    float value = 0.0f;
    for (int x = left_x; x < right_x; ++x)
    {
      int x_ = x;

      // mirror boundary
      if (x_ < 0) x_ = -x_-1;
      if (x_ >= w) x_ = 2*w-1-x_;

      // x_ = max(min(x_, w-1), 0);
      float gx2 = (x+0.5f)/scale_w;
      float wx = max(1.0f-abs(gx+0.5-gx2), 0.0f);

      for (int y = left_y; y < right_y; ++y)
      {
        int y_ = y;

        // mirror boundary
        if (y_ < 0) y_ = -y_-1;
        if (y_ >= h) y_ = 2*h-1-y_;

        // y_ = max(min(y_, h-1), 0);
        float gy2 = (y+0.5f)/scale_h;
        float wy = max(1.0f-abs(gy+0.5-gy2), 0.0f);

        int guide_idx = x_ + w*y_ + h*w*b;
        float gz2 = guide[guide_idx]*gd;
        // float wz = max(1.0f-diff_abs(gz+0.5f - gz2), 0.0f);
        float wz = weight_z(gz+0.5f-gz2);
        if ((gz==0 && gz2<0.5f) || (gz==gd-1 && gz2>gd-0.5f)) {
          wz = 1.0f;
        }

        int back_idx = c + sx*x_ + sy*y_ + sb*b;
        value += wz*wx*wy*backprop[back_idx];
      }
    }
    out[idx] = value;
  }
}

__global__ void BilateralSliceGuideGradKernel(
    int64 nthreads,
    const float* grid, const float* guide, const float* backprop, 
    const int bs, const int h, const int w, const int chans,
    const int gh, const int gw, const int gd,
    float* out)
{
  CUDA_1D_KERNEL_LOOP(idx, nthreads) {
    int x = idx  % w;
    int y = (idx / w) % h;
    int b = (idx / (w*h));

    float gx = (x+0.5f)*gw/(1.0f*w);
    float gy = (y+0.5f)*gh/(1.0f*h);
    float gz = guide[x + w*(y + h*b)]*gd;

    int fx = static_cast<int>(floor(gx-0.5f));
    int fy = static_cast<int>(floor(gy-0.5f));
    int fz = static_cast<int>(floor(gz-0.5f));

    int sz = chans;
    int sx = chans*gd;
    int sy = chans*gd*gw;
    int sb = chans*gd*gw*gh;

    float value = 0.0f;
    for (int c = 0; c < chans; ++c) {
      float chan_val = 0.0f;
      for (int xx = fx; xx < fx+2; ++xx) {
        int x_ = max(min(xx, gw-1), 0);
        float wx = max(1.0f-abs(xx+0.5-gx), 0.0f);
        for (int yy = fy; yy < fy+2; ++yy)
        {
          int y_ = max(min(yy, gh-1), 0);
          float wy = max(1.0f-abs(yy+0.5-gy), 0.0f);
          for (int zz = fz; zz < fz+2; ++zz)
          {
            int z_ = max(min(zz, gd-1), 0);
            float dwz = gd*d_weight_z(zz+0.5-gz);

            int grid_idx = c + sz*z_ + sx*x_ + sy*y_ + sb*b;
            chan_val += grid[grid_idx]*wx*wy*dwz;
          }
        }
      }
      chan_val *= backprop[c + chans*(x + w*(y + h*b))];
      value += chan_val;
    }
    out[idx] = value;
  }
}

__global__ void BilateralSliceApplyKernel(
    int64 nthreads,
    const float* grid, const float* guide, const float* input,
    const int bs, const int h, const int w, 
    const int gh, const int gw, const int gd,
    const int input_chans, const int output_chans, const bool has_offset,
    float* out)
{
  // - Samples centered at 0.5.
  // - Repeating boundary conditions

  int grid_chans = input_chans*output_chans;
  int coeff_stride = input_chans;
  if(has_offset) {
    grid_chans += output_chans;
    coeff_stride += 1;
  }

  CUDA_1D_KERNEL_LOOP(idx, nthreads) {
    int out_c = idx % output_chans;
    int x = (idx / output_chans) % w;
    int y = (idx / (output_chans*w)) % h;
    int b = (idx / (output_chans*w*h));

    float gx = (x+0.5f)*gw/(1.0f*w);
    float gy = (y+0.5f)*gh/(1.0f*h);
    float gz = guide[x + w*(y + h*b)]*gd;

    int fx = static_cast<int>(floor(gx-0.5f));
    int fy = static_cast<int>(floor(gy-0.5f));
    int fz = static_cast<int>(floor(gz-0.5f));


    // Grid strides
    int sz = grid_chans;
    int sx = grid_chans*gd;
    int sy = grid_chans*gd*gw;
    int sb = grid_chans*gd*gw*gh;

    float value = 0.0f;
    for (int in_c = 0; in_c < coeff_stride; ++in_c) {
      float coeff_sample = 0.0f;
      for (int xx = fx; xx < fx+2; ++xx) {
        int x_ = max(min(xx, gw-1), 0);
        float wx = max(1.0f-abs(xx+0.5-gx), 0.0f);
        for (int yy = fy; yy < fy+2; ++yy)
        {
          int y_ = max(min(yy, gh-1), 0);
          float wy = max(1.0f-abs(yy+0.5-gy), 0.0f);
          for (int zz = fz; zz < fz+2; ++zz)
          {
            int z_ = max(min(zz, gd-1), 0);
            float wz = weight_z(zz+0.5-gz);
            int grid_idx = (coeff_stride*out_c + in_c) + sz*z_ + sx*x_ + sy*y_ + sb*b;
            coeff_sample += grid[grid_idx]*wx*wy*wz;
          }
        }
      } // Grid trilinear interpolation
      if(in_c < input_chans) {
        int input_idx = in_c + input_chans*(x + w*(y + h*b));
        value += coeff_sample*input[input_idx];
      } else { // Offset term
        value += coeff_sample;
      }
    }
    out[idx] = value;
  }
}


__global__ void BilateralSliceApplyGridGradKernel(
    int64 nthreads,
    const float* grid, const float* guide, const float* input, const float* backprop, 
    const int bs, const int h, const int w, 
    const int gh, const int gw, const int gd,
    const int input_chans, const int output_chans, const bool has_offset,
    float* out)
{
  int grid_chans = input_chans*output_chans;
  int coeff_stride = input_chans;
  if(has_offset) {
    grid_chans += output_chans;
    coeff_stride += 1;
  }

  CUDA_1D_KERNEL_LOOP(idx, nthreads) {
    int c = idx % grid_chans;
    int gz = (idx / grid_chans) % gd;
    int gx = (idx / (grid_chans*gd)) % gw;
    int gy = (idx / (grid_chans*gd*gw)) % gh;
    int b = (idx / (grid_chans*gd*gw*gh));

    float scale_w = w*1.0/gw;
    float scale_h = h*1.0/gh;

    int left_x = static_cast<int>(floor(scale_w*(gx+0.5-1)));
    int right_x = static_cast<int>(ceil(scale_w*(gx+0.5+1)));
    int left_y = static_cast<int>(floor(scale_h*(gy+0.5-1)));
    int right_y = static_cast<int>(ceil(scale_h*(gy+0.5+1)));

    // Strides in the output
    int sx = output_chans;
    int sy = output_chans*w;
    int sb = output_chans*w*h;
    
    // Strides in the input
    int isx = input_chans;
    int isy = input_chans*w;
    int isb = input_chans*w*h;

    int out_c = c / coeff_stride;
    int in_c = c % coeff_stride;

    float value = 0.0f;
    for (int x = left_x; x < right_x; ++x)
    {
      int x_ = x;

      // mirror boundary
      if (x_ < 0) x_ = -x_-1;
      if (x_ >= w) x_ = 2*w-1-x_;

      float gx2 = (x+0.5f)/scale_w;
      float wx = max(1.0f-abs(gx+0.5-gx2), 0.0f);

      for (int y = left_y; y < right_y; ++y)
      {
        int y_ = y;

        // mirror boundary
        if (y_ < 0) y_ = -y_-1;
        if (y_ >= h) y_ = 2*h-1-y_;

        float gy2 = (y+0.5f)/scale_h;
        float wy = max(1.0f-abs(gy+0.5-gy2), 0.0f);

        int guide_idx = x_ + w*y_ + h*w*b;
        float gz2 = guide[guide_idx]*gd;
        float wz = weight_z(gz+0.5f-gz2);
        if ((gz==0 && gz2<0.5f) || (gz==gd-1 && gz2>gd-0.5f)) {
          wz = 1.0f;
        }

        int back_idx = out_c + sx*x_ + sy*y_ + sb*b;
        if (in_c < input_chans) {
          int input_idx = in_c + isx*x_ + isy*y_ + isb*b;
          value += wz*wx*wy*backprop[back_idx]*input[input_idx];
        } else { // offset term
          value += wz*wx*wy*backprop[back_idx];
        }
      }
    }
    out[idx] = value;
  }
}


__global__ void BilateralSliceApplyGuideGradKernel(
    int64 nthreads,
    const float* grid, const float* guide, const float* input, const float* backprop, 
    const int bs, const int h, const int w,
    const int gh, const int gw, const int gd,
    const int input_chans, const int output_chans, const bool has_offset,
    float* out)
{

  int grid_chans = input_chans*output_chans;
  int coeff_stride = input_chans;
  if(has_offset) {
    grid_chans += output_chans;
    coeff_stride += 1;
  }

  CUDA_1D_KERNEL_LOOP(idx, nthreads) {
    int x = idx  % w;
    int y = (idx / w) % h;
    int b = (idx / (w*h));

    float gx = (x+0.5f)*gw/(1.0f*w);
    float gy = (y+0.5f)*gh/(1.0f*h);
    float gz = guide[x + w*(y + h*b)]*gd;

    int fx = static_cast<int>(floor(gx-0.5f));
    int fy = static_cast<int>(floor(gy-0.5f));
    int fz = static_cast<int>(floor(gz-0.5f));

    // Grid stride 
    int sz = grid_chans;
    int sx = grid_chans*gd;
    int sy = grid_chans*gd*gw;
    int sb = grid_chans*gd*gw*gh;

    float out_sum = 0.0f;
    for (int out_c = 0; out_c < output_chans; ++out_c) {

      float in_sum = 0.0f;
      for (int in_c = 0; in_c < coeff_stride; ++in_c) {

        float grid_sum = 0.0f;
        for (int xx = fx; xx < fx+2; ++xx) {
          int x_ = max(min(xx, gw-1), 0);
          float wx = max(1.0f-abs(xx+0.5-gx), 0.0f);
          for (int yy = fy; yy < fy+2; ++yy)
          {
            int y_ = max(min(yy, gh-1), 0);
            float wy = max(1.0f-abs(yy+0.5-gy), 0.0f);
            for (int zz = fz; zz < fz+2; ++zz)
            {
              int z_ = max(min(zz, gd-1), 0);
              float dwz = gd*d_weight_z(zz+0.5-gz);

              int grid_idx = (coeff_stride*out_c + in_c) + sz*z_ + sx*x_ + sy*y_ + sb*b;
              grid_sum += grid[grid_idx]*wx*wy*dwz;
            } // z
          } // y
        } // x, grid trilinear interp

        if(in_c < input_chans) {
          in_sum += grid_sum*input[in_c + input_chans*(x + w*(y + h*b))];
        } else {  // offset term
          in_sum += grid_sum;
        }
      } // in_c

      out_sum += in_sum*backprop[out_c + output_chans*(x + w*(y + h*b))];
    } // out_c

    out[idx] = out_sum;
  }
}


__global__ void BilateralSliceApplyInputGradKernel(
    int64 nthreads,
    const float* grid, const float* guide, const float* input, const float* backprop, 
    const int bs, const int h, const int w,
    const int gh, const int gw, const int gd,
    const int input_chans, const int output_chans, const bool has_offset,
    float* out)
{
  int grid_chans = input_chans*output_chans;
  int coeff_stride = input_chans;
  if(has_offset) {
    grid_chans += output_chans;
    coeff_stride += 1;
  }

  CUDA_1D_KERNEL_LOOP(idx, nthreads) {
    int in_c = idx % input_chans;
    int x = (idx / input_chans) % w;
    int y = (idx / (input_chans*w)) % h;
    int b = (idx / (input_chans*w*h));

    float gx = (x+0.5f)*gw/(1.0f*w);
    float gy = (y+0.5f)*gh/(1.0f*h);
    float gz = guide[x + w*(y + h*b)]*gd;

    int fx = static_cast<int>(floor(gx-0.5f));
    int fy = static_cast<int>(floor(gy-0.5f));
    int fz = static_cast<int>(floor(gz-0.5f));

    // Grid stride 
    int sz = grid_chans;
    int sx = grid_chans*gd;
    int sy = grid_chans*gd*gw;
    int sb = grid_chans*gd*gw*gh;

    float value = 0.0f;
    for (int out_c = 0; out_c < output_chans; ++out_c) {
      float chan_val = 0.0f;
      for (int xx = fx; xx < fx+2; ++xx) {
        int x_ = max(min(xx, gw-1), 0);
        float wx = max(1.0f-abs(xx+0.5-gx), 0.0f);
        for (int yy = fy; yy < fy+2; ++yy)
        {
          int y_ = max(min(yy, gh-1), 0);
          float wy = max(1.0f-abs(yy+0.5-gy), 0.0f);
          for (int zz = fz; zz < fz+2; ++zz)
          {

            int z_ = max(min(zz, gd-1), 0);

            float wz = weight_z(zz+0.5-gz);

            int grid_idx = (coeff_stride*out_c + in_c) + sz*z_ + sx*x_ + sy*y_ + sb*b;
            chan_val += grid[grid_idx]*wx*wy*wz;
          } // z
        } // y
      } // x, grid trilinear interp

      value += chan_val*backprop[out_c + output_chans*(x + w*(y + h*b))];
    } // out_c
    out[idx] = value;
  }
}


// -- KERNEL LAUNCHERS ---------------------------------------------------------
bool BilateralSliceKernelLauncher(
    const GPUDevice& d,
    int bs, int gh, int gw, int gd, int chans,
    int h, int w,
    const float* const grid, const float* const guide, float* const out)
{
  int total_count = bs*h*w*chans;
  if (total_count > 0) {
    CudaLaunchConfig config = GetCudaLaunchConfig(total_count, d);
    BilateralSliceKernel<<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
        total_count, grid, guide, 
        bs, h, w, chans, gh, gw, gd,
        out);
  }

  return d.ok();
}

bool BilateralSliceGradKernelLauncher(
    const GPUDevice& d,
    const float* grid, const int64* grid_size,
    const float* guide, const int64* guide_size,
    const float* backprop,
    float* grid_grad, float* guide_grad)
{
  int64 bs = grid_size[0];
  int64 gh = grid_size[1];
  int64 gw = grid_size[2];
  int64 gd = grid_size[3];
  int64 chans = grid_size[4];

  int64 h = guide_size[1];
  int64 w = guide_size[2];

  int64 grid_count = bs*gh*gw*gd*chans;
  if (grid_count > 0) {
    CudaLaunchConfig config = GetCudaLaunchConfig(grid_count, d);
    BilateralSliceGridGradKernel<<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
        grid_count, grid, guide, backprop,
        bs, h, w, chans, gh, gw, gd,
        grid_grad);
  }

  int64 guide_count = bs*h*w;
  if (guide_count > 0) {
    CudaLaunchConfig config = GetCudaLaunchConfig(guide_count, d);
    BilateralSliceGuideGradKernel<<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
        guide_count, grid, guide, backprop,
        bs, h, w, chans, gh, gw, gd,
        guide_grad);
  }

  return d.ok();
}

bool BilateralSliceApplyKernelLauncher(
    const GPUDevice& d,
    int bs, int gh, int gw, int gd, 
    int input_chans, int output_chans, bool has_offset,
    int h, int w,
    const float* const grid, const float* const guide, const float* const input,
    float* const out)
{
  int total_count = bs*h*w*output_chans;
  if (total_count > 0) {
    CudaLaunchConfig config = GetCudaLaunchConfig(total_count, d);
    BilateralSliceApplyKernel<<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
        total_count, grid, guide, input,
        bs, h, w, gh, gw, gd, input_chans, output_chans, has_offset,
        out);
  }

  return d.ok();
}


bool BilateralSliceApplyGradKernelLauncher(
    const GPUDevice& d,
    const float* grid, const int64* grid_size,
    const float* guide, const int64* guide_size,
    const float* input, const int64* input_size,
    const float* backprop,
    bool has_offset,
    float* grid_grad, float* guide_grad, float* input_grad)
{
  int64 gh = grid_size[1];
  int64 gw = grid_size[2];
  int64 gd = grid_size[3];
  int64 coeff_chans = grid_size[4];
  int64 bs = guide_size[0];
  int64 h = guide_size[1];
  int64 w = guide_size[2];
  int64 input_chans = input_size[3];

  int64 output_chans = 0;
  if (has_offset) {
    output_chans = coeff_chans/(input_chans+1);
  } else {
    output_chans = coeff_chans/input_chans;
  }


  int64 grid_count = bs*gh*gw*gd*coeff_chans;
  if (grid_count > 0) {
    CudaLaunchConfig config = GetCudaLaunchConfig(grid_count, d);
    BilateralSliceApplyGridGradKernel<<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
        grid_count, grid, guide, input, backprop,
        bs, h, w, gh, gw, gd,
        input_chans, output_chans, has_offset,
        grid_grad);
  }

  int64 guide_count = bs*h*w;
  if (guide_count > 0) {
    CudaLaunchConfig config = GetCudaLaunchConfig(guide_count, d);
    BilateralSliceApplyGuideGradKernel<<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
        guide_count, grid, guide, input, backprop,
        bs, h, w, gh, gw, gd,
        input_chans, output_chans, has_offset,
        guide_grad);
  }

  int64 input_count = bs*h*w*input_chans;
  if (input_count > 0) {
    CudaLaunchConfig config = GetCudaLaunchConfig(input_count, d);
    BilateralSliceApplyInputGradKernel<<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
        input_count, grid, guide, input, backprop,
        bs, h, w, gh, gw, gd,
        input_chans, output_chans, has_offset,
        input_grad);
  }

  return d.ok();
}

#endif
