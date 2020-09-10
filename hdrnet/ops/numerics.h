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

#ifndef HDRNET_OPS_NUMERICS_H_
#define HDRNET_OPS_NUMERICS_H_

#include <algorithm>
#include <cmath>

// TODO(jiawen): Document this elsewhere:
// From LLVM:
// https://releases.llvm.org/3.9.1/docs/CompileCudaWithLLVM.html#detecting-clang-vs-nvcc
// #if GOOGLE_CUDA
// // Set globally by --config=cuda.
// #endif

// #if defined(__clang__) && defined(__CUDA__) && !defined(__CUDA_ARCH__)
// // clang compiling CUDA code, host mode.
// #endif

// #if defined(__clang__) && defined(__CUDA__) && defined(__CUDA_ARCH__)
// // clang compiling CUDA code, device mode.
// #endif

// TODO(jiawen): For some reason, the following doesn't work if we also check
// #if defined(__CUDA_ARCH__). The compiler seems to make two passes over this
// file when compiling *.cu.cc, once with __CUDA_ARCH__ defined and once
// without (as detected by using #pragma message).
#if defined(__clang__) && defined(__CUDA__)
// clang compiling CUDA code, device mode.
#define HDRNET_CUDA_INLINE_FUNC __device__ __host__ __forceinline__
#else
#define HDRNET_CUDA_INLINE_FUNC inline
#endif

// Returns the linear interpolation weight of the sample at integer coordinate
// `x` with respect to a sample at floating point coordinate `xf`.
//
// The integer coordinates `x` are at pixel centers.
// The floating point coordinates `xf` are at pixel edges.
// (OpenGL convention).
//
// This function is:
// - 1 when x = xf - 0.5f (equivalently, when x + 0.5f = xf).
// - 0 when |x - xf| > 0.5.
//
// TODO(jiawen): Move the 0.5 to the caller.
HDRNET_CUDA_INLINE_FUNC float LerpWeight(int x, float xf) {
  const float dx = x + 0.5f - xf;
  const float abs_dx = std::abs(dx);
  return std::max(1.0f - abs_dx, 0.0f);
}

// Computes the coordinates under mirror boundary conditions at x = 0 and
// `extent`.
//
// -3 -> 2
// -2 -> 1
// -1 -> 0
//  0 -> 0
//  1 -> 1
// ...
// extent - 1 -> extent - 1
// extent     -> extent - 1
// extent + 1 -> extent - 2
// extent + 2 -> extent - 3
HDRNET_CUDA_INLINE_FUNC int MirrorBoundary(int x, int extent) {
  if (x < 0) {
    return -x - 1;
  } else if (x >= extent) {
    return 2 * extent - 1 - x;
  } else {
    return x;
  }
}

// A smoothed version of |x| with improved numerical stability.
HDRNET_CUDA_INLINE_FUNC float SmoothedAbs(float x, float eps = 1.0e-8f) {
  return std::sqrt(x * x + eps);
}

// Gradient of SmoothedAbs with respect to x. This is a smoothed version of
// sign(x) with improved numerical stability.
HDRNET_CUDA_INLINE_FUNC float SmoothedAbsGrad(float x, float eps = 1.0e-8f) {
  return x / std::sqrt(x * x + eps);
}

// A smoothed version of `LerpWeight` with gradients more suitable for back
// propagation.
//
// Let f(x, xf) = LerpWeight(x, xf)
//              = max(1 - |x + 0.5 - xf|, 0)
//              = max(1 - |dx|, 0)
//
// f is not smooth when:
// - |dx| is close to 0. We smooth this by replacing |dx| with
//   SmoothedAbs(dx, eps) = sqrt(dx * dx + eps), which has derivative
//   dx / sqrt(dx * dx + eps).
// - |dx| = 1. When smoothed, this happens when dx = sqrt(1 - eps). Like ReLU,
//   We just ignore this (in the implementation below, when the floats are
//   exactly equal, we choose the SmoothedAbsGrad path since it is more useful
//   than returning a 0 gradient).
//
// TODO(jiawen): Move 0.5f to the caller.
HDRNET_CUDA_INLINE_FUNC float SmoothedLerpWeight(int x, float xf,
                                                 float eps = 1.0e-8f) {
  const float dx = x + 0.5f - xf;
  const float abs_dx = SmoothedAbs(dx, eps);
  return std::max(1.0f - abs_dx, 0.0f);
}

// Gradient of `SmoothedLerpWeight`.
// TODO(jiawen): Move 0.5f to the caller.
HDRNET_CUDA_INLINE_FUNC float SmoothedLerpWeightGrad(int x, float xf,
                                                     float eps = 1.0e-8f) {
  const float dx = x + 0.5f - xf;
  const float abs_dx = SmoothedAbs(dx, eps);
  if (abs_dx > 1.0f) {
    // Gradient when the *smoothed* |dx| exceeds 1.
    return 0.0f;
  } else {
    return SmoothedAbsGrad(dx, eps);
  }
}

#undef HDRNET_CUDA_INLINE_FUNC

#endif  // HDRNET_OPS_NUMERICS_H_
