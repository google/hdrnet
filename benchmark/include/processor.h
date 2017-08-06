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

#ifndef PROCESSOR_H_Q1UHFSEZ
#define PROCESSOR_H_Q1UHFSEZ

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/platform/env.h>
#include <tensorflow/core/graph/default_device.h>
#include <iostream>
#include <fstream>

#include "renderer.h"
#include "timer.h"

namespace tf=tensorflow;

typedef struct BenchmarkResult {
  double downsampling = 0.0;
  double convert_to_float = 0.0;
  double forward_pass = 0.0;
  // TODO: HACK: GL mode only populates the gl rendering times.
  // direct mode only populates the direct rendering time.
  // But total_time() works fine since it'll be set to 0.
  double rendering_gl_coeff = 0.0;
  double rendering_gl_draw = 0.0;
  double rendering_gl_readback = 0.0;
  double rendering_direct = 0.0;

  double total_time() {
    return downsampling+convert_to_float+forward_pass
        +rendering_gl_coeff+rendering_gl_draw+rendering_gl_readback
        +rendering_direct;
  }

  BenchmarkResult operator+ (BenchmarkResult other) {
    BenchmarkResult result;
    result.downsampling = downsampling+other.downsampling;
    result.convert_to_float = convert_to_float+other.convert_to_float;
    result.forward_pass = forward_pass+other.forward_pass;
    result.rendering_gl_coeff = rendering_gl_coeff+other.rendering_gl_coeff;
    result.rendering_gl_draw = rendering_gl_draw+other.rendering_gl_draw;
    result.rendering_gl_readback =
      rendering_gl_readback+other.rendering_gl_readback;
    result.rendering_direct = rendering_direct+other.rendering_direct;

    return result;
  }

  BenchmarkResult operator/= (double ratio) {
    downsampling = downsampling/ratio;
    convert_to_float = convert_to_float/ratio;
    forward_pass = forward_pass/ratio;
    rendering_gl_coeff = rendering_gl_coeff/ratio;
    rendering_gl_draw = rendering_gl_draw/ratio;
    rendering_gl_readback = rendering_gl_readback/ratio;
    rendering_direct = rendering_direct/ratio;
  }

  void save (const std::string &filename) {
    std::ofstream file;
    file.open(filename, std::ios::out);
    if(!file) {
      std::cout << "Failed to open file for writing " << filename << std::endl;
      throw;
    }
    file << "{" << std::endl;
    file << "\"downsampling\": " << downsampling << "," << std::endl;
    file << "\"convert_to_float\": " <<convert_to_float << "," <<  std::endl;
    file << "\"forward_pass\": " << forward_pass << "," <<  std::endl;
    file << "\"rendering_gl_coeff\": " << rendering_gl_coeff << "," << std::endl;
    file << "\"rendering_gl_draw\": " << rendering_gl_draw << "," << std::endl;
    file << "\"rendering_gl_readback\": " << rendering_gl_readback << "," << std::endl;
    file << "\"rendering_direct\": " << rendering_direct << std::endl;
    file << "}" << std::endl;
    file.close();
  }

} BenchmarkResult;


class Processor
{
public:
  Processor(int image_width, int image_height, std::string checkpoint_path, bool use_gpu);
  virtual BenchmarkResult process(const cv::Mat &input, cv::Mat &output) = 0;
  virtual ~Processor ();

protected:
  Timer timer_;

  int image_width_;
  int image_height_;
  std::string checkpoint_path_;

  tf::Tensor input_tensor_;
  std::vector<std::pair<std::string, tf::Tensor>> inputs_;
  std::vector<tf::Tensor> outputs_;
  const std::string output_name_ = "output_coefficients";
  const std::string input_name_ = "lowres_input";

  tf::Session *session_;
  tf::GraphDef graph_def_;
};


class HybridGLProcessor : public Processor
{
public:
  explicit HybridGLProcessor(
      int image_width, int image_height,
      std::string checkpoint_path, bool use_gpu, std::string shader_root);
  virtual BenchmarkResult process(const cv::Mat &input, cv::Mat &output) override;
  virtual ~HybridGLProcessor ();

protected:
  const int net_input_size_ = 256;

  std::string vertex_shader_;
  std::string fragment_shader_;

  int grid_width_;
  int grid_height_;
  int grid_depth_;

  Renderer *renderer_;
};

class StandardProcessor : public HybridGLProcessor
{
public:
  explicit StandardProcessor(
      int image_width, int image_height,
      std::string checkpoint_path, bool use_gpu, std::string shader_root);
};


class MultiscaleProcessor : public HybridGLProcessor
{
public:
  explicit MultiscaleProcessor(
      int image_width, int image_height,
      std::string checkpoint_path, bool use_gpu, std::string shader_root);
};


class DirectNetProcessor : public Processor
{
public:
  explicit DirectNetProcessor(int image_width, int image_height, std::string checkpoint_path, bool use_gpu)
    : Processor(image_width, image_height, checkpoint_path, use_gpu) {};
  virtual BenchmarkResult process(const cv::Mat &input, cv::Mat &output) override;

protected:
  tf::Tensor input_tensor_;
  std::vector<std::pair<std::string, tf::Tensor>> inputs_;
  std::vector<tf::Tensor> outputs_;

  Renderer *renderer_;
};

#endif /* end of include guard: PROCESSOR_H_Q1UHFSEZ */

