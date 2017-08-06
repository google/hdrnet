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

#include "processor.h"


Processor::Processor(int image_width, int image_height, std::string checkpoint_path, bool use_gpu)
  : image_width_(image_width), image_height_(image_height),
    checkpoint_path_(checkpoint_path)
{
  // Create Session
  tf::Status status = tf::NewSession(tf::SessionOptions(), &session_);
  if (!status.ok()) {
    std::cout << status.ToString() << std::endl;
    throw;
  }

  // Read in the protobuf graph
  status = tf::ReadBinaryProto(
      tf::Env::Default(), checkpoint_path+"optimized_graph.pb", &graph_def_);
  if (!status.ok()) {
    std::cout << status.ToString() << std::endl;
    throw;
  }
  std::string device_name;
  if (use_gpu) {
    device_name = "/gpu:0";
  } else {
    device_name = "/cpu:0";
  }
  std::cout << "Using device \"" << device_name << "\" for inference." <<
    std::endl;
  tf::graph::SetDefaultDevice(device_name, &graph_def_);

  // Add the graph to the session
  status = session_->Create(graph_def_);
  if (!status.ok()) {
    std::cout << status.ToString() << std::endl;
    throw;
  }
}


Processor::~Processor()
{
  session_->Close();
}


HybridGLProcessor::HybridGLProcessor(int image_width, int image_height,
    std::string checkpoint_path, bool use_gpu, std::string shader_root)
    : Processor(image_width, image_height, checkpoint_path, use_gpu)
{
  input_tensor_  = tf::Tensor(
      tf::DT_FLOAT, tf::TensorShape({1, net_input_size_,
      net_input_size_, 3}));
  inputs_ = {
    {input_name_, input_tensor_},
  };

  tf::Status status = session_->Run(inputs_, {output_name_}, {}, &outputs_);
  grid_depth_  = outputs_[0].dim_size(1);
  grid_height_ = outputs_[0].dim_size(2);
  grid_width_ = outputs_[0].dim_size(3);
}


StandardProcessor::StandardProcessor(int image_width, int image_height,
    std::string checkpoint_path, bool use_gpu, std::string shader_root)
    : HybridGLProcessor(image_width, image_height, checkpoint_path, use_gpu, shader_root)
{
  vertex_shader_ = shader_root+"std.vert";
  fragment_shader_ = shader_root+"std.frag";
  renderer_ = new StandardRenderer(image_width, image_height,
      grid_width_, grid_height_, grid_depth_,
      vertex_shader_, fragment_shader_, checkpoint_path);
};


MultiscaleProcessor::MultiscaleProcessor(int image_width, int image_height,
    std::string checkpoint_path, bool use_gpu, std::string shader_root)
    : HybridGLProcessor(image_width, image_height, checkpoint_path, use_gpu, shader_root)
{
  vertex_shader_ = shader_root+"std.vert";
  fragment_shader_ = shader_root+"gpyrnn.frag";
  renderer_ = new MultiscaleRenderer(image_width, image_height,
      grid_width_, grid_height_, grid_depth_,
      vertex_shader_, fragment_shader_, checkpoint_path);
};


BenchmarkResult HybridGLProcessor::process(const cv::Mat &input, cv::Mat &output) {
  // Upload image to GPU while we process the lowres on CPU
  renderer_->upload_input(input);

  BenchmarkResult result;

  // Downsample
  timer_.start();
  cv::Mat input_lowres;
  cv::resize(input, input_lowres, cv::Size(net_input_size_, net_input_size_), 0, 0, cv::INTER_NEAREST);
  result.downsampling = timer_.duration();

  timer_.start();
  float* lowres_data = input_tensor_.flat<float>().data();
  for (int y = 0; y < net_input_size_; ++y)
  for (int x = 0; x < net_input_size_; ++x)
  for (int c = 0; c < 3; ++c) {
    lowres_data[c+3*(x+net_input_size_*y)] = input_lowres.data[c+3*(x+net_input_size_*y)]/255.0f;
  }
  result.convert_to_float = timer_.duration();

  timer_.start();
  tf::Status status = session_->Run(inputs_, {output_name_}, {}, &outputs_);
  if (!status.ok()) {
    std::cout << status.ToString() << std::endl;
    throw;
  }
  result.forward_pass = timer_.duration();

  float* coeffs_data = outputs_[0].flat<float>().data();
  renderer_->render(coeffs_data, output, &(result.rendering_gl_coeff),
          &(result.rendering_gl_draw), &(result.rendering_gl_readback));

  return result;
}

HybridGLProcessor::~HybridGLProcessor()
{
  delete renderer_;
}


BenchmarkResult DirectNetProcessor::process(const cv::Mat &input, cv::Mat &output) {
  BenchmarkResult result;

  // Make input/output tensors
  input_tensor_  = tf::Tensor(tf::DT_FLOAT, tf::TensorShape({1, image_height_,
                              image_width_, 3}));

  // Feed input
  inputs_ = { {input_name_, input_tensor_}, };

  timer_.start();
  // TODO: this could be sped up, but we don't count it
  float* data = input_tensor_.flat<float>().data();
  for (int y = 0; y < image_height_; ++y)
  for (int x = 0; x < image_width_; ++x)
  for (int c = 0; c < 3; ++c) {
    data[c+3*(x+image_width_*y)] = input.data[c+3*(x+image_width_*y)]/255.0f;
  }
  result.convert_to_float = timer_.duration();

  std::cout << "Direct Processor: running inference on low-res stream.\n";
  timer_.start();
  tf::Status status = session_->Run(inputs_, {output_name_}, {}, &outputs_);
  if (!status.ok()) {
    std::cout << status.ToString() << std::endl;
    throw;
  }
  result.forward_pass = timer_.duration();

  std::cout << "Direct Processor: rendering on CPU (as copy).\n";
  timer_.start();
  // TODO: this could be sped up
  float* out_data = outputs_[0].flat<float>().data();
  for (int y = 0; y < image_height_; ++y)
  for (int x = 0; x < image_width_; ++x)
  for (int c = 0; c < 3; ++c) {
    float val =  std::max(std::min(out_data[c+3*(x+image_width_*y)], 1.0f), 0.0f)*255.0f;
    output.data[c+3*(x+image_width_*y)] = val;
  }
  result.convert_to_float += timer_.duration();

  return result;
}
