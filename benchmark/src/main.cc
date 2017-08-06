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

#include <algorithm>
#include <cstdio>
#include <iostream>
#include <fstream>

#include <GL/glew.h>
#include <GL/freeglut.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <gflags/gflags.h>

#include "timer.h"
#include "processor.h"
#include "utils.h"

DEFINE_bool(use_gpu, false, "Run computation on gpu.");
DEFINE_int32(burn_iters, 2, "Iterations to run without benchmarking.");
DEFINE_int32(iters, 2, "Benchmark averaging iterations.");
DEFINE_string(output_directory, "", "Destination for the output image and report.");
DEFINE_string(input_path, "", "Path to the input image file (tested on square images only).");
DEFINE_string(checkpoint_path, "", "Path to the network checkpoint");
DEFINE_string(mode, "HDRNetCurves", "Type of network (HDRNetCurves, HDRNetGaussianPyrNN, Direct)");

int main(int argc, char *argv[])
{
  glutInit(&argc, argv);
  google::ParseCommandLineFlags(&argc, &argv, true);

  std::string root = argv[0];
  root = root.substr(0, root.find_last_of("/"));
  root = root.substr(0, root.find_last_of("/"))+"/";

  // Settings
  std::string input_path = FLAGS_input_path;
  std::string checkpoint_path = FLAGS_checkpoint_path;

  if (FLAGS_input_path.empty()) {
      std::cerr << "--input_path is required." << std::endl;
      return 1;
  }
  if (FLAGS_checkpoint_path.empty()) {
      std::cerr << "--checkpoint_path is required." << std::endl;
      return 1;
  }

  std::string model_name = checkpoint_path.substr(
          checkpoint_path.find_last_of("/")+1, checkpoint_path.size()-1);
  checkpoint_path += "/";
  std::string image_output_path =
    FLAGS_output_directory + "/" + model_name + ".png";
  std::string json_output_path =
    FLAGS_output_directory + "/" + model_name + ".json";


  int burn_iters = FLAGS_burn_iters;
  int iters = FLAGS_iters;
  bool use_gpu = FLAGS_use_gpu;

  std::cout << std::endl << "Model: " << model_name << "." << std::endl;

  std::cout << "Loading " << input_path << ".";
  cv::Mat image = load_image(input_path);
  if (!image.data) {
    std::cout << " FAILED" << std::endl;
    return 1;
  }
  int image_width = image.size().width;
  int image_height = image.size().height;
  std::cout <<  image_width << "x" << image_height << std::endl;

  Processor *processor = nullptr;
  if(FLAGS_mode == "HDRNetCurves") {
    processor = new StandardProcessor(
        image_width, image_height, checkpoint_path, use_gpu, root+"assets/");
  } else if (FLAGS_mode == "Direct"){
    processor = new DirectNetProcessor(
        image_width, image_height, checkpoint_path, use_gpu);
  } else if(FLAGS_mode == "HDRNetGaussianPyrNN") {
    int net_input_size = 256;
    processor = new MultiscaleProcessor(
        image_width, image_height, checkpoint_path, use_gpu, root+"assets/");
  } else {
    std::cout << "Unrecognized mode " << FLAGS_mode << std::endl;
    return 1;
  }

  cv::Mat output_rgb(image_height, image_width, CV_8UC3, cv::Scalar(0));

  // Discard first few iterations.
  for (int i = 0; i < burn_iters; ++i) {
    printf("Burning in: iteration %d of %d.\r", i, burn_iters);
    processor->process(image, output_rgb);
  }
  printf("\n");

  // -- Processing ------------------------------
  BenchmarkResult result;
  for (int i = 0; i < iters; ++i) {
    printf("Running actual benchmark: iteration %d of %d.\r", i, iters);
    result = result + processor->process(image, output_rgb);
  }
  printf("\n");

  result /= iters;
  std::cout << std::endl;
  std::cout << "------------------------------" << std::endl;
  std::cout << "Benchmark (" << iters << " iterations)" << std::endl;
  std::cout << "------------------------------" << std::endl;
  std::cout << "Downsampling (CPU, Nearest): " <<
    result.downsampling << " ms" << std::endl;
  std::cout << "Convert input to float: " <<
    result.convert_to_float << " ms" << std::endl;
  std::cout << "Net forward pass: " <<
    result.forward_pass << " ms" << std::endl;
  std::cout << "Rendering (direct): " <<
    result.rendering_direct << " ms" << std::endl;
  std::cout << "Rendering (GL: upload coeffs): " <<
    result.rendering_gl_coeff << " ms" << std::endl;
  std::cout << "Rendering (GL: draw): " <<
    result.rendering_gl_draw << " ms" << std::endl;
  std::cout << "Rendering (GL: readback): " <<
    result.rendering_gl_readback << " ms" << std::endl;
  std::cout << "Total: " << result.total_time() << " ms" << std::endl;
  std::cout << "------------------------------" << std::endl;
  std::cout << std::endl;

  result.save(json_output_path);

  std::cout << "Saving image " << image_output_path << " ";
  cv::Mat output_bgr;
  cv::cvtColor(output_rgb, output_bgr, CV_RGB2BGR, 3);
  cv::imwrite(image_output_path, output_bgr);
  std::cout << "done." << std::endl;


  delete processor;

  return 0;
}
