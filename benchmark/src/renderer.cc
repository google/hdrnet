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

#include "renderer.h"

#include <cstdio>
#include <chrono>
#include <thread>

#include "utils.h"

Renderer::Renderer(int output_width, int output_height,
    int grid_width, int grid_height, int grid_depth,
    std::string vertex_shader_path, std::string fragment_shader_path,
    std::string checkpoint_path)
  : output_width_(output_width), output_height_(output_height),
    grid_width_(grid_width), grid_height_(grid_height), grid_depth_(grid_depth)
{
  // Global glInit
  glutInitWindowSize(output_width_, output_height_);
  glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH);
  glutCreateWindow("renderer");
  glutHideWindow();

  GLenum err = glewInit();
  if (err != GLEW_OK) {
    std::cout << "Failed to initialize GLEW: " << err << std::endl;
    throw;
  }

  std::cout << "OpenGL context initialized." << std::endl;
  std::cout << "GL_VENDOR: " << glGetString(GL_VENDOR) << std::endl;
  std::cout << "GL_RENDERER: " << glGetString(GL_RENDERER) << std::endl;
  std::cout << "GL_VERSION: " << glGetString(GL_VERSION) << std::endl;

  std::cout << "Loading fragment shader from: " << fragment_shader_path
    << std::endl;
  vertex_shader_ = glCreateShader(GL_VERTEX_SHADER);
  fragment_shader_ = glCreateShader(GL_FRAGMENT_SHADER);
  shader_from_file(vertex_shader_path, vertex_shader_);
  shader_from_file(fragment_shader_path, fragment_shader_);

  // Create program
  program_ = glCreateProgram();
  glAttachShader(program_, vertex_shader_);
  glAttachShader(program_, fragment_shader_);
  glLinkProgram(program_);
  GLint link_success;
  glGetProgramiv(program_, GL_LINK_STATUS, &link_success);
  if (link_success == GL_FALSE) {
    std::cout << "Failed to link program" << std::endl;
    GLint logSize = 0;
    glGetProgramiv(program_, GL_INFO_LOG_LENGTH, &logSize);
    std::vector<GLchar> errorLog(logSize);
    glGetProgramInfoLog(program_, logSize, &logSize, &errorLog[0]);
    std::cout << errorLog.data() << std::endl;
    glDeleteProgram(program_);
    program_ = 0;
    throw;
  }
  glUseProgram(program_);

  // Geometry
  int ph = glGetAttribLocation(program_, "vPosition");
  int tch = glGetAttribLocation(program_, "vTexCoord");
  glVertexAttribPointer(ph, 2, GL_FLOAT, false, 4*2, static_cast<GLvoid*>(pVertex_));
  glVertexAttribPointer(tch, 2, GL_FLOAT, false, 4*2, static_cast<GLvoid*>(pTexCoord_));
  glEnableVertexAttribArray(ph);
  glEnableVertexAttribArray(tch);

  // Output texture
  glGenTextures(1, &output_texture_);
  glBindTexture(GL_TEXTURE_2D, output_texture_);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, output_width_, output_height_, 0, GL_RGB, GL_UNSIGNED_BYTE, 0);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

  // Input texture
  glGenTextures(1, &input_texture_);
  glBindTexture(GL_TEXTURE_2D, input_texture_);
  glTexStorage2D(GL_TEXTURE_2D, 1 , GL_RGB8, output_width_, output_height_);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glProgramUniform1i(program_, glGetUniformLocation(program_, "sRGB"), 1);

  // Output framebuffer
  glGenFramebuffers(1, &framebuffer_);
  glBindFramebuffer(GL_FRAMEBUFFER, framebuffer_);
  glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, output_texture_, 0);
  GLenum draw_buffers[1] = {GL_COLOR_ATTACHMENT0};
  glDrawBuffers(1, draw_buffers); // "1" is the size of DrawBuffers

  // Always check that our framebuffer is ok
  if(glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
    std::cout << "Frame buffer did not complete operation" << std::endl;
    throw;
  }

  glBindFramebuffer(GL_FRAMEBUFFER, framebuffer_);
  glViewport(0, 0, output_width_, output_height_);

  // Timer queries.
  glGenQueries(kNumQueries, query_ids_);
}

void Renderer::render(const float* const coeffs_data, cv::Mat & output,
        double *upload_coeff_time, double *draw_time, double *readback_time) {
  glQueryCounter(query_ids_[0], GL_TIMESTAMP);

  // Upload coefficient grid to GPU
  upload_coefficients(coeffs_data);

  glQueryCounter(query_ids_[1], GL_TIMESTAMP);

  glClear(GL_COLOR_BUFFER_BIT);
  glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

  glQueryCounter(query_ids_[2], GL_TIMESTAMP);

  glReadPixels(0, 0, output_width_, output_height_,
               GL_RGB, GL_UNSIGNED_BYTE, (GLvoid*) output.data);
  glQueryCounter(query_ids_[3], GL_TIMESTAMP);

  // Wait until all results are available.
  for (int i = 0; i < kNumQueries; ++i) {
    GLint is_available = 0;
    glGetQueryObjectiv(query_ids_[i],
                       GL_QUERY_RESULT_AVAILABLE,
                       &is_available);
    while (is_available == 0) {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
      glGetQueryObjectiv(query_ids_[i],
                         GL_QUERY_RESULT_AVAILABLE,
                         &is_available);
    }
  }

  // Retrieve the timestamps.
  GLuint64 timestamps[kNumQueries];
  for (int i = 0; i < kNumQueries; ++i) {
    glGetQueryObjectui64v(query_ids_[i], GL_QUERY_RESULT, &(timestamps[i]));
  }

  *upload_coeff_time = (timestamps[1] - timestamps[0]) * 1e-6;
  *draw_time = (timestamps[2] - timestamps[1]) * 1e-6;
  *readback_time = (timestamps[3] - timestamps[2]) * 1e-6;
#if 0
  printf("Coefficient upload took %zu ns, %lf ms.\n",
          timestamps[1] - timestamps[0],
          (timestamps[1] - timestamps[0]) * 1e-6);
  printf("Rendering took %zu ns, %lf ms.\n",
          timestamps[2] - timestamps[1],
          (timestamps[2] - timestamps[1]) * 1e-6);
  printf("Readback took %zu ns, %lf ms.\n",
          timestamps[3] - timestamps[2],
          (timestamps[3] - timestamps[2]) * 1e-6);
#endif
}


void Renderer::upload_input(const cv::Mat &image) {
  // Upload input image.
  glActiveTexture(GL_TEXTURE1);
  glBindTexture(GL_TEXTURE_2D, input_texture_);
  glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, output_width_, output_height_,
                  GL_RGB, GL_UNSIGNED_BYTE, image.data);
}

StandardRenderer::StandardRenderer(
    int output_width, int output_height,
    int grid_width, int grid_height, int grid_depth,
    std::string vertex_shader, std::string fragment_shader,
    std::string checkpoint_path) :
  Renderer(output_width, output_height,
      grid_width, grid_height, grid_depth,
      vertex_shader, fragment_shader,
      checkpoint_path)
{
  // Bind affine coefficients to three texture samplers, one row each.
  gl_extra_setup();
  load_guide_parameters(checkpoint_path);
};

void StandardRenderer::load_guide_parameters(std::string checkpoint_path) {
  float ccm[3*4] = {0};
  float mix_matrix[4*1] = {0};
  float shifts[16*3] = {0};
  float slopes[16*3] = {0};

  load_binary_data(checkpoint_path+"guide_ccm_f32_3x4.bin", 3*4, ccm);
  load_binary_data(checkpoint_path+"guide_mix_matrix_f32_1x4.bin", 4, mix_matrix);
  load_binary_data(checkpoint_path+"guide_shifts_f32_16x3.bin", 16*3, shifts);
  load_binary_data(checkpoint_path+"guide_slopes_f32_16x3.bin", 16*3, slopes);

  glProgramUniformMatrix3x4fv(
      program_,
      glGetUniformLocation(program_, "uGuideCcm"),
      1, false, ccm);
  glProgramUniform4fv(
      program_,
      glGetUniformLocation(program_, "uMixMatrix"),
      1, mix_matrix);
  glProgramUniform3fv(
      program_,
      glGetUniformLocation(program_, "uGuideShifts"),
      16, shifts);
  glProgramUniform3fv(
      program_,
      glGetUniformLocation(program_, "uGuideSlopes"),
      16, slopes);
}


void StandardRenderer::gl_extra_setup() {
  glGenTextures(3, coeffs_textures_);
  for (int i = 0; i < 3; ++i) {
    glBindTexture(GL_TEXTURE_3D, coeffs_textures_[i]);
    glTexStorage3D(GL_TEXTURE_3D, 1, GL_RGBA16F, grid_width_, grid_height_, grid_depth_);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  }
  glProgramUniform1i(program_, glGetUniformLocation(program_, "sAffineGridRow0"), 2);
  glProgramUniform1i(program_, glGetUniformLocation(program_, "sAffineGridRow1"), 3);
  glProgramUniform1i(program_, glGetUniformLocation(program_, "sAffineGridRow2"), 4);
}

void StandardRenderer::upload_coefficients(const float* const coeffs_data) {
  for (int i = 0; i < 3; ++i) {
    glActiveTexture(GL_TEXTURE2 + i);
    glBindTexture(GL_TEXTURE_3D, coeffs_textures_[i]);
    glTexSubImage3D(GL_TEXTURE_3D,
            0, // level
            0, 0, 0, // x, y, z
            grid_width_, grid_height_, grid_depth_,
            GL_RGBA, GL_FLOAT, coeffs_data + i*grid_width_*grid_height_*grid_depth_*4);
  }
}

MultiscaleRenderer::MultiscaleRenderer(
    int output_width, int output_height,
    int grid_width, int grid_height, int grid_depth,
    std::string vertex_shader, std::string fragment_shader,
    std::string checkpoint_path) :
  Renderer(output_width, output_height,
      grid_width, grid_height, grid_depth,
      vertex_shader, fragment_shader,
      checkpoint_path)
{
  // Bind affine coefficients to three texture samplers, one row each.
  gl_extra_setup();
  load_guide_parameters(checkpoint_path);
};


void MultiscaleRenderer::load_guide_parameters(std::string checkpoint_path) {
  float *guide_conv1 = new float[4*16*3]();
  float *guide_conv2 = new float[17*1*3]();

  for (int i = 0; i < 3; ++i)
  {
    std::stringstream sstm;
    sstm << "guide_level" << i << "_conv1.bin";
    load_binary_data(checkpoint_path+sstm.str(), 4*16, guide_conv1 + 4*16*i);

    std::stringstream sstm2;
    sstm2 << "guide_level" << i << "_conv2.bin";
    load_binary_data(checkpoint_path+sstm2.str(), 17, guide_conv2 + 17*i);
  }

  glProgramUniform4fv(
      program_,
      glGetUniformLocation(program_, "uGuideConv1"),
      3*16, guide_conv1);

  glProgramUniform1fv(
      program_,
      glGetUniformLocation(program_, "uGuideConv2"),
      3*17, guide_conv2);

  delete[] guide_conv1;
  delete[] guide_conv2;
}


void MultiscaleRenderer::gl_extra_setup() {
  glGenTextures(9, coeffs_textures_);
  for (int i = 0; i < 9; ++i) {
    glBindTexture(GL_TEXTURE_3D, coeffs_textures_[i]);
    glTexStorage3D(GL_TEXTURE_3D, 1, GL_RGBA16F, grid_width_, grid_height_, grid_depth_);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    std::stringstream sstm;
    sstm << "sAffineGridRow" << i;
    glProgramUniform1i(program_, glGetUniformLocation(program_, sstm.str().c_str()), 4+i);
  }

  // Lowres input texture
  glGenTextures(1, &input_texture_ds2_);
  glBindTexture(GL_TEXTURE_2D, input_texture_ds2_);
  glTexStorage2D(GL_TEXTURE_2D, 1 , GL_RGB8, output_width_, output_height_);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glProgramUniform1i(program_, glGetUniformLocation(program_, "sRGBds2"), 2);

  // Lowres input texture
  glGenTextures(1, &input_texture_ds4_);
  glBindTexture(GL_TEXTURE_2D, input_texture_ds4_);
  glTexStorage2D(GL_TEXTURE_2D, 1 , GL_RGB8, output_width_, output_height_);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glProgramUniform1i(program_, glGetUniformLocation(program_, "sRGBds4"), 3);
}


void MultiscaleRenderer::upload_coefficients(const float* const coeffs_data) {
  for (int i = 0; i < 9; ++i) {
    glActiveTexture(GL_TEXTURE4 + i);
    glBindTexture(GL_TEXTURE_3D, coeffs_textures_[i]);
    glTexSubImage3D(GL_TEXTURE_3D,
            0, // level
            0, 0, 0, // x, y, z
            grid_width_, grid_height_, grid_depth_,
            GL_RGBA, GL_FLOAT, coeffs_data + i*grid_width_*grid_height_*grid_depth_*4);
  }
}

void MultiscaleRenderer::upload_input(const cv::Mat &image) {
  // TODO(jiawen): use direct state access.
  // Upload input image
  glActiveTexture(GL_TEXTURE1);
  glBindTexture(GL_TEXTURE_2D, input_texture_);
  glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, output_width_, output_height_,
                  GL_RGB, GL_UNSIGNED_BYTE, image.data);

  cv::Mat input_ds2;
  cv::Mat input_ds4;
  cv::resize(image, input_ds2, cv::Size(0,0), 0.5, 0.5, cv::INTER_LINEAR);
  cv::resize(input_ds2, input_ds4, cv::Size(0,0), 0.5, 0.5, cv::INTER_LINEAR);

  std::cout << "upload dds images" << std::endl;
  std::cout << "size2 " << input_ds2.size() <<  std::endl;
  std::cout << "size4 " << input_ds4.size() <<  std::endl;
  glActiveTexture(GL_TEXTURE2);
  glBindTexture(GL_TEXTURE_2D, input_texture_ds2_);
  glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, input_ds2.size().width, input_ds2.size().height,
                  GL_RGB, GL_UNSIGNED_BYTE, input_ds2.data);


  glActiveTexture(GL_TEXTURE3);
  glBindTexture(GL_TEXTURE_2D, input_texture_ds4_);
  glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, input_ds4.size().width, input_ds4.size().height,
                  GL_RGB, GL_UNSIGNED_BYTE, input_ds4.data);
  // TODO(jiawen): this glFinish is unnecessary.
  glFinish();
}


Renderer::~Renderer() {
  // GL cleanup
  glDeleteQueries(kNumQueries, query_ids_);
  glDetachShader(program_, vertex_shader_);
  glDetachShader(program_, fragment_shader_);
  glDeleteShader(vertex_shader_);
  glDeleteShader(fragment_shader_);
  glDeleteProgram(program_);
}
