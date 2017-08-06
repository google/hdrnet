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

#ifndef TIMER_H_6PNELJKM
#define TIMER_H_6PNELJKM

#include <chrono>

namespace chrono=std::chrono;

class Timer
{
public:
  void start() {
    start_ = chrono::steady_clock::now();
  }

  double duration() {
    return chrono::duration<double, std::milli>(chrono::steady_clock::now() - start_).count();
  }

private:
  chrono::time_point<chrono::steady_clock> start_;
};

#endif /* end of include guard: TIMER_H_6PNELJKM */
