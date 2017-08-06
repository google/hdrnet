#!/bin/bash
# encoding: utf-8
# Copyright 2016 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


TF_BASE=$HOME/projects/third_party/tensorflow

CHKPT=$1

echo $CHKPT

CURRENT=`pwd`

cd $TF_BASE
bazel build -c opt --config=cuda tensorflow/tools/graph_transforms:summarize_graph
bazel build -c opt --config=cuda tensorflow/tools/graph_transforms:transform_graph

cd $CURRENT

echo $CHKPT

$TF_BASE/bazel-bin/tensorflow/tools/graph_transforms/summarize_graph \
  --in_graph=$CHKPT/frozen_graph.pb

$TF_BASE/bazel-bin/tensorflow/tools/graph_transforms/transform_graph \
  --in_graph=$CHKPT/frozen_graph.pb \
  --out_graph=$CHKPT/optimized_graph.pb \
  --inputs='lowres_input' \
  --outputs='output_coefficients' \
  --transforms='strip_unused_nodes remove_nodes(op=Identity, op=CheckNumerics) merge_duplicate_nodes fold_constants(ignore_errors=true) fold_batch_norms sort_by_execution_order strip_unused_nodes'

$TF_BASE/bazel-bin/tensorflow/tools/graph_transforms/summarize_graph \
  --in_graph=$CHKPT/optimized_graph.pb
