#!/usr/bin/env sh
set -e

TOOLS=./build/tools

$TOOLS/caffe train \
    --solver=models/info_GAN_mnist/solver.prototxt $@

