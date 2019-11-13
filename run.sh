#!/bin/bash

cd build
make -j6
cd -

./build/LEARN_SVO -s="/home/ubuntu/data/rgbd_dataset_freiburg1_xyz/rgb/" -v=false
