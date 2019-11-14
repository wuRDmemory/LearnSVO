#!/bin/bash

cd build
make -j6
cd -

./build/LEARN_SVO -s="/home/ubuntu/Documents/DATA/SLAM/rgbd_dataset_freiburg1_xyz/rgb/" -v=true
