#!/usr/bin/env bash

# Script to run pwc_net in a separate virtual env.

source ~/pwc_venv/bin/activate
rosrun pwc_net_ros node.py \
    image_in:=/camera/color/image_raw \
    optical_flow_out:=/optical_flow_out \
    optical_flow_hsv_out:=/optical_flow_hsv_out \
    optical_flow_warp_out:=/optical_flow_warp_out