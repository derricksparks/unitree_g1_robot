#!/usr/bin/env python3
"""Quick DDS camera test"""
import sys
import numpy as np
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.unitree_go.msg.dds_ import AudioData_

def rgb_handler(msg):
    print(f"RGB: {len(msg.data)} bytes")

def depth_handler(msg):
    print(f"Depth: {len(msg.data)} bytes")

ChannelFactoryInitialize(0, "lo")
rgb_sub = ChannelSubscriber("rt/camera/rgb", AudioData_)
depth_sub = ChannelSubscriber("rt/camera/depth", AudioData_)
rgb_sub.Init(rgb_handler, 10)
depth_sub.Init(depth_handler, 10)

print("Waiting for camera data...")
import time
time.sleep(10)
print("Done")
