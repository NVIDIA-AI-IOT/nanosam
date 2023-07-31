#!/bin/bash

python scripts/export_image_encoder_cnn_onnx.py \
    --checkpoint ./data/models/resnet18_huber_1024_v5/checkpoint.pth \
    --output ./data/resnet18_huber_1024_v5.onnx \
    --size 1024

/usr/src/tensorrt/bin/trtexec \
    --onnx=data/resnet18_huber_1024_v5.onnx \
    --saveEngine=data/resnet18_huber_1024_v5.engine \
    --fp16