#!/bin/bash

python scripts/export_image_encoder_cnn_onnx.py \
    --checkpoint ./data/models/resnet18_1024_l1/checkpoint.pth \
    --output ./data/mobile_sam_image_encoder_cnn_1024.onnx \
    --size 1024

/usr/src/tensorrt/bin/trtexec \
    --onnx=data/mobile_sam_image_encoder_cnn_1024.onnx \
    --saveEngine=data/mobile_sam_image_encoder_cnn_1024.engine \
    --fp16