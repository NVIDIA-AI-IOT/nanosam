#!/bin/bash

python scripts/export_image_encoder_cnn_onnx.py \
    --checkpoint ./data/models/resnet18/checkpoint.pth \
    --output ./data/mobile_sam_image_encoder_cnn.onnx

/usr/src/tensorrt/bin/trtexec \
    --onnx=data/mobile_sam_image_encoder_cnn.onnx \
    --saveEngine=data/mobile_sam_image_encoder_cnn.engine \
    --fp16