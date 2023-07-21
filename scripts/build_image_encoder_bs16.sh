#!/bin/bash

python scripts/export_image_encoder_onnx.py \
    --checkpoint ./weights/mobile_sam.pt \
    --model-type vit_t \
    --output ./data/mobile_sam_image_encoder_bs16.onnx \
    --batch_size 16

/usr/src/tensorrt/bin/trtexec \
    --onnx=data/mobile_sam_image_encoder_bs16.onnx \
    --saveEngine=data/mobile_sam_image_encoder_bs16.engine