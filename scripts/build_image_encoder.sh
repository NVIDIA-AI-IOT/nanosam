#!/bin/bash

python scripts/export_image_encoder_onnx.py --checkpoint ./weights/mobile_sam.pt --model-type vit_t --output ./data/mobile_sam_image_encoder.onnx

/usr/src/tensorrt/bin/trtexec \
    --onnx=data/mobile_sam_image_encoder.onnx \
    --saveEngine=data/mobile_sam_image_encoder.engine