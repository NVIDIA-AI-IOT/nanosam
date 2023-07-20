#!/bin/bash

python scripts/export_mask_decoder_onnx.py --checkpoint ./weights/mobile_sam.pt --model-type vit_t --output ./data/mobile_sam_mask_decoder.onnx

/usr/src/tensorrt/bin/trtexec \
    --onnx=data/mobile_sam_mask_decoder.onnx \
    --saveEngine=data/mobile_sam_mask_decoder.engine \
    --minShapes=point_coords:1x1x2,point_labels:1x1 \
    --optShapes=point_coords:1x1x2,point_labels:1x1 \
    --maxShapes=point_coords:1x10x2,point_labels:1x10