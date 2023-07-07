# MobileSam TRT

### Export Image Encoder ONNX

```
python scripts/export_image_encoder_onnx.py --checkpoint ./weights/mobile_sam.pt --model-type vit_t --output ./data/mobile_sam_image_encoder.onnx
```

### Export Mask Decoder ONNX

```bash
python scripts/export_mask_decoder_onnx.py --checkpoint ./weights/mobile_sam.pt --model-type vit_t --output ./data/mobile_sam_mask_decoder.onnx
```

### Build image encoder TensorRT engine

```bash
/usr/src/tensorrt/bin/trtexec \
    --onnx=data/mobile_sam_image_encoder.onnx \
    --saveEngine=data/mobile_sam_image_encoder.engine \
    --fp16
```

### Build mask decoder TensorRT engine

```bash
/usr/src/tensorrt/bin/trtexec \
    --onnx=data/mobile_sam_mask_decoder.onnx \
    --saveEngine=data/mobile_sam_mask_decoder.engine \
    --fp16
```