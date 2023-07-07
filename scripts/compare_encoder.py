import PIL.Image
import numpy as np
import torch
import matplotlib.pyplot as plt

import tensorrt as trt
from torch2trt import TRTModule

from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

inference_size = 1024
image_mean = torch.tensor([123.675, 116.28, 103.53])[:, None, None]
image_std = torch.tensor([58.395, 57.12, 57.375])[:, None, None]

image_pil = PIL.Image.open("assets/dog.jpg")
aspect_ratio = image_pil.width / image_pil.height
if aspect_ratio >= 1:
    resize_width = inference_size
    resize_height = int(inference_size / aspect_ratio)
else:
    resize_height = inference_size
    resize_width = int(inference_size * aspect_ratio)

image_pil_resized = image_pil.resize((resize_width, resize_height))
image_np_resized = np.asarray(image_pil_resized)
image_torch_resized = torch.from_numpy(image_np_resized).permute(2, 0, 1)
image_torch_resized_normalized = (image_torch_resized.float() - image_mean) / image_std
image_tensor = torch.zeros((1, 3, 1024, 1024))
image_tensor[0, :, :resize_height, :resize_width] = image_torch_resized_normalized

image_tensor = image_tensor.cuda()


with trt.Logger() as logger, trt.Runtime(logger) as runtime:
    with open("data/mobile_sam_image_encoder.engine", 'rb') as f:
        engine_bytes = f.read()
    engine = runtime.deserialize_cuda_engine(engine_bytes)

input_name = engine.get_binding_name(0)
output_name = engine.get_binding_name(1)

image_encoder_trt = TRTModule(
    engine=engine,
    input_names=[engine.get_binding_name(0)],
    output_names=[engine.get_binding_name(1)]
)
print("Executing trt encoder")
output_trt = image_encoder_trt(image_tensor)


mobile_sam = sam_model_registry["vit_t"](checkpoint="weights/mobile_sam.pt")
mobile_sam.to(device="cuda")
mobile_sam = mobile_sam.eval()
image_encoder = mobile_sam.image_encoder

print("Executing original encoder")
output = image_encoder(image_tensor)

print("Max abs error:")
print(torch.max(torch.abs(output_trt - output)))