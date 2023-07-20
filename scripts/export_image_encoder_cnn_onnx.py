import torch

from mobile_sam.modeling.image_encoder_cnn import ImageEncoderCNN

import argparse

parser = argparse.ArgumentParser(
    description="Export the SAM image encoder to an ONNX model."
)

parser.add_argument(
    "--checkpoint", type=str, required=True, help="The path to the SAM model checkpoint."
)

parser.add_argument(
    "--output", type=str, required=True, help="The filename to save the ONNX model to."
)

parser.add_argument(
    "--opset",
    type=int,
    default=16,
    help="The ONNX opset version to use. Must be >=11",
)

args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

with torch.no_grad():

    model = ImageEncoderCNN()
    model.load_state_dict(torch.load(args.checkpoint)["model"])
    model = model.cuda().eval()

    data = torch.randn(1, 3, 512, 512).to(device)

    torch.onnx.export(
        model,
        (data,),
        args.output,
        input_names=["image"],
        output_names=["image_embeddings"],
        opset_version=args.opset
    )
