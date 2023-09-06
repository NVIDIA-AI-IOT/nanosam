import torch
from nanosam.models import create_model
import argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Export the NanoSAM image encoder to an ONNX model."
    )

    parser.add_argument(
        "--model_name", type=str, default="resnet18"
    )

    parser.add_argument(
        "--checkpoint", type=str, required=True, help="The path to the SAM model checkpoint."
    )

    parser.add_argument(
        "--output", type=str, required=True, help="The filename to save the ONNX model to."
    )

    parser.add_argument(
        "--input_size", type=int, default=1024
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

        model = create_model(args.model_name)
        data = torch.randn(1, 3, args.input_size, args.input_size).to(device)
        model.load_state_dict(torch.load(args.checkpoint)["model"])
        model = model.cuda().eval()

        torch.onnx.export(
            model,
            (data,),
            args.output,
            input_names=["image"],
            output_names=["image_embeddings"],
            opset_version=args.opset
        )
