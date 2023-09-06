import torch
from nanosam.mobile_sam import sam_model_registry
import argparse

if __name__ == "__main__":
        
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
        "--model-type",
        type=str,
        required=True,
        help="In ['default', 'vit_h', 'vit_l', 'vit_b']. Which type of SAM model to export.",
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1
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
        mobile_sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
        mobile_sam.to(device=device)
        mobile_sam.eval()

        data = torch.randn(args.batch_size, 3, 1024, 1024).to(device)

        torch.onnx.export(
            mobile_sam.image_encoder,
            (data,),
            args.output,
            input_names=["image"],
            output_names=["image_embeddings"],
            dynamic_axes={
                "image": {0: "batch_size"}
            },
            opset_version=args.opset
        )
