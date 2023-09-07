# NanoSAM

NanoSAM is a Segment Anything (SAM) model variant that is capable of running in real-time on NVIDIA Jetson platforms with NVIDIA TensorRT.  

*Why NanoSAM?*

While other lightweight SAM architectures exist, like MobileSAM, we find that after TensorRT optimization, these models are still bottlenecked by the image encoder and achieve sub-realtime framerates on NVIDIA Jetson Orin Nano.  NanoSAM is trained by distilling the MobileSAM image
encoder into an architecture that runs an order of magnitude faster on NVIDIA Jetson with little
loss in accuracy. This enables real-time inference and unlocks new applications like turning pre-trained detectors into instance segmentors or performing segmentation based tracking.

## Usage

NanoSAM provides a simple Python interface for inference.  With this interface,
you provide a set of points and point labels, and use these points to predict 
the object mask.

```python3
from nanosam.utils.predictor import Predictor

predictor = Predictor(
    image_encoder="data/resnet18_image_encoder.engine",
    mask_decoder="data/mobile_sam_mask_decoder.engine"
)

image = PIL.Image.open("dog.jpg")

predictor.set_image(image)

mask, _, _ = predictor.predict(np.array([[x, y]]), np.array([1]))
```

The point labels may be

| Point Label | Description |
|:--------------------:|-------------|
| 0 | Background point |
| 1 | Foreground point |
| 2 | Bounding box top-left |
| 3 | Bounding box bottom-right |

The engine files are constructed by downloading the corresponding ONNX files,
and building the engines with TensorRT.  You can also train a new NanoSAM image
encoder by following the [training instructions](#training).

## Performance

<table style="border-top: solid 1px; border-left: solid 1px; border-right: solid 1px; border-bottom: solid 1px">
    <thead>
        <tr>
            <th rowspan=2 style="text-align: center; border-right: solid 1px">Model †</th>
            <th colspan=2 style="text-align: center; border-right: solid 1px">Jetson Orin Nano (ms)</th>
            <th colspan=2 style="text-align: center; border-right: solid 1px">Jetson AGX Orin (ms)</th>
            <th colspan=4 style="text-align: center; border-right: solid 1px">Accuracy (mIoU) ‡</th>
        </tr>
        <tr>
            <th style="text-align: center; border-right: solid 1px">Image Encoder</th>
            <th style="text-align: center; border-right: solid 1px">Full Pipeline</th>
            <th style="text-align: center; border-right: solid 1px">Image Encoder</th>
            <th style="text-align: center; border-right: solid 1px">Full Pipeline</th>
            <th style="text-align: center; border-right: solid 1px">All</th>
            <th style="text-align: center; border-right: solid 1px">Small</th>
            <th style="text-align: center; border-right: solid 1px">Medium</th>
            <th style="text-align: center; border-right: solid 1px">Large</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td style="text-align: center; border-right: solid 1px">MobileSAM</td>
            <td style="text-align: center; border-right: solid 1px"></td>
            <td style="text-align: center; border-right: solid 1px">146</td>
            <td style="text-align: center; border-right: solid 1px">35</td>
            <td style="text-align: center; border-right: solid 1px">39</td>
            <td style="text-align: center; border-right: solid 1px">0.728</td>
            <td style="text-align: center; border-right: solid 1px">0.658</td>
            <td style="text-align: center; border-right: solid 1px">0.759</td>
            <td style="text-align: center; border-right: solid 1px">0.804</td>
        </tr>
        <tr>
            <td style="text-align: center; border-right: solid 1px">NanoSAM (ResNet18)</td>
            <td style="text-align: center; border-right: solid 1px"></td>
            <td style="text-align: center; border-right: solid 1px">27</td>
            <td style="text-align: center; border-right: solid 1px">4.2</td>
            <td style="text-align: center; border-right: solid 1px">8.1</td>
            <td style="text-align: center; border-right: solid 1px">0.706</td>
            <td style="text-align: center; border-right: solid 1px">0.624</td>
            <td style="text-align: center; border-right: solid 1px">0.738</td>
            <td style="text-align: center; border-right: solid 1px">0.796</td>
        </tr>
    </tbody>
</table>

*Notes*

† The MobileSAM image encoder is optimized with FP32 precision because it produced erroneous results when built for FP16 precision with TensorRT.  The NanoSAM image encoder
is built with FP16 precision as we did not notice a significant accuracy degredation.  Both pipelines use the same mask decoder which is built with FP32 precision.  For all models, the accuracy reported uses the same model configuration used to measure latency.

‡ Accuracy is computed by prompting SAM with ground-truth object bounding box annotations from the COCO 2017 validation dataset.  The IoU is then computed between the mask output of the SAM model for the object and the ground-truth COCO segmentation mask for the object.  The mIoU is the average IoU over all objects in the COCO 2017 validation set matching the target object size (small, medium, large).  

## Examples

### Segment from detections

Like other SAM variants, NanoSAM can be used to segment objects given a bounding
box.  We demonstrate this using OWL-ViT for detection.  OWL-ViT is a model
that is capable of open-vocabulary detection.  This allows you to detect objects
given a text prompt.  

For example, below we run NanoSAM on OWL-ViT detections created with the prompt: "A tree" 

<img src="assets/owl_out.png"/>


While OWL-ViT does not run real-time on Jetson Orin Nano (3sec/img), it is nice for experimentation
as it allows you to detect a wide variety of objects.  You could substitute any
other real-time pre-trained object detector to take full advantage of NanoSAM's 
speed.

### Segment from pose

NanoSAM can also be used to segment objects based on foreground and background
points.  Using NanoSAM in conjunction with a real-time human pose estimator,
we're able to easily segment clothing and body parts.  Here we show NanoSAM
predicting segmentation masks for a person detected using [TRTPose](https://github.com/NVIDIA-AI-IOT/trt_pose).
By selecting appropriate keypoints as foreground or background, we can control
which parts we want to segment.

<img src="assets/pose_out.png"/>

### Segment and track (experimental)

## Training

First download the COCO 2017 training images.

> Note: NanoSAM only needs a folder of images.  We choose COCO because it is easy to
> download and coverse diverse real-world scenarios.  You could use your own
> images instead.

```bash
mkdir -p data/coco
cd data/coco
wget http://images.cocodataset.org/zips/train2017.zip
```

Next, extract the images

```bash
unzip train2017.zip
cd ../..
```

Export the MobileSAM image encoder with batch size 16 to ONNX (To be used as teacher model).

```bash
python3 -m nanosam.tools.export_sam_image_encoder_onnx \
    --checkpoint="assets/mobile_sam.pt" \
    --output="data/mobile_sam_image_encoder_bs16.onnx" \
    --model_type=vit_t \
    --batch_size=16
```

Next, build the image encoder TensorRT engine using trtexec

```bash
trtexec \
    --onnx=data/mobile_sam_image_encoder_bs16.onnx \
    --shapes=image:16x3x1024x1024 \
    --saveEngine=data/mobile_sam_image_encoder_bs16.engine
```

Finally, train the NanoSAM model by distilling MobileSAM.

```bash
python3 -m nanosam.tools.train \
    --images=data/coco/train2017 \
    --output_dir=data/models/resnet18 \
    --model_name=resnet18 \
    --teacher_image_encoder_engine=data/mobile_sam_image_encoder_bs16.engine \
    --batch_size=16
```

Once training, visualizations of progress and checkpoints will be saved to
the specified output directory.  You can stop training and resume from the last
saved checkpoint if needed.

For a list of arguments, you can type 

```bash
python3 -m nanosam.tools.train --help
```

## Evaluation

Export the image encoder model to ONNX.

```bash
python3 -m nanosam.tools.export_image_encoder_onnx \
    --model_name=resnet18 \
    --checkpoint="data/models/resnet18/checkpoint.pth" \
    --output="data/resnet18_image_encoder.onnx"
```

Build the image encoder TensorRT engine.

```bash
trtexec \
    --onnx=data/resnet18_image_encoder.onnx \
    --saveEngine=data/resnet18_image_encoder.engine \
    --fp16
```

Export the mask decoder to ONNX

```bash
python -m nanosam.tools.export_sam_mask_decoder_onnx \
    --checkpoint assets/mobile_sam.pt \
    --model-type vit_t \
    --output ./data/mobile_sam_mask_decoder.onnx
```

Build the mask decoder TensorRT engine.

```bash
trtexec \
    --onnx=data/mobile_sam_mask_decoder.onnx \
    --saveEngine=data/mobile_sam_mask_decoder.engine \
    --minShapes=point_coords:1x1x2,point_labels:1x1 \
    --optShapes=point_coords:1x1x2,point_labels:1x1 \
    --maxShapes=point_coords:1x10x2,point_labels:1x10
```

Download the COCO 2017 validation set.

```bash
mkdir -p data/coco
cd data/coco
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
```

Extract the images and annotations

```bash
unzip val2017.zip
unzip annotations_trainval2017.zip
cd ../..
```

Compute the IoU of the mask prediction given the ground truth COCO box,
against the ground truth COCO mask annotation.

```bash
python3 -m nanosam.tools.eval_coco \
    --coco_root=data/coco/val2017 \
    --coco_ann=data/coco/annotations/instances_val2017.json \
    --image_encoder=data/resnet18_image_encoder.engine \
    --mask_decoder=data/mobile_sam_mask_decoder.engine \
    --output=data/resnet18_coco_results.json
```

Compute the average IoU statistics for given filters, like filtering
by object size or category.

```bash
python3 -m nanosam.tools.compute_eval_coco_metrics \
    data/resnet18_coco_results.json \
    --size="all"
```

> For all options type ``python3 -m nanosam.tools.compute_eval_coco_metrics --help``.

To compute the mIoU for a specific category id.

```bash
python3 -m nanosam.tools.compute_eval_coco_metrics \
    data/resnet18_coco_results.json \
    --category_id=1
```


## Acknowledgement

- [SAM](#)
- [MobileSAM](#) 