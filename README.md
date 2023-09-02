# NanoSAM

NanoSAM is a Segment Anything (SAM) model variant that is capable of running in real-time on NVIDIA Jetson platforms with NVIDIA TensorRT.  

*Why NanoSAM?*

While other lightweight SAM architectures exist, like MobileSAM, we find that after TensorRT optimization, these models are still bottlenecked by the image encoder and achieve sub-realtime framerates.  NanoSAM is trained by distilling the MobileSAM image
encoder into an architecture that runs an order of magnitude faster on NVIDIA Jetson with little
loss in accuracy. This enables real-time inference and unlocks new applications like turning pre-trained detectors into instance segmentors or performing segmentation based tracking.

## Performance

<table style="border-top: solid 1px; border-left: solid 1px; border-right: solid 1px; border-bottom: solid 1px">
    <thead>
        <tr>
            <th rowspan=2 style="text-align: center; border-right: solid 1px">Model †</th>
            <th colspan=2 style="text-align: center; border-right: solid 1px">Jetson Orin Nano (ms)</th>
            <th colspan=2 style="text-align: center; border-right: solid 1px">Jetson AGX Orin (ms)</th>
            <th colspan=3 style="text-align: center; border-right: solid 1px">Accuracy (mIoU) ‡</th>
            <th rowspan=2 style="text-align: center; border-right: solid 1px">Download</th>
        </tr>
        <tr>
            <th style="text-align: center; border-right: solid 1px">Image Encoder</th>
            <th style="text-align: center; border-right: solid 1px">Full Pipeline</th>
            <th style="text-align: center; border-right: solid 1px">Image Encoder</th>
            <th style="text-align: center; border-right: solid 1px">Full Pipeline</th>
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
            <td style="text-align: center; border-right: solid 1px">0.759</td>
            <td style="text-align: center; border-right: solid 1px">0.804</td>
            <td style="text-align: center; border-right: solid 1px"><a href="#">ONNX</a></td>
        </tr>
        <tr>
            <td style="text-align: center; border-right: solid 1px">NanoSAM (ResNet18)</td>
            <td style="text-align: center; border-right: solid 1px"></td>
            <td style="text-align: center; border-right: solid 1px">27</td>
            <td style="text-align: center; border-right: solid 1px">4.2</td>
            <td style="text-align: center; border-right: solid 1px">8.1</td>
            <td style="text-align: center; border-right: solid 1px">0.704</td>
            <td style="text-align: center; border-right: solid 1px">0.738</td>
            <td style="text-align: center; border-right: solid 1px">0.796</td>
            <td style="text-align: center; border-right: solid 1px"><a href="#">ONNX</a></td>
        </tr>
    </tbody>
</table>

*Notes*

† The MobileSAM image encoder is optimized with FP32 precision because it produced erroneous results when built for FP16 precision with TensorRT.  The NanoSAM image encoder
is built with FP16 precision as we did not notice a significant accuracy degredation.  Both pipelines use the same mask decoder which is built with FP32 precision.  For all models, the accuracy reported uses the same model configuration used to measure latency.

‡ Accuracy is computed by prompting SAM with ground-truth object bounding box annotations from the COCO 2017 validation dataset.  The IoU is then computed between the mask output of the SAM model for the object and the ground-truth COCO segmentation mask for the object.  The mIoU is the average IoU over all objects in the COCO 2017 validation set matching the target object size (small, medium, large).  

## Examples

### Detection to segmentation

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

### Pose to segmentation

NanoSAM can also be used to segment objects based on foreground and background
points.  Using NanoSAM in conjunction with a real-time human pose estimator,
we're able to easily segment clothing and body parts.  Here we show NanoSAM
predicting segmentation masks for a person detected using [TRTPose](https://github.com/NVIDIA-AI-IOT/trt_pose).
By selecting appropriate keypoints as foreground or background, we can control
which parts we want to segment.

<img src="assets/pose_out.png"/>

### Segmentation tracking

## Training


## Evaluation

## Acknowledgement

- [SAM](#)
- [MobileSAM](#) 