import tqdm
import torch
import tensorrt as trt
import argparse
from torch2trt import TRTModule
import torch.nn as nn
from typing import Tuple
import timm
from mobile_sam.predictor_trt import load_image_encoder_engine
from mobile_sam.modeling.image_encoder_cnn import (
    ImageEncoderCNN_256,
    ImageEncoderCNN_512,
    ImageEncoderCNN_1024
)
from mobile_sam.image_dataset import ImageDataset
import glob
from typing import Optional
import os
import matplotlib.pyplot as plt
from torchvision.transforms import RandomResizedCrop, ToTensor, Normalize, Compose
import PIL.Image
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from pytorch_msssim import ms_ssim

parser = argparse.ArgumentParser()
parser.add_argument("images", type=str)
parser.add_argument("output_dir", type=str)
parser.add_argument("--student_size", type=int, default=512)
parser.add_argument("--model_name", type=str, default="resnet18")
parser.add_argument("--num_images", type=int, default=None)
parser.add_argument("--num_epochs", type=int, default=200)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--num_workers", type=int, default=8)
parser.add_argument("--learning_rate", type=float, default=3e-4)
parser.add_argument("--loss", type=str, default="huber")
parser.add_argument("--teacher_image_encoder_engine", type=str, default="data/mobile_sam_image_encoder_bs16.engine")
args = parser.parse_args()

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

image_encoder_trt = load_image_encoder_engine(args.teacher_image_encoder_engine)

if args.student_size == 512:
    image_encoder_cnn = ImageEncoderCNN_512(model_name=args.model_name, pretrained=True).cuda()
elif args.student_size == 1024:
    image_encoder_cnn = ImageEncoderCNN_1024(model_name=args.model_name, pretrained=True).cuda()
elif args.student_size == 256:
    image_encoder_cnn = ImageEncoderCNN_256(model_name=args.model_name, pretrained=True).cuda()

if args.loss == "huber":
    loss_function = F.huber_loss
elif args.loss == "l1":
    loss_function = F.l1_loss
elif args.loss == "mse":
    loss_function = F.mse_loss
else:
    raise RuntimeError(f"Unsupported loss function {args.loss}")

optimizer = torch.optim.Adam(image_encoder_cnn.parameters(), lr=3e-4)

dataset = ImageDataset("data/coco/train2017/")

if args.num_images is not None:
    dataset, _ = random_split(dataset, [args.num_images, len(dataset) - args.num_images])

loader = DataLoader(dataset, shuffle=True, batch_size=args.batch_size, num_workers=args.num_workers)

checkpoint_path = os.path.join(args.output_dir, "checkpoint.pth")

if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    image_encoder_cnn.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    start_epoch = checkpoint['epoch'] + 1
else:
    start_epoch = 0

for epoch in range(start_epoch, args.num_epochs):
    epoch_loss = 0.
    for image in tqdm.tqdm(iter(loader)):
        image = image.cuda()
        if len(image) != args.batch_size:
            continue
        image_cnn = F.interpolate(image, (args.student_size, args.student_size), mode="area")
        with torch.no_grad():
            features = image_encoder_trt(image)

        optimizer.zero_grad()
        output = image_encoder_cnn(image_cnn)

        loss = loss_function(output, features)

        loss.backward()
        optimizer.step()
        epoch_loss += float(loss)
    epoch_loss /= len(loader)
    print(f"{epoch} - {epoch_loss}")

    with open(os.path.join(args.output_dir, 'log.txt'), 'a') as f:
        f.write(f"{epoch} - {epoch_loss}\n")

    torch.save({
        "model": image_encoder_cnn.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch}, checkpoint_path)
        
    plt.figure(figsize=(10, 10))
    plt.subplot(121)
    plt.imshow(features[0, 0].detach().cpu())
    plt.subplot(122)
    plt.imshow(output[0, 0].detach().cpu())
    plt.savefig(os.path.join(args.output_dir, f"epoch_{epoch}.png"))
    plt.close()