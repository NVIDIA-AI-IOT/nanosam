import tqdm
import torch
import argparse
from nanosam.utils.predictor import load_image_encoder_engine
from nanosam.models import create_model, list_models
from nanosam.datasets.image_folder import ImageFolder
import os
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

if __name__ == "__main__":
        
    parser = argparse.ArgumentParser()
    parser.add_argument("images", type=str, help="The path to images to use for distillation")
    parser.add_argument("output_dir", type=str, help="The directory to store checkpoints and training visualizations.")
    parser.add_argument("--model_name", type=str, default="resnet18", choices=list_models(), help="The NanoSAM model name.")
    parser.add_argument("--student_size", type=int, default=1024, help="The size of image to feed to the student during distillation.")
    parser.add_argument("--num_images", type=int, default=None, help="Limit the number of images per epoch.  Helpful for quick training runs when experimenting.")
    parser.add_argument("--num_epochs", type=int, default=200, help="The number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=16, help="The batch size.")
    parser.add_argument("--num_workers", type=int, default=8, help="The number of data loader workers.")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help='The learning rate.')
    parser.add_argument("--loss", type=str, default="huber", choices=["huber", "l1", "mse"],
        help="The loss function to use for distillation.")
    parser.add_argument("--teacher_image_encoder_engine", 
        type=str, 
        default="data/mobile_sam_image_encoder_bs16.engine",
        help="The path to the image encoder engine to use as a teacher model."
    )
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    image_encoder_trt = load_image_encoder_engine(args.teacher_image_encoder_engine)

    image_encoder_cnn = create_model(args.model_name).cuda()

    if args.loss == "huber":
        loss_function = F.huber_loss
    elif args.loss == "l1":
        loss_function = F.l1_loss
    elif args.loss == "mse":
        loss_function = F.mse_loss
    else:
        raise RuntimeError(f"Unsupported loss function {args.loss}")

    optimizer = torch.optim.Adam(image_encoder_cnn.parameters(), lr=3e-4)

    dataset = ImageFolder(args.images)

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