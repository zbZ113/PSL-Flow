import argparse
import os

import numpy as np
from PIL import Image

import torch
from torch import nn
from torchvision import transforms
from tqdm import tqdm
import cv2

from models import build_tevnet
from utils import TeVloss

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"


def save_image(image_array, path):
    Image.fromarray(image_array.astype(np.uint8)).save(path)


def load_state_dict_flexible(model: nn.Module, weights_file: str, device: torch.device):
    ckpt = torch.load(weights_file, map_location=device)
    state = ckpt.get("state_dict", ckpt)
    cleaned = {}
    for k, v in state.items():
        key = k[len("module.") :] if k.startswith("module.") else k
        cleaned[key] = v
    model.load_state_dict(cleaned, strict=False)


def load_model(weights_file, args, device):
    model = build_tevnet(args, in_channels=3, out_channels=2 + args.vnums).to(device)
    if torch.cuda.device_count() > 1 and not args.no_dp:
        model = nn.DataParallel(model)
    m = model.module if isinstance(model, nn.DataParallel) else model
    load_state_dict_flexible(m, weights_file, device)
    model.eval()
    return model


def process_image(image_path, device):
    image = Image.open(image_path).convert("RGB")
    return transforms.ToTensor()(image).unsqueeze(0).to(device)


def save_decomposed_images(preds, input_tensor, output_img_dir, img_name, lossmodule):
    preds_np = preds.cpu().numpy().squeeze(0)
    rec = lossmodule.rec(preds, input_tensor).mul(255.0).cpu().numpy().squeeze(0).squeeze(0)
    e = lossmodule.rec_e(preds).mul(255.0).cpu().numpy().squeeze(0).squeeze(0)
    T = lossmodule.rec_T(preds).mul(255.0).cpu().numpy().squeeze(0).squeeze(0)
    env = (
        lossmodule.rec_env(preds, torch.mean(input_tensor, dim=1))
        .mul(255.0)
        .cpu()
        .numpy()
        .squeeze(0)
        .squeeze(0)
    )

    T = cv2.normalize(T, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    env = cv2.normalize(env, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    save_image(
        np.transpose(input_tensor.cpu().numpy().squeeze(0), (1, 2, 0)) * 255,
        os.path.join(output_img_dir, f"{img_name}_ori.png"),
    )
    save_image(rec, os.path.join(output_img_dir, f"{img_name}_rec.png"))
    save_image(T, os.path.join(output_img_dir, f"{img_name}_T.png"))
    save_image(e, os.path.join(output_img_dir, f"{img_name}_e.png"))
    save_image(env, os.path.join(output_img_dir, f"{img_name}_env.png"))

    for i in range(2, 2 + preds_np.shape[0] - 2):
        v_i = cv2.normalize(preds_np[i], None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        save_image(v_i, os.path.join(output_img_dir, f"{img_name}_V{i-1}.png"))


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    model = load_model(args.weights_file, args, device)
    lossmodule = TeVloss(vnums=args.vnums)

    loss_list = []
    imglist = [f for f in os.listdir(args.image_dir) if f.lower().endswith(("png", "jpg", "jpeg"))]

    with open(os.path.join(args.output_dir, "test_loss.txt"), "a+") as loss_file:
        for img in tqdm(imglist):
            img_name = os.path.splitext(img)[0]
            out_dir = os.path.join(args.output_dir, img_name)
            os.makedirs(out_dir, exist_ok=True)

            input_tensor = process_image(os.path.join(args.image_dir, img), device)
            with torch.no_grad():
                preds = model(input_tensor)

            loss = lossmodule.loss_rec(preds, input_tensor)
            loss_list.append(loss.item())
            loss_file.write(f"{img}, loss: {loss.item():.6f}\n")

            save_decomposed_images(preds, input_tensor, out_dir, img_name, lossmodule)

        mean_loss = float(np.mean(loss_list)) if loss_list else 0.0
        loss_file.write(f"Mean loss: {mean_loss:.6f}\n")
        print(f"Mean loss: {mean_loss:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights-file", type=str, required=True)
    parser.add_argument("--image-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="./output/")

    parser.add_argument("--vnums", type=int, default=4)
    parser.add_argument("--smp_model", type=str, default="PAN")
    parser.add_argument("--smp_encoder", type=str, default="resnet50")
    parser.add_argument("--smp_encoder_weights", type=str, default="imagenet")
    parser.add_argument("--arch", type=str, default="baseline", choices=["baseline", "thernet", "pp", "phys"])
    parser.add_argument("--v-softmax", action="store_true")
    parser.add_argument("--mrm-k", type=int, default=16)
    parser.add_argument("--mrm-d", type=int, default=32)
    parser.add_argument("--tau-min", type=float, default=0.05)
    parser.add_argument("--tim-kernel", type=int, default=5)
    parser.add_argument("--no-dp", action="store_true")
    args = parser.parse_args()

    if args.arch in ("thernet", "pp", "phys") and not args.v_softmax:
        args.v_softmax = True

    main(args)
