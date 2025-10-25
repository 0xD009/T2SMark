import os
import time
import json
import torch
import random
import hashlib
import numpy as np
import pandas as pd
from typing import Generator
from datasets import load_dataset
from torchvision import transforms
from PIL import Image, ImageFilter


def load_prompt(path: str) -> Generator[str, None, None]:
    if path == "Gustavosta/Stable-Diffusion-Prompts":
        ds = load_dataset(path)
        yield from ds["train"]["Prompt"]
    elif path.endswith(".json"): # coco
        with open(path, "r", encoding="utf-8") as f:
            dataset = json.load(f)['annotations']
        prompt_key = 'caption'
        yield from (d[prompt_key] for d in dataset)
    elif path.endswith(".csv"):
        df = pd.read_csv(path)
        yield from df["Our GT caption"]

def to_tensor(data: Image) -> torch.Tensor:
    np_data = np.array(data)
    data = torch.from_numpy(np_data)
    # analyze_data(data)
    data = data.unsqueeze(0).float()
    data = data / 255
    data = data * 2 - 1
    # b, w, h, c = data.shape
    data = data.permute(0, 3, 1, 2)

    return data

def set_random_seed(seed=0):
    torch.manual_seed(seed + 0)
    torch.cuda.manual_seed(seed + 1)
    torch.cuda.manual_seed_all(seed + 2)
    random.seed(seed + 3)

def image_distortion(img, seed, args):

    if args.jpeg_ratio is not None:
        hash_str = hashlib.md5(str(time.time()).encode()).hexdigest()
        img.save(f"tmp_{args.jpeg_ratio}_{hash_str}.jpg", quality=args.jpeg_ratio)
        img = Image.open(f"tmp_{args.jpeg_ratio}_{hash_str}.jpg")
        os.remove(f"tmp_{args.jpeg_ratio}_{hash_str}.jpg")

    if args.random_crop_ratio is not None:
        set_random_seed(seed)
        width, height, c = np.array(img).shape
        img = np.array(img)
        new_width = int(width * args.random_crop_ratio)
        new_height = int(height * args.random_crop_ratio)
        start_x = np.random.randint(0, width - new_width + 1)
        start_y = np.random.randint(0, height - new_height + 1)
        end_x = start_x + new_width
        end_y = start_y + new_height
        padded_image = np.zeros_like(img)
        padded_image[start_y:end_y, start_x:end_x] = img[start_y:end_y, start_x:end_x]
        img = Image.fromarray(padded_image)

    if args.random_drop_ratio is not None:
        set_random_seed(seed)
        width, height, c = np.array(img).shape
        img = np.array(img)
        new_width = int(width * args.random_drop_ratio)
        new_height = int(height * args.random_drop_ratio)
        start_x = np.random.randint(0, width - new_width + 1)
        start_y = np.random.randint(0, height - new_height + 1)
        padded_image = np.zeros_like(img[start_y:start_y + new_height, start_x:start_x + new_width])
        img[start_y:start_y + new_height, start_x:start_x + new_width] = padded_image
        img = Image.fromarray(img)

    if args.resize_ratio is not None:
        img_shape = np.array(img).shape
        resize_size = int(img_shape[0] * args.resize_ratio)
        img = transforms.Resize(size=resize_size)(img)
        img = transforms.Resize(size=img_shape[0])(img)

    if args.gaussian_blur_r is not None:
        img = img.filter(ImageFilter.GaussianBlur(radius=args.gaussian_blur_r))

    if args.median_blur_k is not None:
        img = img.filter(ImageFilter.MedianFilter(args.median_blur_k))

    if args.gaussian_std is not None:
        img_shape = np.array(img).shape
        g_noise = np.random.normal(0, args.gaussian_std, img_shape) * 255
        g_noise = g_noise.astype(np.uint8)
        img = Image.fromarray(np.clip(np.array(img) + g_noise, 0, 255))

    if args.sp_prob is not None:
        c,h,w = np.array(img).shape
        prob_zero = args.sp_prob / 2
        prob_one = 1 - prob_zero
        rdn = np.random.rand(c,h,w)
        img = np.where(rdn > prob_one, np.zeros_like(img), img)
        img = np.where(rdn < prob_zero, np.ones_like(img)*255, img)
        img = Image.fromarray(img)

    if args.brightness_factor is not None:
        img = transforms.ColorJitter(brightness=args.brightness_factor)(img)

    return img
    
def measure_similarity(image, prompt, model, clip_preprocess, tokenizer, device):
    with torch.no_grad():
        img_batch = clip_preprocess(image).unsqueeze(0).to(device)
        image_features = model.encode_image(img_batch)

        text = tokenizer([prompt]).to(device)
        text_features = model.encode_text(text)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        return (image_features @ text_features.T).mean(-1)