import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    # experiment settings
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save_image", action="store_true", default=False)
    parser.add_argument("--fix_key", action="store_true", default=False) # whether to fix the key. Fix it to simulate the case of a single accout.
    parser.add_argument("--robust_test_num", type=int, default=10) # actually the end idx
    parser.add_argument("--clip_test_num", type=int, default=0) # actually the end idx
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--SDv35M", action="store_true", default=False)

    # diffusion model settings
    parser.add_argument("--model_key", type=str, default="stabilityai/stable-diffusion-2-1") # stabilityai/stable-diffusion-3.5-medium
    parser.add_argument("--guidance_scale", type=float, default=7.5) # 4.0 for SDv35M diversity and image quality
    parser.add_argument("--num_inference_steps", type=int, default=50) # 40 for SDv35M

    # inversion settings
    parser.add_argument("--num_inversion_steps", type=int, default=10)

    # watermark settings
    parser.add_argument("--key_channel_idx", type=int, default=0) # the channel to embed the session key. For SDv35M, it is [0, 1, 2, 3].
    parser.add_argument("--key_length", type=int, default=16) # the session key size
    parser.add_argument("--msg_length", type=int, default=256) # the watermark capacity
    parser.add_argument("--tau", type=float, default=0.674) # the threshold for Tail-Truncated Sampling
    
    # dataset settings
    parser.add_argument("--dataset_key", type=str, default="Gustavosta/Stable-Diffusion-Prompts")

    # noise settings
    parser.add_argument('--jpeg_ratio', type=int, default=None)
    parser.add_argument('--random_crop_ratio', type=float, default=None)
    parser.add_argument('--random_drop_ratio', type=float, default=None)
    parser.add_argument('--gaussian_blur_r', type=int, default=None)
    parser.add_argument('--median_blur_k', type=int, default=None)
    parser.add_argument('--resize_ratio', type=float, default=None)
    parser.add_argument('--gaussian_std', type=float, default=None)
    parser.add_argument('--sp_prob', type=float, default=None)
    parser.add_argument('--brightness_factor', type=float, default=None)

    return parser.parse_args()

args = parse_args()

