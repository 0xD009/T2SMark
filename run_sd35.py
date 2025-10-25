import os
import json
import torch
import open_clip
import numpy as np
from tqdm import tqdm
from sklearn import metrics
from functools import partial

import src.utils as utils
from src.t2s import T2SMark
from src.inversion.inverse_diffusion3 import InversionDiffusion3Pipeline

from option import args

args.key_channel_idx = (0, 1, 2, 3)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
msg_channel_idx = [i for i in range(16) if i not in args.key_channel_idx]

def setup_clip_measure_function():
    open_clip_model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    open_clip_model.to(device)
    open_clip_model.eval()
    open_clip_tokenizer = open_clip.get_tokenizer('ViT-B-32')

    return partial(utils.measure_similarity, model=open_clip_model, 
                   tokenizer=open_clip_tokenizer, clip_preprocess=preprocess, device=device)

def decode(post_reversed_latents, master_key, key, fake_key, msg):
    reversed_key_channel = post_reversed_latents[0, args.key_channel_idx, :, :]
    reversed_msg_channel = post_reversed_latents[0, msg_channel_idx, :, :]

    _, norm1_no_w = t2s_key.decode(reversed_key_channel, fake_key, detection=True)

    reversed_key, norm1_w = t2s_key.decode(reversed_key_channel, master_key, detection=True)
    reversed_msg = t2s_msg.decode(reversed_msg_channel, reversed_key)

    acc_key = (reversed_key == key).float().mean()
    acc_msg = (reversed_msg == msg).float().mean()

    return {
        "norm1_no_w": norm1_no_w,
        "norm1_w": norm1_w,
        "acc_key": acc_key.item(),
        "acc_msg": acc_msg.item()
    }

pipe = InversionDiffusion3Pipeline.from_pretrained(args.model_key, torch_dtype=torch.float16).to(device)
pipe.set_progress_bar_config(disable=True)

t2s_key = T2SMark(m=args.key_length, tau=args.tau, latent_shape=(4, 64, 64))
t2s_msg = T2SMark(m=args.msg_length, tau=args.tau, latent_shape=(12, 64, 64))

settings = vars(args)
if args.fix_key: 
    utils.set_random_seed(args.seed)
    master_key = torch.randint(0, 2, (args.key_length,)).cuda()
    msg = torch.randint(0, 2, (args.msg_length,)).cuda()
    settings["master_key"] = t2s_key.binlist2int(master_key)
print(settings)

os.makedirs(os.path.join(args.output_dir, args.name), exist_ok=True)
with open(os.path.join(args.output_dir, args.name, "settings.json"), "w") as f:
    json.dump(settings, f, indent=4)

if args.save_image:
    image_path = os.path.join(args.output_dir, args.name, f'images')
    os.makedirs(image_path, exist_ok=True)

results = {}
if args.clip_test_num > 0: clip_score_fn = setup_clip_measure_function()

prompt_id = 0
with tqdm(total=max(args.robust_test_num, args.clip_test_num)) as pbar:
    for prompt in utils.load_prompt(args.dataset_key):

        if prompt_id < args.start_idx:
            prompt_id += 1
            pbar.update(1)
            continue

        if prompt_id >= args.clip_test_num and prompt_id >= args.robust_test_num:
            break
        
        results[prompt_id] = {}

        utils.set_random_seed(args.seed + prompt_id)

        if not args.fix_key: 
            master_key = torch.randint(0, 2, (args.key_length,)).cuda()
            msg = torch.randint(0, 2, (args.msg_length,)).cuda()
        key = torch.randint(0, 2, (args.key_length,)).cuda()

        # ensure the fake key is different from the master key
        fake_key = 1 - master_key
        
        z_k = t2s_key.encode(key, master_key)
        z_b = t2s_msg.encode(msg, key)

        initial_latents = torch.zeros(1, 16, 64, 64).cuda()
        initial_latents[:, args.key_channel_idx, :, :] = z_k
        initial_latents[:, msg_channel_idx, :, :] = z_b

        generated_image = pipe(
            prompt,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            height=512,
            width=512,
            latents=initial_latents.half()
        ).images[0]
        # numpy (b, w, h, c) float16 [0, 1]

        if prompt_id < args.clip_test_num:
            clip_score = clip_score_fn(generated_image, prompt).item()
            results[prompt_id]["clip_score"] = clip_score
        if args.save_image:
            generated_image.save(os.path.join(image_path, f'{str(prompt_id).zfill(5)}.png'))
        
        if prompt_id < args.robust_test_num:
            noised_image = utils.image_distortion(generated_image, args.seed + prompt_id, args)

            image_tensor = utils.to_tensor(noised_image).to(device).half()
            latents = pipe.get_image_latents(image_tensor, sample=False)
            reversed_latents = pipe.naive_forward_diffusion(
                latents=latents,
                num_inference_steps=args.num_inversion_steps
            )
            
            decode_result = decode(reversed_latents, master_key, key, fake_key, msg)
            results[prompt_id]["robustness"] = decode_result

        prompt_id += 1
        pbar.update(1)

if args.robust_test_num > 0:
    total_acc = 0
    no_w_metrics = []
    w_metrics = []
    for v in results:
        if "robustness" in results[v]:
            total_acc += results[v]['robustness']['acc_msg']
            no_w_metrics.append(results[v]['robustness']["norm1_no_w"])
            w_metrics.append(results[v]['robustness']["norm1_w"])

    preds = no_w_metrics +  w_metrics
    t_labels = [0] * len(no_w_metrics) + [1] * len(w_metrics)

    fpr, tpr, thresholds = metrics.roc_curve(t_labels, preds, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    acc = np.max(1 - (fpr + (1 - tpr))/2)
    low = tpr[np.where(fpr<1e-6)[0][-1]]

    results["tpr"] = low
    results["bit_accuracy"] = total_acc / args.robust_test_num

if args.clip_test_num > 0:
    total_clip_score = 0
    for v in results:
        if "clip_score" in results[v]:
            total_clip_score += results[v]['clip_score']
    results["clip_score"] = total_clip_score / args.clip_test_num

# save results to json
with open(os.path.join(args.output_dir, args.name, "results.json"), "w") as f:
    json.dump(results, f, indent=4)


