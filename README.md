# T2SMark: Balancing Robustness and Diversity in Noise-as-Watermark for Diffusion Models

## Dependencies

The code requires **Python 3.10.13** and the packages listed in `requirements.txt`. Install them with:

```bash
pip install -r requirements.txt
```

> **Note:** To run T2SMark on Stable Diffusion v3.5 Medium, upgrade `diffusers` from `0.21.4` to `0.32.0`. Otherwise, `StableDiffusion3Pipeline` will not be supported.

## Usage

### Stable Diffusion v2.1

```bash
python run.py --name test
```

### Stable Diffusion v3.5 Medium

```bash
python run_sd35.py --name test_sd35
```

The `--name` argument is required. For additional options, see `option.py`.

## Evaluation

Our code includes built-in evaluation of TPR, bit accuracy, and CLIP score. For LPIPS and FID metrics, we recommend using the following repositories:

- LPIPS score: https://github.com/richzhang/PerceptualSimilarity.git
- FID: https://github.com/mseitzer/pytorch-fid.git

> **Note:** The COCO prompts and ground-truth images used in our experiments are available [here](https://drive.google.com/drive/folders/1saWx-B3vJxzspJ-LaXSEn5Qjm8NIs3r0), sourced from the [Tree-Ring Watermark](https://github.com/YuxinWenRick/tree-ring-watermark.git) repository.


## Baselines

The code for all baseline methods is provided below:

- DwtDct, DwtDctSvd, RivaGan: https://github.com/ShieldMnt/invisible-watermark.git
- Stable Signature: https://github.com/facebookresearch/stable_signature.git
- Tree-Ring Watermark: https://github.com/YuxinWenRick/tree-ring-watermark.git
- Gaussian Shading: https://github.com/bsmhmmlf/Gaussian-Shading.git
- PRC-Watermark: https://github.com/XuandongZhao/PRC-Watermark.git


## Acknowledgements
We borrow the code from [Tree-Ring Watermark](https://github.com/YuxinWenRick/tree-ring-watermark.git) and [Gaussian Shading](https://github.com/bsmhmmlf/Gaussian-Shading.git).  We appreciate the authors for sharing their code.

## Citation
If our work assists your research, feel free to give us a star ⭐ or cite us using:
```
@article{yang2025t2smark,
  title={T2SMark: Balancing Robustness and Diversity in Noise-as-Watermark for Diffusion Models},
  author={Yang, Jindong and Fang, Han and Zhang, Weiming and Yu, Nenghai and Chen, Kejiang},
  journal={arXiv preprint arXiv:2510.22366},
  year={2025}
}
```
