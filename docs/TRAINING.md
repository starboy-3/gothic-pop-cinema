# Training and reproducibility

## Experimental run

| Setting | Value |
| --- | --- |
| Infrastructure | Nebius Serverless Devlab, one RTX PRO 6000 Blackwell Server Edition GPU |
| Training model | `black-forest-labs/flux-3-image-klein-base` |
| Inference model | `black-forest-labs/flux-3-image-klein` |
| Inference model revision | `337337621b8526e120f2275d250242f4a6bedd75` |
| LoRA rank / alpha | 32 / 32 |
| Optimizer / learning rate | AdamW / 1e-4 |
| Precision / batch | BF16 / 1 |
| Training target resolution | 1024 with aspect-ratio buckets |
| Run length | 2000 steps |
| Selected checkpoint | 1500, provisional per-scene quality selection |
| Inference | 16 steps, adapter strength 0.5–0.75 depending on scene |
| Trigger | `gthmvr style` |

Step-1500 adapter SHA-256: `4f5d9efef974faee0365b4956c1e7d0a9c40aa81695c4d7fd7a91ed138ded44d`.

The historical configuration is in `configs/training-run.yaml`. The reusable template retains more checkpoints than the original run. Do not mistake later template changes for settings used historically. The abbreviated run did not explicitly configure the warmup, optimizer betas and decay described in the fuller upstream training prose; defaults were used. No causal claim about quality is made from that difference.

## Components used

- AI Toolkit: `8cf43dd2cc1fe828884001aceeda044649d9abd0`.
- Diffusers inference snapshot: `b800f828f3951c4d206e154411bcbfa693cd322e`.
- BFL prompt upsampler: `black-forest-labs/flux-3-image-klein-pup`, revision `3f3edb52b2c55de0d49feab28598b68f7f8e4830`.
- GPU environment observed: PyTorch 2.13.0+cu130 and Transformers 5.5.3.
- Inference overlay used Hugging Face Hub 1.33.0 and TorchAO 0.15.0. The TorchAO compiled extension was incompatible with that PyTorch build and was not used for BF16 generation.

`requirements.txt` covers only lightweight preparation dependencies. AI Toolkit's installer manages the training environment. For the frame pipeline, a separate compatible CUDA environment also needs PyTorch, Transformers with `Qwen3_5ForConditionalGeneration`, accelerate, safetensors, sentencepiece and the pinned Diffusers snapshot. For example, after provisioning a compatible PyTorch environment:

```bash
pip install 'transformers==5.5.3' accelerate safetensors sentencepiece \
  'huggingface_hub==1.33.0' 'torchao==0.15.0' \
  'git+https://github.com/huggingface/diffusers.git@b800f828f3951c4d206e154411bcbfa693cd322e'
```

This is a component record, not a fully locked cross-platform environment. Review compatibility before installing into an existing training environment. Keep training and experimental inference dependencies separate.

## Honest comparison

The committed florist images use identical model/revision, text, seed, size and inference-step count. Adapter strength is 0 for the control and 0.75 for the styled output. Full settings are in `assets/comparison/settings.json`.

These are single-seed examples, not a benchmark proving universal superiority or that the base model cannot represent Gothic aesthetics. Evaluate new subjects, faces, hands, text and motion before selecting a production checkpoint.

## Data review

The local experimental set had 36 images, including 14 below approximately one megapixel, mixed sources, two watermarks and a sideways reference. Provenance remains unresolved. No raw training images or adapters are bundled. Before any distribution, resolve the rights of the full training set, not merely the six examples shown to judges.
