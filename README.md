# HW3: CFG Flow Matching — Attribute-Conditional Generation & Editing

## Quick Answer: Do I Need to Retrain?

**YES.** Your old HW2 weights are incompatible because:
1. The architecture changed — `CondUNet` has new attribute embedding layers (`attr_embed`) that don't exist in the old `UNet`
2. CFG requires the model to learn both conditional and unconditional denoising during training (via condition dropout)
3. The old model has never seen attribute vectors, so it can't do guided sampling

**Optional speedup:** You can warm-start from HW2 weights for the shared layers (everything except `attr_embed`). See the training section below.

---

## Flow Matching Bug Fixes

### Bug 1: Intermediate Clamping (MAJOR — likely cause of KID 0.075)
**File:** `src/methods/flow_matching.py` line 154  
**Problem:** `x = x.clamp(-1.0, 1.0)` was applied at every Euler step.  
**Why it's bad:** The straight-line ODE `x_t = (1-t)·noise + t·data` passes through values outside [-1,1] at intermediate steps (because Gaussian noise has heavy tails). Clamping every step distorts the trajectory and accumulates error.  
**Fix:** Removed intermediate clamping. Only clamp the final output.

### Bug 2: Sampling Never Reaches t=1 Endpoint
**Problem:** Last evaluation at `t = (S-1)/S` (e.g., 0.999 for S=1000), so model never predicts at the `t=1` boundary it was trained on.  
**Fix:** This is a minor issue, but noted. The sampling grid `k/S` for `k=0..S-1` with `dt=1/S` does integrate to `t=1` after S steps, so the state reaches `t=1` — the model is just never *queried* at `t_idx=T-1`.

### Bug 3: Meaningless DDPM Fields in Config
**Problem:** `beta_start` and `beta_end` in `flow_matching_modal.yaml` are never used by `FlowMatching`.  
**Fix:** Removed from the new config.

### Expected Impact
Removing intermediate clamping should significantly improve Flow Matching quality. Your KID ~0.075 should drop substantially — possibly to 0.01–0.03 range depending on other factors.

---

## New Files for HW3

```
hw3/
├── configs/
│   └── cfg_flow_matching.yaml      # Config for conditional training
├── src/
│   ├── models/
│   │   ├── blocks.py               # (unchanged from HW2)
│   │   ├── unet.py                 # (unchanged from HW2)
│   │   └── unet_cond.py            # NEW: Conditional UNet with attribute embedding
│   ├── methods/
│   │   ├── base.py                 # (unchanged from HW2)
│   │   ├── flow_matching.py        # FIXED: removed clamping bug
│   │   └── cfg_flow_matching.py    # NEW: CFG Flow Matching method
│   ├── data/
│   │   └── celeba_cond.py          # NEW: CelebA dataset with attribute labels
│   └── utils/
│       └── ema.py                  # (unchanged from HW2)
├── train_cfg.py                    # NEW: Training script for CFG
├── sample_cfg.py                   # NEW: Sampling & editing script
└── train_classifier.py             # NEW: Attribute classifier for evaluation
```

---

## How to Run

### 1. Train the Conditional Model

```bash
# Standard training (from scratch)
python train_cfg.py --config configs/cfg_flow_matching.yaml

# Overfit sanity check first
python train_cfg.py --config configs/cfg_flow_matching.yaml --overfit-single-batch

# Resume from checkpoint
python train_cfg.py --config configs/cfg_flow_matching.yaml --resume logs/.../checkpoints/cfg_flow_matching_XXXX.pt
```

### 2. Generate Conditional Samples

```bash
# Attribute comparison grid (each row = different attribute)
python sample_cfg.py --checkpoint CKPT_PATH --mode attr_grid --num_steps 200

# Generate Smiling faces
python sample_cfg.py --checkpoint CKPT_PATH --mode generate \
    --attributes "Smiling=1" --guidance_scale 2.0 --num_samples 64 --grid

# Guidance scale sweep (compare w=0,1,2,3,4)
python sample_cfg.py --checkpoint CKPT_PATH --mode sweep \
    --attributes "Smiling=1" --guidance_scales "0,1,2,3,4"

# Save individual images for KID evaluation
python sample_cfg.py --checkpoint CKPT_PATH --mode generate \
    --attributes "Smiling=1" --num_samples 1000 --output_dir cfg_samples/smiling
```

### 3. Edit Real Images

```bash
# Add smile to real faces
python sample_cfg.py --checkpoint CKPT_PATH --mode edit \
    --input_dir ./data/celeba-subset/train/images \
    --attributes "Smiling=1" --t_edit 0.5 --guidance_scale 2.0 --num_samples 16

# Stronger edit (more change, less identity preservation)
python sample_cfg.py --checkpoint CKPT_PATH --mode edit \
    --attributes "Eyeglasses=1" --t_edit 0.3 --guidance_scale 3.0
```

### 4. Train & Evaluate Attribute Classifier

```bash
# Train classifier
python train_classifier.py --config configs/cfg_flow_matching.yaml --epochs 10

# Evaluate control accuracy of generated samples
python train_classifier.py --config configs/cfg_flow_matching.yaml \
    --eval --eval_dir cfg_samples/smiling --classifier_ckpt classifier.pt \
    --target_attrs "1,0,0"
```

---

## Warm-Starting from HW2 Weights (Optional)

To speed up training, you can initialize the shared layers from your HW2 flow matching checkpoint:

```python
import torch
from src.models.unet_cond import create_cond_model_from_config

# Load old unconditional weights
old_ckpt = torch.load("flow_matching_final.pt", map_location="cpu")
old_state = old_ckpt["model"]

# Create new conditional model
config = ...  # load your cfg_flow_matching.yaml
new_model = create_cond_model_from_config(config)

# Load matching keys (skip attr_embed which is new)
new_state = new_model.state_dict()
transferred = 0
for key in old_state:
    if key in new_state and old_state[key].shape == new_state[key].shape:
        new_state[key] = old_state[key]
        transferred += 1
print(f"Transferred {transferred}/{len(old_state)} weight tensors")

new_model.load_state_dict(new_state)
# Then save and use as starting point for train_cfg.py --resume
```

---

## Key Hyperparameters to Tune

| Parameter | Default | Notes |
|-----------|---------|-------|
| `p_uncond` | 0.1 | Condition dropout rate. Higher = better unconditional mode but slower conditional learning |
| `guidance_scale` (w) | 2.0 | Higher = stronger attribute control but may reduce diversity/quality |
| `t_edit` | 0.5 | Edit strength. Lower = stronger edit, more identity change |
| `num_steps` | 200 | Euler steps. More = better quality but slower |

---

## Architecture Changes (CondUNet vs UNet)

The only difference between `CondUNet` and the original `UNet` is a new `attr_embed` module:

```python
# New: attribute embedding MLP
self.attr_embed = nn.Sequential(
    nn.Linear(num_attributes, time_hidden),    # K -> 512
    nn.SiLU(),
    nn.Linear(time_hidden, time_hidden),       # 512 -> 512
)
```

This adds ~525K parameters (for K=3, time_hidden=512). The embedding is added to the timestep embedding before it enters the ResBlocks:

```python
temb = self.time_embed(t)         # timestep embedding
attr_emb = self.attr_embed(c)    # attribute embedding
temb = temb + attr_emb            # fused embedding
```

For unconditional mode (null condition), pass `c = zeros(B, K)` which produces a near-zero embedding after the MLP.
