# Water Segmentation using Multispectral Optical Data

Pixel-wise segmentation of water bodies from 12-band multispectral satellite imagery. This repository covers two stages:

- **Week 1:** a U-Net trained from scratch (baseline).
- **Week 2:** fine-tuning ImageNet-pretrained encoders (ResNet34) with U-Net and DeepLabV3+ decoders, and comparing them against the Week 1 model.

---

## Dataset

- 306 image/mask pairs, each image has **12 bands** at 128 x 128 pixels.
- Bands: Coastal aerosol, Blue, Green, Red, NIR, SWIR1, SWIR2, QA band, Merit DEM, Copernicus DEM, ESA WorldCover, Water occurrence.
- Masks are binary (1 = water, 0 = background). Overall water pixel ratio is about 0.28.
- Split (fixed seed, same for all experiments): **214 train / 45 validation / 47 test**.
- Preprocessing: NaN/Inf replaced with 0, per-channel percentile normalization to [0, 1] fitted on the **training set only**.

## Week 1: Baseline

- **Model:** U-Net trained from scratch (about 7.8M parameters), 12 input channels.
- **Loss:** BCE + Dice.
- **Optimization:** AdamW, ReduceLROnPlateau, mixed precision, early stopping on validation IoU.
- **Augmentation:** random horizontal/vertical flips and 90-degree rotations.
- **Threshold:** tuned on the validation set (never on test).

## Week 2: Pretrained Encoders

### Approach

- **Library:** [segmentation-models-pytorch](https://github.com/qubvel-org/segmentation_models.pytorch).
- **Encoder:** ResNet34 pretrained on ImageNet. This is the only pretrained part; the decoder is randomly initialized.
- **Decoders:** U-Net and DeepLabV3+.
- **Input adaptation (3 -> 12 channels):** the first convolution of the encoder expects 3 channels, so it is replaced by a 12-channel convolution. Two initialization strategies were tested:
  - **Mean-init:** average the pretrained R/G/B filters, repeat them across all 12 channels, and rescale by 3/12 to keep activation magnitudes comparable.
  - **Zero-init:** copy the pretrained R/G/B filters onto the Red/Green/Blue bands and initialize the weights of the other 9 bands to zero. The network initially behaves like the original RGB model, and the extra bands are learned during fine-tuning.
- **Input scaling:** inputs in [0, 1] are standardized with `(x - 0.5) / 0.25` before entering the pretrained encoder.
- **Fine-tuning:** AdamW with **differential learning rates** (encoder LR = 0.1 x decoder LR), BCE + Dice loss, ReduceLROnPlateau, early stopping on validation IoU, best checkpoint restored.
- **Fair comparison:** same split, same 12 input channels, same loss, same augmentation, and same metrics as Week 1.

### Experiments

| ID | Decoder | Encoder | First-conv init | Pretrained |
|---|---|---|---|---|
| Week 1 | U-Net (custom) | custom | n/a | No |
| P1 | U-Net | ResNet34 | mean | Yes |
| P2 | U-Net | ResNet34 | zero | Yes |
| P3 | DeepLabV3+ | ResNet34 | mean | Yes |
| P4 | U-Net | ResNet34 | mean | **No** (ablation) |

P4 uses exactly the same architecture as P1 but with random weights. It separates the effect of **pretraining** from the effect of the **architecture / model size**.

### Metrics

All metrics are computed at the pixel level, accumulated over the whole test set, from TP/FP/FN:

- **Precision** = TP / (TP + FP)
- **Recall** = TP / (TP + FN)
- **F1** = 2PR / (P + R)
- **IoU** = TP / (TP + FP + FN)

The model is selected by **validation IoU**, and the decision threshold is tuned on the validation set only.

### Results

Test-set metrics (threshold tuned on validation):

| Model | Precision | Recall | F1 | IoU | Best Val IoU | Threshold | Epochs |
|---|---|---|---|---|---|---|---|
| Scratch U-Net (Week 1) | 0.8316 | 0.7878 | 0.8091 | 0.6794 | 0.6978 | 0.55 | 64 |
| P1: U-Net + ResNet34 (mean-init) | 0.8488 | 0.7684 | 0.8066 | 0.6759 | 0.6985 | 0.50 | 30 |
| P2: U-Net + ResNet34 (zero-init) | **0.8743** | 0.7785 | **0.8236** | **0.7001** | 0.7051 | 0.55 | 35 |
| P3: DeepLabV3+ + ResNet34 (mean-init) | 0.8117 | **0.8292** | 0.8204 | 0.6955 | **0.7224** | 0.45 | 52 |
| P4: U-Net + ResNet34 (no pretraining) | 0.8389 | 0.7861 | 0.8116 | 0.6829 | 0.6879 | 0.55 | 66 |

![Comparison](week2_comparison.png)

### Findings

**Which model performs better?**
Both best pretrained models beat the Week 1 scratch U-Net:

- **P3 (DeepLabV3+, mean-init)** reaches the highest validation IoU: **0.7224 vs 0.6978** (+0.025).
- **P2 (U-Net, zero-init)** reaches the highest test IoU: **0.7001 vs 0.6794** (+0.021), with the best precision and F1.

This meets the Week 2 requirement of a measurably higher validation IoU than the scratch model.

**Why?**

1. **Pretraining gives a better starting point.** ImageNet encoders already capture generic low-level features (edges, textures, shapes). With only 214 training images, this is more useful than learning every filter from scratch. Pretrained models also converged faster (30-52 epochs vs 64-66) and had smoother validation curves. The P4 curve shows sharp validation collapses (down to about 0.35-0.39 IoU) that the pretrained models do not show.
2. **Pretraining, not just size, matters.** P4 has the same architecture as P1/P2 but random weights, and it scored lower than P2 on both validation and test IoU.
3. **Zero-init beats mean-init** (P2 vs P1: +0.024 test IoU). Mean-init spreads the RGB filters over all bands, including non-optical bands (NIR, SWIR, DEM, ...) whose statistics differ strongly from natural images, which dilutes the pretrained features. Zero-init keeps the pretrained RGB behavior intact and lets the other bands be learned gradually.
4. **DeepLabV3+ vs U-Net** (P3 vs P1, same encoder and init): DeepLabV3+ has higher recall (0.829 vs 0.768) and higher IoU, but lower precision (0.812 vs 0.849). Its atrous convolutions and ASPP module give a larger receptive field, which helps with large, connected water bodies, while the simpler decoder is less precise at boundaries.
5. **Mean-init U-Net (P1) did not improve** over the scratch model (val IoU 0.6985 vs 0.6978), so pretraining only helps when the first layer is adapted well.

### Limitations

- The validation set (45 images) and test set (47 images) are small. Differences of about 0.01-0.02 IoU may partly be noise.
- Epoch selection and threshold tuning use the validation set, so validation numbers are slightly optimistic; the test numbers are the more reliable estimate.
- Each experiment was run once with a single seed. Reporting mean and standard deviation over several seeds would make the conclusions stronger.
- ImageNet statistics do not match multispectral bands, so the first layer has to adapt during fine-tuning.
- Some input channels (QA band, ESA WorldCover, Water occurrence) may carry information correlated with the label. Ablations without them are in the Week 1 experiments.

### Future work

- DeepLabV3+ with zero-init (combining the two best ideas).
- Other encoders (e.g. EfficientNet) and all 16 channels (12 bands + spectral indices).
- Multi-seed runs or cross-validation.

## Repository Structure

```
.
├── notebook.ipynb          # Week 1 + Week 2 experiments
├── week2_comparison.csv    # Final comparison table
├── week2_comparison.png    # Test metrics and validation IoU curves
└── README.md
```

## How to Run

```bash
pip install torch numpy pandas matplotlib rasterio segmentation-models-pytorch
```

Open the notebook, set the dataset path in the first cell, set `EPOCHS` to the desired value, and run all cells. Internet access is required once to download the ImageNet weights.
