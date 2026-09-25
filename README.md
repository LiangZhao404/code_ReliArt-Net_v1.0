# ReliArt-Net — MATLAB R2025b implementation

This repository accompanies the manuscript **“Reproducibility in Art: Reliability of original and reproduced masterpieces based on ReliArt-Net.”** It provides a MATLAB R2025b (Version 25.2) implementation of the method described in the manuscript: preprocessing, controlled augmentation, three CNN streams, dilated convolutions, pyramid pooling, 16×16 patch embedding, a 6-block/8-head Transformer, adaptive CNN–Transformer fusion, separate catalogue-status and expert visual-fidelity heads, joint masked multi-task training, and validation-only post-hoc temperature scaling.

## Scope and reproducibility
The implementation follows the architecture, training protocol, calibration procedure, and numerical settings explicitly reported in the manuscript. Parameters that are not numerically specified in the manuscript are clearly marked in the configuration/code as **repository defaults** and are not presented as manuscript-reported constants. This distinction is important when attempting exact numerical reproduction of reported results.

The software predicts catalogue-status class and an auxiliary expert-supervised visual-fidelity score. These outputs are distinct and neither should be interpreted independently as proof of physical authorship.

## Environment
- MATLAB R2025b (Version 25.2)
- Deep Learning Toolbox
- Image Processing Toolbox
- Statistics and Machine Learning Toolbox is useful for AUROC (`perfcurve`); evaluation degrades gracefully if unavailable.

The manuscript reports Windows 11 Pro and MATLAB R2025b (Version 25.2) as the experimental environment.

## Manuscript-specified configuration
- Primary input: 512×512 RGB; WikiArt-Baroque: 224×224.
- Direct bilinear resize with antialiasing; no aspect-ratio-preserving padding/cropping at this stage.
- Primary dataset split: 70% train / 15% validation / 15% locked test, stratified by catalogue-status label.
- CNN: 3×3 convolutions, dilation rates 2 and 4, terminal 2048 filters, PPM scales 1×1, 2×2, 3×3, 6×6.
- Transformer: 16×16 patches, D=256, 8 heads, 6 blocks, FFN=1024, dropout 0.1.
- ReliArt-Net row in Table 6: AdamW, initial LR 1e-4, batch size 32, 100 epochs, CNN dropout 0.2, Transformer dropout 0.1, joint-loss λ=0.5.
- Early stopping patience: 15 epochs; cosine learning-rate schedule.
- Post-hoc calibration: fit one scalar temperature on validation classification logits by minimizing NLL with network weights frozen; apply the fitted temperature to test classification logits only.


## Data
Artwork images are not redistributed by this repository. Create `data/metadata.csv` with the following fields:

```csv
image_path,catalogue_label,fidelity_target,fidelity_mask,artwork_family,source
/path/image1.jpg,1,NaN,0,family_001,provider_A
/path/image2.jpg,0,0.875,1,family_002,provider_B
```

`catalogue_label` is the catalogue-derived binary target. `fidelity_target` is a normalized expert visual-fidelity target, not a probability of autograph authorship. Set `fidelity_mask=1` only when an expert target exists.

## Run the pipeline
From MATLAB, change to the repository root and run:

```matlab
addpath(genpath(pwd));
scripts/prepare_splits
scripts/train
scripts/export_logits
scripts/calibrate
scripts/evaluate
```

For the source- and artwork-family-disjoint sensitivity analysis, run `scripts/prepare_group_disjoint_splits.m`, point the training script to the generated metadata file, and retrain from initialization.

## Outputs
- `artifacts/best_model.mat`: validation-selected checkpoint.
- `validation_logits.mat`: validation logits used only to fit temperature.
- `test_outputs.mat`: locked-test logits and fidelity predictions.
- `temperature.mat`: validation-fitted scalar temperature.

## Interpretation of outputs
The calibrated classification probability and the auxiliary fidelity score are different quantities. Temperature scaling is applied only to classification logits. Fidelity regression is evaluated only for observations with expert-derived fidelity targets. A high fidelity estimate is not a probability of autograph status, and a high catalogue-status probability is not a fidelity score.

## Implementation defaults not numerically fixed by the manuscript
The manuscript specifies bounded affine/rotation augmentation, composition-dependent flipping, content-preserving cropping, and moderate photometric perturbation, but it does not provide every numerical probability/bound. Those values are exposed in `config/reliartConfig.m` and explicitly labelled as repository defaults. The manuscript also states that L2 weight decay was used, but the numerical value is not recoverable from the supplied final manuscript; the configuration therefore leaves it at zero rather than attributing an unsupported value to the manuscript. If the original experimental values are available, replace these defaults before attempting exact numerical reproduction.

## Citation
If you use this implementation, please cite the associated manuscript/article and the software release. Citation metadata are provided in `CITATION.cff`.

## License
The code is released under the MIT License. Dataset and artwork-image rights are separate and are not granted by the software license.
