# ReliArt-Net — MATLAB R2024a reference implementation

This repository accompanies the manuscript **“Reproducibility in Art: Reliability of original and reproduced masterpieces based on ReliArt-Net.”** It implements the manuscript-described pipeline in MATLAB R2024a: preprocessing, controlled augmentation, three CNN streams, dilated convolutions, pyramid pooling, 16×16 patch embedding, a 6-block/8-head Transformer, adaptive CNN–Transformer fusion, separate catalogue-status and expert visual-fidelity heads, joint masked multi-task training, and validation-only post-hoc temperature scaling.

## Provenance and scope
This code was reconstructed from the final manuscript supplied by the authors. It is a **manuscript-grounded reference implementation**, not a claim that these files are byte-for-byte the historical source code used to generate every reported table. Numeric choices explicitly reported in the manuscript are kept as such. Where the supplied manuscript does not specify a numeric implementation detail (notably all augmentation probabilities/bounds and the recoverable numeric L2 weight-decay value), the repository labels the choice as a repository default rather than inventing a manuscript constant.

## Environment
- MATLAB R2024a
- Deep Learning Toolbox
- Image Processing Toolbox
- Statistics and Machine Learning Toolbox is useful for AUROC (`perfcurve`); evaluation degrades gracefully if unavailable.

The manuscript reports Windows 11 Pro and MATLAB R2024a as the experimental environment.

## Manuscript-specified configuration
- Primary input: 512×512 RGB; WikiArt-Baroque: 224×224.
- Direct bilinear resize with antialiasing; no aspect-ratio-preserving padding/cropping at this stage.
- Primary dataset split: 70% train / 15% validation / 15% locked test, stratified by catalogue-status label.
- CNN: 3×3 convolutions, dilation rates 2 and 4, terminal 2048 filters, PPM scales 1×1, 2×2, 3×3, 6×6, CNN dropout 0.2.
- Transformer: 16×16 patches, D=256, 8 heads, 6 blocks, FFN=1024, dropout 0.1.
- ReliArt-Net training row: AdamW, initial LR 1e-4, batch 32, 100 epochs, joint-loss λ=0.5.
- Early stopping patience: 15 epochs; cosine learning-rate schedule.
- Post-hoc calibration: fit one scalar temperature on validation classification logits by minimizing NLL with network weights frozen; apply the fitted temperature to test classification logits only.

## Data
Images are not redistributed. Create `data/metadata.csv`:

```csv
image_path,catalogue_label,fidelity_target,fidelity_mask,artwork_family,source
/path/image1.jpg,1,NaN,0,family_001,provider_A
/path/image2.jpg,0,0.875,1,family_002,provider_B
```

`catalogue_label` is the catalogue-derived binary target. `fidelity_target` is a normalized expert visual-fidelity target, not a probability of autograph authorship. Set `fidelity_mask=1` only when an expert target exists.

## Reproduce the pipeline
From MATLAB, `cd` to the repository root and run:

```matlab
addpath(genpath(pwd));
scripts/prepare_splits
scripts/train
scripts/export_logits
scripts/calibrate
scripts/evaluate
```

For the source- and artwork-family-disjoint sensitivity analysis, run `scripts/prepare_group_disjoint_splits.m`, point the training script at the generated metadata file, and retrain from initialization.

## Outputs
`artifacts/best_model.mat` stores the validation-selected checkpoint. `validation_logits.mat` is used only to fit temperature. `test_outputs.mat` stores locked-test logits and fidelity predictions. `temperature.mat` stores the validation-fitted scalar temperature.

## Important interpretation
The classification probability and auxiliary fidelity score are different quantities. Temperature scaling is applied only to classification logits. Fidelity regression is evaluated only where expert-derived fidelity targets exist. Neither output alone establishes physical authorship.

## Known manuscript-limited details
The final manuscript specifies bounded affine/rotation, composition-dependent flipping, content-preserving cropping and moderate photometric perturbation, but does not numerically specify every probability/bound. `config/reliartConfig.m` exposes conservative repository defaults for these values. The supplied manuscript also states L2 weight decay but its numeric value is not recoverable from the supplied final file; the repository therefore defaults it to zero rather than assigning an unsupported manuscript value. Authors should replace these defaults if the original experimental values are available.

## Citation and license
See `CITATION.cff`. Code is released under the MIT License. Dataset/image rights are separate and are not granted by this software license.
