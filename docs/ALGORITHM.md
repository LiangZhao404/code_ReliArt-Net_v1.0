# ReliArt-Net algorithm mapping

The repository follows the manuscript sequence: preprocessing -> three CNN streams (Texture, Color-Tone, Composition) -> dilated convolution and pyramid pooling concept -> Transformer global-context modeling -> adaptive fusion -> catalogue-status classification head + auxiliary expert visual-fidelity regression head -> joint loss -> post-hoc temperature scaling.

Important semantic separation:
- Catalogue-status labels supervise binary classification.
- Expert visual-fidelity ratings supervise only the auxiliary regression head where ratings exist.
- Temperature scaling is fitted after training, using validation classification logits only.
- Calibrated classification probability and predicted visual fidelity are different quantities.

The final manuscript specifies MATLAB R2024a, Deep Learning Toolbox and Image Processing Toolbox; primary input 512x512; WikiArt-Baroque 224x224; AdamW, LR 1e-4, batch 32, 100 epochs, lambda=0.5; CNN dilation rates 2/4; PPM scales 1/2/3/6; Transformer patch 16x16, D=256, 8 heads, depth=6, FFN=1024, dropout=0.1; early stopping patience 15.
