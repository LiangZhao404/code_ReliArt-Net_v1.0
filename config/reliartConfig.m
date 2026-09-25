function cfg = reliartConfig()
%RELIARTCONFIG Manuscript-grounded ReliArt-Net configuration (MATLAB R2025b, Version 25.2).
cfg.environment.matlab = "R2025b";
cfg.environment.matlabVersion = "25.2";
cfg.environment.toolboxes = ["Deep Learning Toolbox","Image Processing Toolbox"];

cfg.data.inputSize = [512 512 3];
cfg.data.wikiArtInputSize = [224 224 3];
cfg.data.split = [0.70 0.15 0.15];
cfg.data.resizeMethod = "bilinear";
cfg.data.antialiasing = true;

cfg.model.cnn.kernelSize = 3;
cfg.model.cnn.dilationRates = [2 4];
cfg.model.cnn.ppmScales = [1 2 3 6];
cfg.model.cnn.terminalFilters = 2048;
cfg.model.cnn.pretraining = "ImageNet";
cfg.model.cnn.dropout = 0.2; % Table 6, ReliArt-Net row.

cfg.model.transformer.patchSize = [16 16];
cfg.model.transformer.embeddingDim = 256;
cfg.model.transformer.numHeads = 8;
cfg.model.transformer.depth = 6;
cfg.model.transformer.ffnDim = 1024;
cfg.model.transformer.dropout = 0.1;

cfg.training.optimizer = "adamw";
cfg.training.initialLearnRate = 1e-4;
cfg.training.batchSize = 32;
cfg.training.maxEpochs = 100;
cfg.training.lambdaFidelity = 0.5;
cfg.training.earlyStoppingPatience = 15;
cfg.training.schedule = "cosine";
cfg.training.randomSeed = 42; % Repository reproducibility seed; not specified in manuscript.
% Manuscript mentions L2 weight decay but the numeric value is not recoverable
% from the supplied final file; keep zero rather than invent a manuscript value.
cfg.training.weightDecay = 0;

% Conservative implementation defaults for augmentation. The manuscript
% specifies augmentation types but not all numeric probabilities/bounds.
cfg.augmentation.rotationProbability = 0.70;
cfg.augmentation.maxRotationDeg = 5;
cfg.augmentation.flipProbability = 0.35;
cfg.augmentation.cropProbability = 0.50;
cfg.augmentation.minCropScale = 0.92;
cfg.augmentation.photometricProbability = 0.50;

cfg.calibration.method = "temperature-scaling";
cfg.calibration.objective = "classification-NLL";
end
