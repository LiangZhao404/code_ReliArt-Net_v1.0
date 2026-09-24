clear; clc; addpath(genpath(fileparts(fileparts(mfilename('fullpath')))));
cfg=reliartConfig(); T=loadMetadata(fullfile('data','metadata.csv'));
[tr,va,te]=stratifiedSplit(T.catalogue_label,cfg.data.split,cfg.training.randomSeed);
T.split=repmat("",height(T),1); T.split(tr)="train"; T.split(va)="validation"; T.split(te)="test";
writetable(T,fullfile('data','metadata_with_split.csv'));
fprintf('Train=%d, Validation=%d, Test=%d\n',sum(tr),sum(va),sum(te));
