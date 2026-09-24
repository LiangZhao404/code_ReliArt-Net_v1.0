clear; clc; addpath(genpath(fileparts(fileparts(mfilename('fullpath')))));
cfg=reliartConfig(); T=loadMetadata(fullfile('data','metadata.csv'));
assert(all(ismember(["artwork_family","source"],string(T.Properties.VariableNames))), 'metadata.csv needs artwork_family and source.');
T.split=groupDisjointSplit(T,["artwork_family","source"],cfg.data.split,cfg.training.randomSeed);
writetable(T,fullfile('data','metadata_group_disjoint.csv'));
