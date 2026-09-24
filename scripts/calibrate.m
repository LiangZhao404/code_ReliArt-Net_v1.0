clear; clc; addpath(genpath(fileparts(fileparts(mfilename('fullpath')))));
S=load(fullfile('artifacts','validation_logits.mat'));
T=temperatureScale(S.logits,S.labels);
save(fullfile('artifacts','temperature.mat'),'T');
fprintf('Validation-fitted temperature T = %.6f\n',T);
