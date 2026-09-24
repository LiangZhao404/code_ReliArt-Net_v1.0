clear; clc; addpath(genpath(fileparts(fileparts(mfilename('fullpath')))));
S=load(fullfile('artifacts','best_model.mat')); I=preprocessReliArtImage('example.jpg',S.cfg.data.inputSize(1:2));
X=dlarray(reshape(I,size(I,1),size(I,2),3,1),'SSCB'); [z,r]=reliartForward(X,S.params,S.cfg,false);
if exist(fullfile('artifacts','temperature.mat'),'file'), C=load(fullfile('artifacts','temperature.mat')); P=calibratedProbabilities(extractdata(z),C.T); else, P=softmax(extractdata(z),1); end
disp(P); fprintf('Visual-fidelity estimate: %.4f\n',extractdata(r));
