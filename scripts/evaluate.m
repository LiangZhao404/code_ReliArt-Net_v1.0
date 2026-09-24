clear; clc; addpath(genpath(fileparts(fileparts(mfilename('fullpath')))));
S=load(fullfile('artifacts','test_outputs.mat')); C=load(fullfile('artifacts','temperature.mat'));
P=calibratedProbabilities(S.logits,C.T); M=evaluateClassification(S.labels,P); disp(M)
if isfield(S,'fidelityMask')
 m=logical(S.fidelityMask); e=S.fidelityPred(m)-S.fidelityTarget(m);
 fprintf('Fidelity MSE=%.6f, MAE=%.6f (n=%d rated observations)\n',mean(e.^2),mean(abs(e)),sum(m));
 if sum(m)>1, fprintf('Fidelity Pearson r=%.6f\n',corr(double(S.fidelityPred(m))',double(S.fidelityTarget(m))','Type','Pearson')); end
end
