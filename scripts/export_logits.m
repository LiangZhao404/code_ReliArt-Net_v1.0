clear; clc; addpath(genpath(fileparts(fileparts(mfilename('fullpath')))));
S=load(fullfile('artifacts','best_model.mat')); cfg=S.cfg; params=S.params; T=loadMetadata(fullfile('data','metadata_with_split.csv'));
for name=["validation","test"]
 U=T(T.split==name,:); [logits,fidelityPred]=runInference(U,params,cfg); labels=double(U.catalogue_label)'; if min(labels)==0,labels=labels+1;end
 fidelityTarget=single(U.fidelity_target)'; fidelityMask=single(U.fidelity_mask)';
 save(fullfile('artifacts',name+"_outputs.mat'),'logits','labels','fidelityPred','fidelityTarget','fidelityMask');
 if name=="validation", save(fullfile('artifacts','validation_logits.mat'),'logits','labels'); end
end
