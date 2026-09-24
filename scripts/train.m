clear; clc; addpath(genpath(fileparts(fileparts(mfilename('fullpath')))));
cfg=reliartConfig(); rng(cfg.training.randomSeed);
T=loadMetadata(fullfile('data','metadata_with_split.csv'));
tr=T(T.split=="train",:); va=T(T.split=="validation",:);
params=initReliArtNet(cfg); avg=emptyOptimizerState(params); avgSq=emptyOptimizerState(params);
bestAcc=-inf; bad=0; iter=0; history=[];
if ~exist('artifacts','dir'), mkdir('artifacts'); end
for epoch=1:cfg.training.maxEpochs
 lr=cfg.training.initialLearnRate*0.5*(1+cos(pi*(epoch-1)/max(cfg.training.maxEpochs-1,1)));
 order=randperm(height(tr)); L=0; nb=0;
 for s=1:cfg.training.batchSize:height(tr)
  idx=order(s:min(s+cfg.training.batchSize-1,height(tr))); [X,Y,R,M]=readBatch(tr,idx,cfg,true);
  [loss,grads]=dlfeval(@modelGradients,params,X,Y,R,M,cfg); iter=iter+1;
  [params,avg,avgSq]=adamwUpdate(params,grads,avg,avgSq,iter,lr,cfg.training.weightDecay); L=L+double(gather(extractdata(loss))); nb=nb+1;
 end
 [z,~]=runInference(va,params,cfg); y=double(va.catalogue_label)'; if min(y)==0,y=y+1;end; [~,yp]=max(z,[],1); acc=mean(yp==y);
 history=[history; epoch L/nb acc lr]; fprintf('Epoch %3d | loss %.5f | val acc %.4f | lr %.3g\n',epoch,L/nb,acc,lr);
 if acc>bestAcc, bestAcc=acc; bad=0; save(fullfile('artifacts','best_model.mat'),'params','cfg','history','bestAcc','-v7.3'); else, bad=bad+1; end
 if bad>=cfg.training.earlyStoppingPatience, fprintf('Early stopping.\n'); break; end
end
