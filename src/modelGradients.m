function [loss,grads,stats] = modelGradients(params,X,Y,R,M,cfg)
[logits,fid]=reliartForward(X,params,cfg,true);
P=softmax(logits,1); idx=sub2ind(size(P),double(Y),1:numel(Y));
Lcls=-mean(log(max(P(idx),1e-7)));
Lfid=maskedFidelityMSE(fid,R,M);
loss=Lcls+cfg.training.lambdaFidelity*Lfid;
grads=dlgradient(loss,params);
stats.classificationLoss=Lcls; stats.fidelityLoss=Lfid;
end
