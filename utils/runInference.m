function [logits,fid] = runInference(T,params,cfg)
N=height(T); logits=zeros(2,N,'single'); fid=zeros(1,N,'single'); bs=cfg.training.batchSize;
for s=1:bs:N
 idx=s:min(N,s+bs-1); [X,~,~,~]=readBatch(T,idx,cfg,false);
 [z,r]=reliartForward(X,params,cfg,false); logits(:,idx)=gather(extractdata(z)); fid(:,idx)=gather(extractdata(r));
end
end
