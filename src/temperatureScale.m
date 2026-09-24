function T = temperatureScale(logits, labels)
% Fit one scalar T>0 on validation logits by minimizing classification NLL.
% logits: C x N; labels: 1 x N integer class indices (1..C).
obj=@(u) nll(exp(u),logits,labels);
u=fminsearch(obj,0); T=exp(u);
end
function L=nll(T,z,y)
p=softmax(z./T,1); idx=sub2ind(size(p),double(y),1:numel(y));
L=-mean(log(max(p(idx),eps)));
end
