function [tr,va,te]=stratifiedSplit(labels,ratios,seed)
rng(seed); labels=categorical(labels); tr=false(numel(labels),1); va=tr; te=tr;
cls=categories(labels);
for k=1:numel(cls)
 idx=find(labels==cls{k}); idx=idx(randperm(numel(idx))); n=numel(idx);
 ntr=floor(ratios(1)*n); nv=floor(ratios(2)*n);
 tr(idx(1:ntr))=true; va(idx(ntr+1:ntr+nv))=true; te(idx(ntr+nv+1:end))=true;
end
end
