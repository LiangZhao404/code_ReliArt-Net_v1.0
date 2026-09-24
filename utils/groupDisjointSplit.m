function split=groupDisjointSplit(T,groupColumns,ratios,seed)
%GROUPDISJOINTSPLIT Keep each combined group wholly within one partition.
rng(seed); key=strings(height(T),1);
for c=1:numel(groupColumns), key=key+"|"+string(T.(groupColumns(c))); end
u=unique(key); u=u(randperm(numel(u))); n=numel(u); ntr=floor(ratios(1)*n); nv=floor(ratios(2)*n);
tr=u(1:ntr); va=u(ntr+1:ntr+nv); split=repmat("test",height(T),1); split(ismember(key,tr))="train"; split(ismember(key,va))="validation";
end
