function s=emptyOptimizerState(p)
% Recursively create optimizer state with empty leaves matching p.
if isstruct(p)
 s=p; fn=fieldnames(p);
 for i=1:numel(fn)
  f=fn{i};
  if numel(p.(f))>1, for k=1:numel(p.(f)), s.(f)(k)=emptyOptimizerState(p.(f)(k)); end
  else, s.(f)=emptyOptimizerState(p.(f)); end
 end
else, s=[];
end
end
