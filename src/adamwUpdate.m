function [params,avg,avgSq] = adamwUpdate(params,grads,avg,avgSq,iter,lr,wd)
%ADAMWUPDATE Adam moments + decoupled weight decay over nested structures.
[params,avg,avgSq]=walk(params,grads,avg,avgSq,'',iter,lr,wd);
end
function [p,a,v]=walk(p,g,a,v,path,iter,lr,wd)
fn=fieldnames(p);
for i=1:numel(fn)
 f=fn{i}; q=path+"."+f;
 if isstruct(p.(f))
  if numel(p.(f))>1
   for k=1:numel(p.(f)), [p.(f)(k),a.(f)(k),v.(f)(k)]=walk(p.(f)(k),g.(f)(k),a.(f)(k),v.(f)(k),q+string(k),iter,lr,wd); end
  else, [p.(f),a.(f),v.(f)]=walk(p.(f),g.(f),a.(f),v.(f),q,iter,lr,wd); end
 else
  if isempty(a.(f)), a.(f)=zeros(size(p.(f)),'like',p.(f)); v.(f)=zeros(size(p.(f)),'like',p.(f)); end
  beta1=.9; beta2=.999; eps0=1e-8; grad=g.(f);
  a.(f)=beta1*a.(f)+(1-beta1)*grad; v.(f)=beta2*v.(f)+(1-beta2)*(grad.^2);
  ah=a.(f)/(1-beta1^iter); vh=v.(f)/(1-beta2^iter);
  p.(f)=p.(f)-lr*ah./(sqrt(vh)+eps0);
  if wd>0 && endsWith(q,".W"), p.(f)=p.(f)-lr*wd*p.(f); end
 end
end
end
