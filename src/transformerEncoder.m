function Z = transformerEncoder(X, params, cfg, training)
%TRANSFORMERENCODER 6-block, 8-head pre-norm Transformer encoder.
if nargin<4, training=false; end
Z=X; H=cfg.model.transformer.numHeads; D=cfg.model.transformer.embeddingDim;
assert(mod(D,H)==0,'Embedding dimension must be divisible by number of heads.');
for k=1:cfg.model.transformer.depth
    p=params(k);
    N=layernormLocal(Z);
    A=multiHeadAttention(N,p,H);
    A=dropoutToken(A,cfg.model.transformer.dropout,training);
    Z=Z+A;
    N=layernormLocal(Z);
    F=relu(pagemtimes(p.W1,N)+p.b1);
    F=dropoutToken(F,cfg.model.transformer.dropout,training);
    F=pagemtimes(p.W2,F)+p.b2;
    F=dropoutToken(F,cfg.model.transformer.dropout,training);
    Z=Z+F;
end
end
function O=multiHeadAttention(Z,p,H)
D=size(Z,1); N=size(Z,2); B=size(Z,3); dh=D/H;
Q=pagemtimes(p.Wq,Z); K=pagemtimes(p.Wk,Z); V=pagemtimes(p.Wv,Z);
Q=reshape(Q,dh,H,N,B); K=reshape(K,dh,H,N,B); V=reshape(V,dh,H,N,B);
O=zeros(dh,H,N,B,'like',Q);
for b=1:B
 for h=1:H
  q=Q(:,h,:,b); q=reshape(q,dh,N); k=reshape(K(:,h,:,b),dh,N); v=reshape(V(:,h,:,b),dh,N);
  S=(q'*k)/sqrt(dh); A=softmax(S,2); O(:,h,:,b)=reshape(v*A',[dh 1 N 1]);
 end
end
O=reshape(O,D,N,B); O=pagemtimes(p.Wo,O);
end
function Y=layernormLocal(X)
mu=mean(X,1); v=mean((X-mu).^2,1); Y=(X-mu)./sqrt(v+1e-5);
end
function Y=dropoutToken(X,p,training)
if training && p>0, m=rand(size(X),'like',extractdata(X))>p; Y=X.*dlarray(single(m))/(1-p); else, Y=X; end
end
