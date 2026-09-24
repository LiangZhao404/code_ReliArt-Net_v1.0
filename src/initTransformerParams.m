function params=initTransformerParams(cfg)
D=cfg.model.transformer.embeddingDim; F=cfg.model.transformer.ffnDim;
for k=1:cfg.model.transformer.depth
 params(k).Wq=dlarray(glorot([D D])); params(k).Wk=dlarray(glorot([D D]));
 params(k).Wv=dlarray(glorot([D D])); params(k).Wo=dlarray(glorot([D D]));
 params(k).W1=dlarray(glorot([F D])); params(k).b1=dlarray(zeros(F,1,'single'));
 params(k).W2=dlarray(glorot([D F])); params(k).b2=dlarray(zeros(D,1,'single'));
end
end
function W=glorot(sz)
lim=sqrt(6/(sz(1)+sz(2))); W=(2*rand(sz,'single')-1)*lim;
end
