function [logits,fidelity,aux] = reliartForward(X,params,cfg,training)
%RELIARTFORWARD Full multi-branch CNN -> patch Transformer -> adaptive fusion.
if nargin<4, training=false; end
ft=cnnBranchForward(X,params.branches.texture,cfg,training);
fc=cnnBranchForward(X,params.branches.color,cfg,training);
fp=cnnBranchForward(X,params.branches.composition,cfg,training);
C=cat(3,ft,fc,fp); C=conv1x1(C,params.branchFuse.W,params.branchFuse.b); C=relu(C);
% True 16x16 non-overlapping patch embedding from spatial CNN feature map.
Z=patchEmbed(C,params.patch.W,params.patch.b,cfg.model.transformer.patchSize(1));
Z=transformerEncoder(Z,params.transformer,cfg,training);
T=squeeze(mean(Z,2)); if isvector(T), T=reshape(T,[],1); end
Cvec=squeeze(mean(C,[1 2])); if isvector(Cvec), Cvec=reshape(Cvec,[],1); end
[F,alpha]=adaptiveFusion(Cvec,T,params);
hc=relu(params.cls.W1*F+params.cls.b1); logits=params.cls.W2*hc+params.cls.b2;
hf=relu(params.fid.W1*F+params.fid.b1); fidelity=sigmoid(params.fid.W2*hf+params.fid.b2);
aux.alpha=alpha; aux.fused=F; aux.cnn=Cvec; aux.transformer=T;
end
function Y=cnnBranchForward(X,p,cfg,training)
Y=relu(dlconv(X,p.c1.W,p.c1.b,'Padding','same')); Y=maxpool(Y,[2 2],'Stride',[2 2]);
Y=relu(dlconv(Y,p.c2.W,p.c2.b,'Padding','same')); Y=maxpool(Y,[2 2],'Stride',[2 2]);
Y=relu(dlconv(Y,p.c3.W,p.c3.b,'Padding','same','DilationFactor',[2 2]));
Y=relu(dlconv(Y,p.c4.W,p.c4.b,'Padding','same','DilationFactor',[4 4]));
Y=relu(dlconv(Y,p.term.W,p.term.b,'Padding','same'));
Y=dropoutLocal(Y,cfg.model.cnn.dropout,training);
Y=relu(dlconv(Y,p.proj.W,p.proj.b,'Padding','same'));
% Pyramid pooling at manuscript scales. Each pooled map is resized and averaged
% with the base map, preserving D channels and spatial dimensions.
base=Y; acc=base; scales=cfg.model.cnn.ppmScales;
for s=scales
    P=adaptiveAveragePool(base,s);
    P=resizeFeature(P,size(base,1),size(base,2));
    acc=acc+P;
end
Y=acc/(numel(scales)+1);
end
function Y=adaptiveAveragePool(X,s)
H=size(X,1); W=size(X,2); C=size(X,3); B=size(X,4);
Y=zeros(s,s,C,B,'like',X);
for iy=1:s
 y1=floor((iy-1)*H/s)+1; y2=ceil(iy*H/s);
 for ix=1:s
  x1=floor((ix-1)*W/s)+1; x2=ceil(ix*W/s);
  Y(iy,ix,:,:)=mean(X(y1:y2,x1:x2,:,:),[1 2]);
 end
end
end
function Y=resizeFeature(X,H,W)
% Bilinear interpolation for dlarray-compatible numeric data.
raw=extractdata(X); B=size(raw,4); C=size(raw,3); out=zeros(H,W,C,B,'single');
for b=1:B, for c=1:C, out(:,:,c,b)=imresize(raw(:,:,c,b),[H W],'bilinear'); end, end
Y=dlarray(out,'SSCB');
end
function Y=conv1x1(X,W,b), Y=dlconv(X,W,b,'Padding','same'); end
function Z=patchEmbed(X,W,b,p)
Y=dlconv(X,W,b,'Stride',[p p]); % Ht x Wt x D x B
H=size(Y,1); Wd=size(Y,2); D=size(Y,3); B=size(Y,4);
Z=permute(Y,[3 1 2 4]); Z=reshape(Z,D,H*Wd,B);
end
function Y=dropoutLocal(X,p,training)
if training && p>0, mask=rand(size(X),'like',extractdata(X))>p; Y=X.*dlarray(single(mask),'SSCB')/(1-p); else, Y=X; end
end
function y=sigmoid(x), y=1./(1+exp(-x)); end
