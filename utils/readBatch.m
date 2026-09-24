function [X,Y,R,M] = readBatch(T,idx,cfg,isTraining)
B=numel(idx); H=cfg.data.inputSize(1); W=cfg.data.inputSize(2); X=zeros(H,W,3,B,'single');
for j=1:B
 I=preprocessReliArtImage(T.image_path(idx(j)),[H W]);
 if isTraining, I=augmentReliArtImage(I,cfg); end
 X(:,:,:,j)=I;
end
X=dlarray(X,'SSCB');
% Internally use MATLAB class indices 1..2; metadata may use 0/1.
y=double(T.catalogue_label(idx)); if min(y)==0, y=y+1; end; Y=y(:)';
R=single(T.fidelity_target(idx))'; M=single(T.fidelity_mask(idx))'; R(isnan(R))=0;
end
