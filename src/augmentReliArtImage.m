function I = augmentReliArtImage(I,cfg)
%AUGMENTRELIARTIMAGE Controlled training-only augmentation.
% Manuscript specifies bounded affine/rotation, non-uniform flips,
% content-preserving cropping and moderate photometric perturbation, but not
% every numeric bound/probability. Values in cfg.augmentation are therefore
% documented repository defaults, not claimed manuscript constants.
if nargin<2, cfg=reliartConfig(); end
a=cfg.augmentation;
if rand<a.rotationProbability
 angle=(2*rand-1)*a.maxRotationDeg; I=imrotate(I,angle,'bilinear','crop');
end
if rand<a.flipProbability, I=fliplr(I); end
if rand<a.cropProbability
 s=a.minCropScale+(1-a.minCropScale)*rand; h=size(I,1); w=size(I,2);
 ch=max(2,round(h*s)); cw=max(2,round(w*s)); y=randi(h-ch+1); x=randi(w-cw+1);
 I=imresize(I(y:y+ch-1,x:x+cw-1,:),[h w],'bilinear','Antialiasing',true);
end
if rand<a.photometricProbability
 H=rgb2hsv(I); H(:,:,1)=mod(H(:,:,1)+(rand-.5)*.04,1);
 H(:,:,2)=min(max(H(:,:,2)*(.9+.2*rand),0),1); H(:,:,3)=min(max(H(:,:,3)*(.9+.2*rand),0),1); I=hsv2rgb(H);
end
I=im2single(I);
end
