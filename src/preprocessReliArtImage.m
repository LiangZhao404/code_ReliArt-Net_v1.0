function I = preprocessReliArtImage(I, targetSize)
% Direct resizing used by ReliArt-Net. No aspect-ratio preserving padding/cropping.
arguments
    I
    targetSize (1,2) double = [512 512]
end
if ischar(I) || isstring(I), I = imread(I); end
if size(I,3)==1, I = repmat(I,1,1,3); end
if size(I,3)>3, I = I(:,:,1:3); end
I = im2single(I);
I = imresize(I,targetSize,"bilinear","Antialiasing",true);
% Pixel-intensity normalization to a stable image range.
I = min(max(I,0),1);
end
