function L=maskedFidelityMSE(pred,target,mask)
mask=single(mask); denom=max(sum(mask,"all"),1);
L=sum(mask.*(pred-target).^2,"all")/denom;
end
