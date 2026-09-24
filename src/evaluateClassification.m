function M=evaluateClassification(yTrue,P)
[~,yPred]=max(P,[],1); yTrue=double(yTrue(:)'); yPred=double(yPred(:)'); C=size(P,1);
cm=confusionmat(yTrue,yPred,'Order',1:C); M.confusionMatrix=cm; M.accuracy=mean(yPred==yTrue);
prec=zeros(1,C); rec=zeros(1,C); spec=zeros(1,C); f1=zeros(1,C);
for c=1:C
 tp=cm(c,c); fp=sum(cm(:,c))-tp; fn=sum(cm(c,:))-tp; tn=sum(cm,'all')-tp-fp-fn;
 prec(c)=tp/max(tp+fp,1); rec(c)=tp/max(tp+fn,1); spec(c)=tn/max(tn+fp,1); f1(c)=2*prec(c)*rec(c)/max(prec(c)+rec(c),eps);
end
M.macroPrecision=mean(prec); M.macroRecall=mean(rec); M.specificity=mean(spec); M.macroF1=mean(f1); M.balancedAccuracy=mean(rec);
idx=sub2ind(size(P),yTrue,1:numel(yTrue)); M.nll=-mean(log(max(P(idx),eps)));
Y=zeros(size(P)); Y(idx)=1; M.brier=mean(sum((P-Y).^2,1));
[M.ece,M.mce]=calibrationErrors(yTrue,P,15);
if C==2
 try, [~,~,~,M.auroc]=perfcurve(yTrue,P(2,:),2); catch, M.auroc=NaN; end
else, M.auroc=NaN; end
end
function [ece,mce]=calibrationErrors(y,P,K)
[conf,pred]=max(P,[],1); ok=pred==y; edges=linspace(0,1,K+1); ece=0; mce=0;
for k=1:K
 if k<K, m=conf>=edges(k)&conf<edges(k+1); else, m=conf>=edges(k)&conf<=edges(k+1); end
 if any(m), gap=abs(mean(ok(m))-mean(conf(m))); ece=ece+mean(m)*gap; mce=max(mce,gap); end
end
end
