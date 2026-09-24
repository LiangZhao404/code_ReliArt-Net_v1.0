function P=calibratedProbabilities(logits,T)
P=softmax(logits./T,1);
end
