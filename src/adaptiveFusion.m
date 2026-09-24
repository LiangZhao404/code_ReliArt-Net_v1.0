function [F,alpha] = adaptiveFusion(C,T,params)
%ADAPTIVEFUSION Project CNN and Transformer streams to shared space and gate.
C=params.alignCNN.W*C+params.alignCNN.b;
T=params.alignTR.W*T+params.alignTR.b;
H=[C;T]; alpha=softmax(params.gate.W*H+params.gate.b,1);
F=C.*alpha(1,:)+T.*alpha(2,:);
end
