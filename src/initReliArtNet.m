function params = initReliArtNet(cfg)
%INITRELIARTNET Initialize trainable parameters for ReliArt-Net.
rng(cfg.training.randomSeed);
D=cfg.model.transformer.embeddingDim; F=cfg.model.transformer.ffnDim;
params=struct;
params.branches.texture = initBranch(cfg);
params.branches.color = initBranch(cfg);
params.branches.composition = initBranch(cfg);
% 1x1 projection of concatenated 3-branch feature maps (3D -> D).
params.branchFuse.W = dlarray(glorot([D 3*D]));
params.branchFuse.b = dlarray(zeros(D,1,'single'));
% Patch embedding on D-channel CNN map, 16x16 stride 16 -> D tokens.
p=cfg.model.transformer.patchSize(1);
params.patch.W = dlarray(glorot([p p D D]));
params.patch.b = dlarray(zeros(D,1,'single'));
params.transformer = initTransformerParams(cfg);
% Alignment + attention fusion.
params.alignCNN.W=dlarray(glorot([D D])); params.alignCNN.b=dlarray(zeros(D,1,'single'));
params.alignTR.W=dlarray(glorot([D D])); params.alignTR.b=dlarray(zeros(D,1,'single'));
params.gate.W=dlarray(glorot([2 2*D])); params.gate.b=dlarray(zeros(2,1,'single'));
% Dual heads.
params.cls.W1=dlarray(glorot([D D])); params.cls.b1=dlarray(zeros(D,1,'single'));
params.cls.W2=dlarray(glorot([2 D])); params.cls.b2=dlarray(zeros(2,1,'single'));
params.fid.W1=dlarray(glorot([D D])); params.fid.b1=dlarray(zeros(D,1,'single'));
params.fid.W2=dlarray(glorot([1 D])); params.fid.b2=dlarray(zeros(1,1,'single'));
end
function b=initBranch(cfg)
D=cfg.model.transformer.embeddingDim;
b.c1.W=dlarray(he([3 3 3 64])); b.c1.b=dlarray(zeros(64,1,'single'));
b.c2.W=dlarray(he([3 3 64 128])); b.c2.b=dlarray(zeros(128,1,'single'));
b.c3.W=dlarray(he([3 3 128 256])); b.c3.b=dlarray(zeros(256,1,'single'));
b.c4.W=dlarray(he([3 3 256 512])); b.c4.b=dlarray(zeros(512,1,'single'));
% Terminal layer specified as 2048 filters; project to D for fusion/tokens.
b.term.W=dlarray(he([1 1 512 cfg.model.cnn.terminalFilters])); b.term.b=dlarray(zeros(cfg.model.cnn.terminalFilters,1,'single'));
b.proj.W=dlarray(glorot([1 1 cfg.model.cnn.terminalFilters D])); b.proj.b=dlarray(zeros(D,1,'single'));
end
function W=he(sz)
fanIn=prod(sz(1:end-1)); W=sqrt(2/fanIn)*randn(sz,'single');
end
function W=glorot(sz)
fanIn=prod(sz(1:end-1)); fanOut=sz(end); if numel(sz)==2, fanIn=sz(2); fanOut=sz(1); end
lim=sqrt(6/(fanIn+fanOut)); W=(2*rand(sz,'single')-1)*lim;
end
