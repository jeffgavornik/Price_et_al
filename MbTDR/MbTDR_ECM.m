function [W,S,B,D,numParams,CnInv] = MbTDR_ECM(Z,X,stimDim,R,trials,NR,Sinit)
% MbTDR_ECM.m
%  ECME and maximum marginal likelihood for dimensionality reduction model 
%   on neural data, based on "Model-Based Targeted Dimensionality Reduction 
%   for Neuronal Population Data" Aoi & Pillow 2018
%
%
%INPUTS: Z - neural data, a firing rate matrix of size N-M, 
%         for N neurons/units, M for total trials (or time points)
%        X - cell array P-1, for P predictors, each cell contains
%          a matrix of size M-d, containing the trial-dependent predictor
%          stimulus (the value of the predictor P at each timepoint M)
%        stimDim - cell array P-1, for P predictors, each contains a 2-1
%          vector indicating the size of the corresponding dimension's
%          predictor (in most cases, T-1, for T timepoints within a trial)
%        R - P-1 vector indicating the dimensionality of the model for each
%          predictor
%        trials - a matrix N-M, indicating for each of the N neurons which
%          trials that neuron participated in
%        NR - the number of neurons related to each predictor (in most
%          cases, N, the total number of neurons)
%        
%  (optional)
%        Sinit - estimate of S for initialization
%
%OUTPUTS: W - the "neuron factors", the neuron-dependent mixing weights, a
%          cell array of size P-1, a matrix of size N-R in each cell, 
%          rp being the rank of the matrix 
%         S - a common set of basis factors, a cell array P-1, 
%             with a matrix of size d-R in each cell
%         B - baseline firing rates, a vector of size N-1
%         D - firing rate variance, a vector of size N-1
%         numParams - total number of tunable parameters in the model
%         CnInv - cell array of N-1, with the precision of the posterior on
%           the W for each neuron/unit
%
%Created: 2020/04/02
% Byron Price
%Updated: 2021/10/14
% By: Byron Price

trials = logical(trials);
[N,M] = size(Z);

P = length(R);

d = zeros(P,1);
for pp=1:P
    d(pp) = prod(stimDim{pp});
end


fprintf('Computing Stimulus Sufficient Statistics ...\n');
[X,~,~] = ComputeSuffStats(X,P,R);
XtX = ComputeNeuronSpecCov(X,N,trials);

fprintf('Initializing Parameters ...\n');
if nargin<7
    [S,B,D] = InitializeParams(Z,d,N,R);
else
    [S,B,D] = InitializeParams(Z,d,N,R,Sinit);
end

numIter = 5e3;numSkip = 1;
tolerance = 1e-3; % when to stop ascent based on likelihood improvement
zeroInds = find(S==0);
fprintf('Fitting Model ...\n');

% run ECME to maximize marginal posterior (ignoring W)
CnInv = cell(N,1);
[CnInv] = GetWExpectationFull(S,B,D,Z,CnInv,X,XtX,N,R,trials,0);
% [S,B,D,CnInv] = RunECME(S,B,D,Z,CnInv,X,XtX,N,R,zeroInds,trials);
oldHeldOutLikelihood = GetHeldOutLikelihood(S,B,D,CnInv,N,X,Z,trials);

for ii=1:50
    if nargin<7
        [SS,BB,DD] = InitializeParams(Z,d,N,R);
    else
        [SS,BB,DD] = InitializeParams(Z,d,N,R,Sinit);
    end
    CnInv2 = cell(N,1);
    CnInv2 = GetWExpectationFull(SS,BB,DD,Z,CnInv2,X,XtX,N,R,trials,0);
    [SS,BB,DD,CnInv2] = RunECME(SS,BB,DD,Z,CnInv2,X,XtX,N,R,zeroInds,trials);
    oldHeldOutLikelihood2 = GetHeldOutLikelihood(SS,BB,DD,CnInv2,N,X,Z,trials);
    
    if oldHeldOutLikelihood2>oldHeldOutLikelihood
        S = SS;
        B = BB;
        D = DD;
        CnInv = CnInv2;
        oldHeldOutLikelihood = oldHeldOutLikelihood2;
    end
end

% prevS = S;inds = find(S);

likelyDiff = Inf;
for iter=1:numIter

    [S,B,D,CnInv] = RunECME(S,B,D,Z,CnInv,X,XtX,N,R,zeroInds,trials);

     if mod(iter,numSkip)==0
        heldOutLikelihood = GetHeldOutLikelihood(S,B,D,CnInv,N,X,Z,trials);

        likelyDiff = heldOutLikelihood-oldHeldOutLikelihood;

        oldHeldOutLikelihood = heldOutLikelihood;
        plot(iter/numSkip,heldOutLikelihood,'.');hold on;pause(1/100);
% %         plot(iter/numSkip,logPost,'.');pause(1/100);
     end

    if likelyDiff<tolerance
        break;
    end
end

% minimization of negative marginal log likelihood with fmincon

sSize = size(S);
x0 = ConvertParams(S,B,D,R,stimDim);
XtX = ComputeNeuronSpecCov(X,N,trials);

x0(1:N) = log(x0(1:N));

% fun = @(x) ObjectiveFun(x,N,R,X,XtX,Z,sSize,stimDim,trials);
% 
% options = optimoptions('fminunc','Display','off','Algorithm','trust-region',...
%     'MaxIterations',1500,'SpecifyObjectiveGradient',true,'CheckGradients',false);
% x = fminunc(fun,x0,options);

% alpha = 1e-4;
% [x0] = BlockAscent(x0,N,R,X,XtX,Z,sSize,stimDim,trials,alpha); % block coordinate ascent

fun = @(x) ObjectiveFun(x,N,R,X,XtX,Z,sSize,stimDim,trials);

options = optimoptions('fminunc','Display','off','Algorithm','trust-region',...
    'MaxIterations',1000,'SpecifyObjectiveGradient',true,'CheckGradients',false);
x0 = fminunc(fun,x0,options);

disp('Trust Region Complete');

x(1:N) = exp(x(1:N));
[S,B,D] = ConvertParamsBack(x,N,R,sSize,stimDim);

% reshape
newS = cell(P,1);

rowIndex = 1;
columnIndex = 1;
for pp=1:P
    if R(pp)>0
        newS{pp} = S(rowIndex:rowIndex+d(pp)-1,columnIndex:columnIndex+R(pp)-1);
        rowIndex = rowIndex+d(pp);
        columnIndex = columnIndex+R(pp);
    end
end
S = blkdiag(newS{:});

% get W and convert back to more intuitive format
[CnInv,W] = GetWExpectationFull(S,B,D,Z,CnInv,X,XtX,N,R,trials,1);

numParams = 2*N+sum(d(:).*R(:))+sum(NR.*R(:));

newW = cell(P,1);

rowIndex = 1;
columnIndex = 1;
for pp=1:P
    if R(pp)>0
        
        tmpW = zeros(N,R(pp));
        for nn=1:N
            tmpW(nn,:) = W{nn}(columnIndex:columnIndex+R(pp)-1);
        end
        newW{pp} = tmpW;
        rowIndex = rowIndex+d(pp);
        columnIndex = columnIndex+R(pp);
    end
end

W = newW;
S = newS;
end

function [S,B,D,CnInv] = RunECME(S,B,D,Z,CnInv,X,XtX,N,R,inds,trials)

XtZB = zeros(size(X,2),N);

for nn=1:N
    XtZB(:,nn) = X(trials(nn,:),:)'*(Z(nn,trials(nn,:))'-B(nn));
end

[CnInv] = GetWExpectationFull(S,B,D,Z,CnInv,X,XtX,N,R,trials,0);

[D] = CM_D(S,B,D,CnInv,N,Z,XtZB,XtX,trials);

[CnInv] = GetWExpectationFull(S,B,D,Z,CnInv,X,XtX,N,R,trials,0);

[S] = CM_S(S,D,CnInv,XtZB,XtX,N,inds);

[CnInv] = GetWExpectationFull(S,B,D,Z,CnInv,X,XtX,N,R,trials,0);

[B] = CM_B(S,B,D,CnInv,Z,X,N,trials);

[CnInv] = GetWExpectationFull(S,B,D,Z,CnInv,X,XtX,N,R,trials,0);

end


function [CnInv,muW] = GetWExpectationFull(S,B,D,Z,CnInv,X,XtX,N,R,trials,getMu)

if getMu
    muW = cell(N,1);
    U = X*S;
else
    muW = 0;
end

Ir = ones(sum(R),1);

for nn=1:N
    UtU = S'*XtX{nn}*S;
    [V,Q] = eig(UtU);
    
    diagQQ = diag(Q);
    CnInv{nn}(:,:) = bsxfun(@times,V,1./(Ir+diagQQ./D(nn))')*V';
%     Cn{nn} = diag(Ir)+UtU./D(nn);
    if getMu
        muW{nn} = CnInv{nn}*(U(trials(nn,:),:)'./D(nn))*(Z(nn,trials(nn,:))'-B(nn));
    end
end

end

function [D] = CM_D(S,B,D,CnInv,N,Z,XtZB,XtX,trials)

StRS = zeros(size(S,2));
% traceProd = @(A,B) A(:).'*reshape(B.',[],1)

tmp = S'*XtZB;
for nn=1:N
%     tmp(:) = S'*(XtZ(:,nn)-X'*(B(nn).*mOnes));
    UtU = S'*XtX{nn}*S;
    StRS(:,:) = (tmp(:,nn)*tmp(:,nn)');

    c0 = (Z(nn,trials(nn,:))-B(nn))*(Z(nn,trials(nn,:))'-B(nn));
    c1 = traceProd(UtU,CnInv{nn});
    c2 = (1/(D(nn)*D(nn)))*traceProd(StRS*CnInv{nn},UtU*CnInv{nn});
    c3 = (-2/D(nn))*traceProd(StRS,CnInv{nn});
    
    D(nn) = max((c0+c1+c2+c3)/sum(trials(nn,:)),0);
end

end

function [S] = CM_S(S,D,CnInv,XtZB,XtX,N,inds)

% LHS = zeros(size(S'));
% RHS = zeros(size(S,2));
% CnInvStRn = zeros(size(S'));
% 
% for nn=1:N
%    tmp(:) = XtZ(:,nn)-X'*(B(nn).*mOnes);
%    Rn = tmp*tmp';
%    CnInvStRn(:,:) = (CnInv{nn}*S'*XtZB(:,nn))*XtZB(:,nn)';
%    LHS(:,:) = LHS+(1/(D(nn)*D(nn)))*(CnInvStRn);
%    RHS(:,:) = RHS+(1/D(nn))*(CnInv{nn}+(1/(D(nn)*D(nn)))*CnInvStRn*S*CnInv{nn});
% end
% 
% tmp = (LHS/(XtX*mean(covMultiplier)))';%tmp(inds) = 0;
% S(:,:) = (RHS\tmp')'; % or *XtXinv
% S(inds) = 0;
% 
% % St = kron(XtX,RHS)\vec(LHS); % if matrix normal prior
% %                            % (kron(XtX,RHS)+kron(Uinv,Vinv))\vec(LHS)
% % S = reshape(St,size(S'))';

LHS = zeros(size(S));
RHS = zeros(numel(S));
RnSCnInv = zeros(size(S));

for nn=1:N
%    tmp(:) = XtZ(:,nn)-X'*(B(nn).*mOnes);
%    Rn = tmp*tmp';
   RnSCnInv(:,:) = ((CnInv{nn}*S'*XtZB(:,nn))*XtZB(:,nn)')';
   LHS(:,:) = LHS+(1/(D(nn)*D(nn)))*(RnSCnInv);
   RHS(:,:) = RHS+kron(((1/D(nn))*(CnInv{nn}+(1/(D(nn)*D(nn)))*RnSCnInv'*S*CnInv{nn}))',XtX{nn});
end

% tmp = LHS/RHS;
% vecS = RHS\LHS(:); % RHS*S = LHS(:) 
lb = -Inf.*ones(size(S));
ub = Inf*ones(size(S));
lb(inds) = 0;ub(inds) = 0;
options = optimoptions('lsqlin','Display','off');
vecS = lsqlin(RHS,LHS(:),[],[],[],[],lb(:),ub(:),S(:),options);
% vecS = kron(XtXsum',RHS)\LHS(:);
S(:,:) = reshape(vecS,size(S));
S(inds) = 0;
end

function [B] = CM_B(S,B,D,CnInv,Z,X,N,trials)

U = X*S;
% Im = eye(M);
for nn=1:N
    % slow
%     DInv = Im./D(nn)-(U*CnInv{nn}*U')./(D(nn)*D(nn));
%     B(nn) = (Z(nn,:)*DInv*oneVec)/(oneVec'*DInv*oneVec);

    % fast
    oneVec = ones(sum(trials(nn,:)),1);

    tmp = U(trials(nn,:),:)*chol(CnInv{nn})';

    DInv = -(tmp*(tmp'*oneVec))./(D(nn)*D(nn))+oneVec./D(nn);
    B(nn) = (Z(nn,trials(nn,:))*DInv)/(oneVec'*DInv);
end

end

function [x] = ConvertParams(S,B,D,R,stimDim)
x = [D;B];

columnIndex = 1;
rowIndex = 1;
for pp=1:length(R)
    sLen = prod(stimDim{pp});
    if R(pp)>0
        for rr=1:R(pp)
            x = [x;S(rowIndex:rowIndex+sLen-1,columnIndex)];
            columnIndex = columnIndex+1;
        end
        rowIndex = rowIndex+sLen;
    end
end

end

function [S,B,D] = ConvertParamsBack(x,N,R,sSize,stimDim)
D = x(1:N);
B = x(N+1:2*N);

Sprime = x(2*N+1:end);

S = zeros(sSize);
rowIndex = 1;
columnIndex = 1;
xIndex = 1;
for pp=1:length(R)
    sLen = prod(stimDim{pp});
    
    if R(pp)>0
        for rr=1:R(pp)
            
            S(rowIndex:rowIndex+sLen-1,columnIndex) = Sprime(xIndex:xIndex+sLen-1);
            columnIndex = columnIndex+1;
            xIndex = xIndex+sLen;
        end
        rowIndex = rowIndex+sLen;
    end
end

end

function [T] = traceProd(A,B)
T = A(:).'*reshape(B.',[],1);
end

function [x] = BlockAscent(x,N,R,X,XtX,Z,sSize,stimDim,trials,alpha)
CnInv = cell(N,1);
gNorm = Inf;
tol = 1e-4;
nBlocks = 2+sum(R);

blockInds = cell(nBlocks,1);
blockInds{1} = 1:N;
blockInds{2} = N+1:2*N;

count = 2;
for pp=1:length(R)
    if R(pp)>0
        for rr=1:R(pp)
            blockInds{count+1} = blockInds{count}(end)+1:blockInds{count}(end)+stimDim{pp}(1);
            count = count+1;
        end
    end
end
maxIter = 1000;
alpha = alpha.*ones(nBlocks,1);
iter = 0;
while gNorm>tol && iter<maxIter
    iter = iter+1;
    [f,g] = ObjectiveFun(x,N,R,X,XtX,Z,sSize,stimDim,trials);
    for bb=randperm(nBlocks,nBlocks)
        xtmp = x;
        xtmp(blockInds{bb}) = x(blockInds{bb})-alpha(bb)*g(blockInds{bb});
        
        [S,B,D] = ConvertParamsBack(xtmp,N,R,sSize,stimDim);
        D = exp(D);
        CnInv = GetWExpectationFull(S,B,D,Z,CnInv,X,XtX,N,R,trials,0);
        
        loglikelihood = GetHeldOutLikelihood(S,B,D,CnInv,N,X,Z,trials);
        
        
        if loglikelihood>-f
            while loglikelihood>-f
                f = -loglikelihood;
                x = xtmp;
                alpha(bb) = alpha(bb)*2;
                xtmp(blockInds{bb}) = x(blockInds{bb})-alpha(bb)*g(blockInds{bb});
                
                [S,B,D] = ConvertParamsBack(xtmp,N,R,sSize,stimDim);
                D = exp(D);
                CnInv = GetWExpectationFull(S,B,D,Z,CnInv,X,XtX,N,R,trials,0);
                
                loglikelihood = GetHeldOutLikelihood(S,B,D,CnInv,N,X,Z,trials);
               
            end
        else
            alpha = alpha*0.5;
        end
    end
    gNorm = norm(g);
    alpha = min(max(alpha,1e-7),1e-1);
end

end

function [f,g] = ObjectiveFun(x,N,R,X,XtX,Z,sSize,stimDim,trials)

[S,B,D] = ConvertParamsBack(x,N,R,sSize,stimDim);

D = exp(D);

U = X*S;
Ir = ones(size(S,2),1);Ir2 = eye(size(S,2));
Ip = eye(size(S,1));IpInds = find(Ip);clear Ip;


loglikelihood = 0;
g = zeros(size(x));
dInds = 1:N;
bInds = N+1:2*N;
sInds = 2*N+1:length(x);

traceProd = @(A,B) A(:).'*reshape(B.',[],1);

Sprime = zeros(size(S));
for nn=1:N
    Mn = sum(trials(nn,:));
    oneVec = ones(Mn,1);
    
    XtXS = XtX{nn}*S;
    UtU = S'*XtXS;
    [V,Q] = eig(UtU);
    diagQQ = diag(Q);

    Q = Z(nn,trials(nn,:))'-B(nn);
    XtQ = X(trials(nn,:),:)'*Q;
    UtQ = S'*XtQ;
    
    % get likelihood
     CnInv = bsxfun(@times,V,1./(Ir+diagQQ./D(nn))')*V';

   % DnInv = Im./D(nn)-(U*CnInv*U')./(D(nn)*D(nn));
    logDnDet = Mn*log(D(nn))+2*sum(log(diag(chol(Ir2+UtU./D(nn)))));
    
    QtUCnInvUtQ = UtQ'*CnInv*UtQ;
    QtQ = Q'*Q;
    Dn2 = D(nn)*D(nn);
    loglikelihood = loglikelihood-0.5*(logDnDet)-...
        0.5*QtQ./D(nn)+0.5*(QtUCnInvUtQ)./(Dn2);
    
    % get gradients
%     diagD = oneVec./D(nn)-sum(tmp'.*tmp',2)./Dn2;
%     g(bInds(nn)) = -Z(nn,:)*diagD+B(nn)*sum(diagD);
    tmp = chol(CnInv)*U(trials(nn,:),:)';
    DInv = -(tmp'*(tmp*oneVec))./Dn2+oneVec./D(nn);
%     DInv(ImInds) = 1/D(nn)+DInv(ImInds);DInv = DInv*oneVec;
    g(bInds(nn)) = -Z(nn,trials(nn,:))*DInv+(oneVec'*DInv)*B(nn);
    
    CnInvUtQ = CnInv*UtQ;
    CnInvUtQQtUCnInv = CnInvUtQ*CnInvUtQ'/(Dn2*Dn2);
    g(dInds(nn)) = 0.5*Mn/D(nn)-0.5*QtQ/Dn2-...
        (0.5/Dn2)*traceProd(CnInv,UtU)+(QtUCnInvUtQ)./(D(nn)*Dn2)-...
        0.5*traceProd(CnInvUtQQtUCnInv,UtU);
    
    tmp = CnInv*(XtXS'./D(nn)); % size is total rank by total predictors
    
  %  tmp2 = (Ip-S*tmp)./Dn2; % total predictors by total predictors
    tmp2 = -S*tmp./Dn2;tmp2(IpInds) = 1/Dn2+tmp2(IpInds);
%     CnInvStR = (CnInv*S'*X'*Q)*Q'*X;
    Sprime = Sprime+(tmp-CnInvUtQ*(XtQ'*tmp2))';
end

[xprime] = ConvertParams(Sprime,B,D,R,stimDim);

g(dInds) = g(dInds).*D;
g(sInds) = xprime(sInds);
f = -loglikelihood;

end


function [loglikelihood] = GetHeldOutLikelihood(S,B,D,CnInv,N,X,Z,trials)
% marginal likelihood

U = X*S;
% [V,Q] = eig(UtU);
Ir = ones(size(S,2),1);
% diagQQ = diag(Q);

% Im = eye(Mvalid);

loglikelihood = 0;
for nn=1:N
%     CnInv = bsxfun(@times,V,1./(Ir+diagQQ./D(nn))')*V';
    
   % DnInv = Im./D(nn)-(U*CnInv*U')./(D(nn)*D(nn));
    UtU = U(trials(nn,:),:)'*U(trials(nn,:),:);

    logDnDet = 2*sum(log(diag(chol(diag(Ir)+UtU./D(nn)))))+sum(trials(nn,:))*log(D(nn));
    Q = Z(nn,trials(nn,:))'-B(nn);
    tmp = chol(CnInv{nn})*U(trials(nn,:),:)'*Q;
    loglikelihood = loglikelihood-0.5*logDnDet-...
        0.5*(Q'*Q)./D(nn)+0.5*(tmp'*tmp)./(D(nn)*D(nn));
end
end

function [S,B,D] = InitializeParams(Z,d,N,R,Sinit)
P = length(R);
S = cell(P,1);
D = zeros(N,1);
fprintf('... Initializing...\n');
if nargin<=4
    for pp=1:P
        if R(pp)>0
            S{pp} = zeros(d(pp),R(pp));
            for rr=1:R(pp)
                S{pp}(:,rr) = normrnd(0,1,[d(pp),1]);
            end
        end
    end
else
    for pp=1:P
        if R(pp)>0
            oldRank = size(Sinit{pp},2);
            if R(pp)==oldRank && oldRank>0
                S{pp} = Sinit{pp}+normrnd(0,0.01,[d(pp),oldRank]);
            elseif R(pp)>oldRank && oldRank>0
                tmp = normrnd(0,1,[d(pp),1]);
                S{pp} = [tmp,Sinit{pp}+normrnd(0,0.01,[d(pp),oldRank])];
            else
                tmp = normrnd(0,1,[d(pp),R(pp)]);
                S{pp} = tmp;
            end
            [UU,SS,~] = svd(S{pp},'econ');
            S{pp}(:,:) = UU*SS;
        end
    end
end

Q = Z';

B = mean(Z,2);
for nn=1:N
   Q(:,nn) = Q(:,nn)-B(nn);
   D(nn) = var(Q(:,nn));
end

S = blkdiag(S{:});

end

function [XtX] = ComputeNeuronSpecCov(X,N,trials)

XtX = cell(N,1);
for nn=1:N
    XtX{nn} = X(trials(nn,:),:)'*X(trials(nn,:),:);
end
end

function [X,XtX,XtXInv] = ComputeSuffStats(X,P,R)

if R(1)>0
    newX = X{1};
    
    for pp=2:P
        if R(pp)>0
            newX = [newX,X{pp}];
        end
    end

else
    newX = X{2};
    for pp=3:P
        if R(pp)>0
            newX = [newX,X{pp}];
        end
    end
end

X = newX;

XtX = X'*X;

XtXInv = pinv(XtX);

end

