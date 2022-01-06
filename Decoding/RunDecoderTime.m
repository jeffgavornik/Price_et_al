function [timeDecode,timeDecodeCond,neuronsPerTrial,elemDecodeCond] ...
    = RunDecoderTime(Z,X,B,S,W,D,R,trials,nBins,binStarts,CnInv)
% RunDecoderTime.m
%   Bayes' optimal trial stimulus decoder for model from MbTDR_ECM.m ...
%    specifically set up for sequence stimulus experiment

[N,M] = size(Z);
origM = M/nBins;
P = length(S);

Q = zeros(size(Z));

for nn=1:N
    Q(nn,trials(nn,:)) = Z(nn,trials(nn,:))-B(nn);
end

elemDecodeCond = zeros(nBins,origM);
timeDecodeCond = zeros(nBins,origM);
timeDecode = zeros(nBins,origM);
neuronsPerTrial = zeros(origM,1);
trialCount = 1;
for mm=1:origM
    trialInds = trialCount:trialCount+nBins-1;
    neuronParticipation = logical(zeros(N,1));
    for nn=1:N
       neuronParticipation(nn) = sum(trials(nn,trialInds))>0;
    end
    neuronsPerTrial(mm) = sum(neuronParticipation);
    
    tmpD = D(neuronParticipation);
    tmpQ = Q(neuronParticipation,trialInds);
    
    tmpW = W;
    tmpX = X;
    for pp=1:P
        if R(pp)>0
            tmpW{pp} = W{pp}(neuronParticipation,:);
            tmpX{pp} = X{pp}(trialInds,:);
        end
    end
    
    day1Indicator = sum(tmpW{2}(:))==0 && sum(tmpW{3}(:))==0;
    
    if nargin<11
         fun = @(x) MbTDRFull_Likelihood(x,tmpX,S,tmpW,tmpD,neuronsPerTrial(mm),nBins,tmpQ,P,R,binStarts,day1Indicator);
    else
        tmpCnInv = cell(sum(neuronParticipation),1);
        count = 0;
        for nn=1:N
            if neuronParticipation(nn)==1
               count = count+1;
               tmpCnInv{count} = CnInv{nn};
            end
        end
        fun = @(x) MbTDRMarginal_Likelihood(x,S,tmpW,tmpCnInv,tmpD,neuronsPerTrial(mm),nBins,tmpQ,P,R,binStarts,day1Indicator);
    end
    
    for ee=1:nBins
        
        lb = [1,60*pi/180,0,0,1];
        ub = [4,180*pi/180,1,1,600];
        
        mc = zeros(1e2,1);
        loglikes = zeros(nBins,1);
        for iter=1:1e2
            if day1Indicator
                x = [lb(2)+rand*(ub(2)-lb(2)),binornd(1,0.5),...
                    binornd(1,0.5),lb(5)+rand*(ub(5)-lb(5))];
            else
                x = [lb(1)+rand*(ub(1)-lb(1)),lb(2)+rand*(ub(2)-lb(2)),...
                    binornd(1,0.5),binornd(1,0.5),lb(5)+rand*(ub(5)-lb(5))];
            end
            [~,XSW,~] = MbTDRFull_Likelihood(x,tmpX,S,tmpW,tmpD,...
                neuronsPerTrial(mm),nBins,tmpQ,P,R,binStarts,day1Indicator);
            
            for ll=1:nBins
                loglikes(ll) = -0.5*sum(log(D(neuronParticipation)))...
                    -0.5*trace((tmpQ(:,ee)-XSW(ll,:)')'*...
                    diag(1./D(neuronParticipation))*(tmpQ(:,ee)-XSW(ll,:)'));
            end
            summation = LogSum(loglikes,nBins);
            prob = exp(loglikes-summation)./sum(exp(loglikes-summation));
            mc(iter) = find(mnrnd(1,prob));
        end
        nWins = zeros(nBins,1);
        for ll=1:nBins
            nWins(ll) = sum(mc==ll);
        end
        [~,ind] = max(nWins);
        timeDecode(ee,mm) = ind;
    end
    trialCount = trialCount+nBins;
    
    [~,~,XSWtrue] = MbTDRFull_Likelihood(x,tmpX,S,tmpW,tmpD,neuronsPerTrial(mm),nBins,tmpQ,P,R,binStarts,day1Indicator);
    
    
    for tt=1:nBins
        loglikes = zeros(nBins,1);
        for uu=1:nBins
            loglikes(uu) = -0.5*sum(log(D(neuronParticipation)))...
                -0.5*trace((tmpQ(:,tt)-XSWtrue(uu,:)')'*diag(1./D(neuronParticipation))*(tmpQ(:,tt)-XSWtrue(uu,:)'));
        end
        [~,ind] = max(loglikes);
        timeDecodeCond(tt,mm) = ind;
    end
    
    binsPerElem = 6;
    count = -1;
    for tt=1:nBins
        if mod(tt-1,binsPerElem)==0
           count = count+1;
        end
        loglikes = zeros(nBins,1);
        for uu=1:nBins
            loglikes(uu) = -0.5*sum(log(D(neuronParticipation)))...
                -0.5*trace((tmpQ(:,tt)-XSWtrue(uu,:)')'*diag(1./D(neuronParticipation))*(tmpQ(:,tt)-XSWtrue(uu,:)'));
        end
        range = (1:binsPerElem)+count*binsPerElem;
        [~,ind] = max(loglikes(range));
        elemDecodeCond(tt,mm) = ind;
    end

end

end

function [loglikelihood,XSW,XSWtrue] = MbTDRFull_Likelihood(x,Xtrue,S,W,D,N,M,Q,P,R,binStarts,day1Indicator)

% AOrient = 75*pi/180;
BOrient = 120*pi/180;
% COrient = 35*pi/180;
% DOrient = 160*pi/180;
% EOrient = 10*pi/180;
blockSize = 50;

% full likelihood for MbTDR model
if day1Indicator
    logd = 0;
    angleRadian = x(1);
    IB = x(2);
    IE = x(3);
    trialID = x(4);
else    
    logd = log(x(1));
    angleRadian = x(2);
    IB = x(3);
    IE = x(4);
    trialID = x(5);
end

X = cell(P,1);
firstMat = diag(ones(M,1));
X{1} = firstMat;
X{2} = logd.*firstMat;
X{3} = (logd*logd).*firstMat;

oneZeroVec = zeros(M,1);oneZeroVec(binStarts(1,4):binStarts(2,4)) = 1;
secondMat = diag(oneZeroVec);
angleDiff = angdiff(angleRadian,BOrient);
X{4} = angleDiff.*secondMat;
X{5} = (angleDiff*angleDiff).*secondMat;
X{6} = (angleDiff*logd).*secondMat;
X{7} = (angleDiff*angleDiff*logd).*secondMat;

X{8} = (IE*angleDiff).*secondMat;
X{9} = (IE*angleDiff*angleDiff).*secondMat;
X{10} = (IE*angleDiff*logd).*secondMat;
X{11} = (IE*angleDiff*angleDiff*logd).*secondMat;

X{12} = (abs(angleDiff)>=(15*pi/180)).*secondMat;
X{13} = ((abs(angleDiff)>=(15*pi/180))*logd).*secondMat;

X{14} = ((abs(angleDiff)>=(15*pi/180))*IE).*secondMat;
X{15} = ((abs(angleDiff)>=(15*pi/180))*IE*logd).*secondMat;

oneZeroVec = zeros(M,1);oneZeroVec(binStarts(1,16):binStarts(2,16)) = 1;
thirdMat = diag(oneZeroVec);

X{16} = IB.*thirdMat;
X{17} = (IB*logd).*thirdMat;
X{18} = (IB*angleDiff).*thirdMat;
X{19} = (IB*angleDiff*logd).*thirdMat;
X{20} = (IB*angleDiff*angleDiff).*thirdMat;
X{21} = (IB*angleDiff*angleDiff*logd).*thirdMat;

X{22} = ((abs(angleDiff)>=(15*pi/180))*IB).*thirdMat;
X{23} = ((abs(angleDiff)>=(15*pi/180))*IB*logd).*thirdMat;

oneZeroVecE = zeros(M,1);oneZeroVecE(binStarts(1,24):binStarts(2,24)) = 1;
fourthMat = diag(oneZeroVecE);

X{24} = (IE).*fourthMat;
X{25} = (IE*logd).*fourthMat;
X{26} = (IE*IB).*thirdMat;
X{27} = (IE*IB*logd).*thirdMat;
X{28} = ((abs(angleDiff)>=(15*pi/180))*IE*IB*logd).*thirdMat;

trialVec = log(trialID);blockVec = log(mod(trialID-1,blockSize)+1);
X{29} = (-trialVec).*firstMat;
X{30} = (-trialVec*trialVec).*firstMat;
X{31} = (-blockVec).*firstMat;
X{32} = (-blockVec*blockVec).*firstMat;
X{33} = (-blockVec*trialVec).*firstMat;

XSW = zeros(size(Q,2),N);
XSWtrue = zeros(size(Q,2),N);
for pp=1:P
    if R(pp)>0 
        X{pp} = X{pp}(:,binStarts(1,pp):binStarts(2,pp));
        XSW = XSW+X{pp}*S{pp}*W{pp}';

        XSWtrue = XSWtrue+Xtrue{pp}*S{pp}*W{pp}';
    end
end

% Im = eye(M);
% loglikelihood = 0;
% for nn=1:N
%     
%     logDnDet = M*log(D(nn));
%     tmpQ = Q(nn,:)'-XSW(:,nn);
%     loglikelihood = loglikelihood-0.5*logDnDet-...
%         0.5*(tmpQ'*tmpQ)./D(nn);
% end
loglikelihood = -0.5*M*sum(log(D))-0.5*trace((Q-XSW')'*diag(1./D)*(Q-XSW'));
loglikelihood = -loglikelihood;
end

function [loglikelihood,XSW] = MbTDRMarginal_Likelihood(x,S,W,CnInv,D,N,M,Q,P,R,binStarts,day1Indicator)

% AOrient = 75*pi/180;
BOrient = 120*pi/180;
% COrient = 35*pi/180;
% DOrient = 160*pi/180;
% EOrient = 10*pi/180;
blockSize = 50;

% full likelihood for MbTDR model
if day1Indicator
    logd = 0;
    angleRadian = x(1);
    IB = x(2);
    IE = x(3);
    trialID = x(4);
else    
    logd = log(x(1));
    angleRadian = x(2);
    IB = x(3);
    IE = x(4);
    trialID = x(5);
end

X = cell(P,1);
firstMat = diag(ones(M,1));
X{1} = firstMat;
X{2} = logd.*firstMat;
X{3} = (logd*logd).*firstMat;

oneZeroVec = zeros(M,1);oneZeroVec(binStarts(1,4):binStarts(2,4)) = 1;
secondMat = diag(oneZeroVec);
angleDiff = angdiff(angleRadian,BOrient);
X{4} = angleDiff.*secondMat;
X{5} = (angleDiff*angleDiff).*secondMat;
X{6} = (angleDiff*logd).*secondMat;
X{7} = (angleDiff*angleDiff*logd).*secondMat;

X{8} = (IE*angleDiff).*secondMat;
X{9} = (IE*angleDiff*angleDiff).*secondMat;
X{10} = (IE*angleDiff*logd).*secondMat;
X{11} = (IE*angleDiff*angleDiff*logd).*secondMat;

X{12} = (abs(angleDiff)>=(20*pi/180)).*secondMat;
X{13} = ((abs(angleDiff)>=(20*pi/180))*logd).*secondMat;

X{14} = ((abs(angleDiff)>=(20*pi/180))*IE).*secondMat;
X{15} = ((abs(angleDiff)>=(20*pi/180))*IE*logd).*secondMat;

oneZeroVec = zeros(M,1);oneZeroVec(binStarts(1,16):binStarts(2,16)) = 1;
thirdMat = diag(oneZeroVec);

X{16} = IB.*thirdMat;
X{17} = (IB*logd).*thirdMat;
X{18} = (IB*angleDiff).*thirdMat;
X{19} = (IB*angleDiff*logd).*thirdMat;
X{20} = (IB*angleDiff*angleDiff).*thirdMat;
X{21} = (IB*angleDiff*angleDiff*logd).*thirdMat;

X{22} = ((abs(angleDiff)>=(20*pi/180))*IB).*thirdMat;
X{23} = ((abs(angleDiff)>=(20*pi/180))*IB*logd).*thirdMat;

X{24} = (IE).*firstMat;
X{25} = (IE*logd).*firstMat;
X{26} = (IE*IB).*thirdMat;
X{27} = (IE*IB*logd).*thirdMat;
X{28} = ((abs(angleDiff)>=(20*pi/180))*IE*IB*logd).*thirdMat;

trialVec = log(trialID);blockVec = log(mod(trialID-1,blockSize)+1);
X{29} = (-trialVec).*firstMat;
X{30} = (-trialVec*trialVec).*firstMat;
X{31} = (-blockVec).*firstMat;
X{32} = (-blockVec*blockVec).*firstMat;
X{33} = (-blockVec*trialVec).*firstMat;

newX = X{1};
XSW = X{1}*S{1}*W{1}';
for pp=2:P
    if R(pp)>0 
        X{pp} = X{pp}(:,binStarts(1,pp):binStarts(2,pp));
        newX = [newX,X{pp}];
        XSW = XSW+X{pp}*S{pp}*W{pp}';
    end
end

U = newX*blkdiag(S{:});

Im = eye(M);

loglikelihood = 0;
for nn=1:N
    Sigma = Im*D(nn)+U*CnInv{nn}*U';
    
   logDnDet = sum(log(diag(chol(Sigma))));
    tmpQ = Q(nn,:)'-XSW(:,nn);
    loglikelihood = loglikelihood-logDnDet-...
        0.5*(tmpQ'*(Sigma\tmpQ))./D(nn);
end
loglikelihood = -loglikelihood;
end
