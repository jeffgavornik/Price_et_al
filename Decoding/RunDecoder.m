function [Xdecode,neuronsPerTrial] = RunDecoder(Z,B,S,W,D,R,trials,nBins,binStarts,CnInv)
% RunDecoder.m
%   Bayes' optimal trial stimulus decoder for model from MbTDR_ECM.m ...
%    specifically set up for sequence stimulus experiment

[N,M] = size(Z);
origM = M/nBins;
P = length(S);

Q = zeros(size(Z));

for nn=1:N
    Q(nn,trials(nn,:)) = Z(nn,trials(nn,:))-B(nn);
end

Xdecode = zeros(5,origM);
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
    for pp=1:P
        if R(pp)>0
            tmpW{pp} = W{pp}(neuronParticipation,:);
        end
    end
    
    day1Indicator = sum(tmpW{2}(:))==0 && sum(tmpW{3}(:))==0;
    
    if nargin<10
         fun = @(x) MbTDRFull_Likelihood(x,S,tmpW,tmpD,neuronsPerTrial(mm),nBins,tmpQ,P,R,binStarts,day1Indicator);
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
    
    nIter = 50;
    negativeLogLike = zeros(nIter,1);
    if day1Indicator
        x = zeros(4,nIter);
        for tt=1:nIter
            lb = [60*pi/180,0,0,0.5];
            ub = [180*pi/180,1,1,600.5];

            x0 = [60*pi/180+rand*120*pi/180,rand,rand,rand*600];
            
            options = optimoptions('fmincon','Display','off','Algorithm','interior-point',...
                'MaxIterations',500);
            [x(:,tt),negativeLogLike(tt)] = fmincon(fun,x0,[],[],[],[],lb,ub,[],options);
        end
    else
        x = zeros(5,nIter);
        for tt=1:nIter
            lb = [0.5,60*pi/180,0,0,0.5];
            ub = [6,180*pi/180,1,1,600.5];
            
            x0 = [0.5+rand*4,60*pi/180+rand*120*pi/180,rand,rand,rand*600];
            
            options = optimoptions('fmincon','Display','off','Algorithm','interior-point',...
                'MaxIterations',500);
            [x(:,tt),negativeLogLike(tt)] = fmincon(fun,x0,[],[],[],[],lb,ub,[],options);
        end
    end

    [~,index] = min(negativeLogLike);
    if day1Indicator
        Xdecode(1,mm) = 1;
        Xdecode(2:end,mm) = x(:,index);
    else
        Xdecode(:,mm) = x(:,index);
    end
    trialCount = trialCount+nBins;
end

end

function [loglikelihood] = MbTDRFull_Likelihood(x,S,W,D,N,M,Q,P,R,binStarts,day1Indicator)

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
for pp=1:P
    if R(pp)>0 
        X{pp} = X{pp}(:,binStarts(1,pp):binStarts(2,pp));
        XSW = XSW+X{pp}*S{pp}*W{pp}';
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

function [loglikelihood] = MbTDRMarginal_Likelihood(x,S,W,CnInv,D,N,M,Q,P,R,binStarts,day1Indicator)

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
