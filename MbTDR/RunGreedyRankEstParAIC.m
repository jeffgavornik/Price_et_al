% RunGreedyRankEstParAIC.m

%  greedy algorithm to estimate the rank of the model ... each predictor
%   can have a rank anywhere from 0 to the total number of neurons used for
%   that predictor (max of 140 in this case)

N = 140;binSize = 25;
filename = sprintf('SeqRFExp_DataForMbTDR-%dNeurons-%dmsBins.mat',N,binSize);
load(filename,'Z','X','N','M','trialDivision','trialCodes',...
    'nBins','binSize','totalStimTime','elementLen','binsPerElement','totDays',...
    'nPreds','binStartPoint','neuronTrials','stimDim','minimalX','abcdOrients');

P = nPreds;

% divide up data into training, testing, and decoding
%   model is fit with maximum likelihood on training data, model is chosen based
%    on maximizing the likelihood of the held-out testing set, model is
%    then used to decode stimulus ID on held-out decoding set

% there is some trickery here with the indexing ... the model is fit in the
%  wrong way in some sense [because the data has been formatted from
%  neuron-by-trials-by-(time per trial) to neurons-by-(trials X time per trial)] 
%  ... to break it up properly into training and testing and decoding, however, 
%  we must preserve the original trial structure

neuronTrials = logical(neuronTrials);

Xtrain = cell(P,1);
Xdecode = cell(P,1);

origTrials = M/nBins;
  
trainP = 0.9;
decodeP = 1-trainP;

Mtrain = round(origTrials*trainP);
Mdecode = round(origTrials*decodeP);

Ztrain = zeros(N,Mtrain*nBins);
trainTrials = zeros(N,Mtrain*nBins);
Zdecode = zeros(N,Mdecode*nBins);
decodeTrials = zeros(N,Mdecode*nBins);

for pp=1:P
    Xtrain{pp} = zeros(Mtrain*nBins,stimDim{pp}(1));
    Xdecode{pp} = zeros(Mdecode*nBins,stimDim{pp}(1));
end

% set random number generator with specific seed for replicability
rng(2134812443);

trainCounter = 1;
decodeCounter = 1;
for ii=1:length(trialCodes)
   whichTrials = find(trialDivision==trialCodes(ii));
   
   nOrigTrials = length(whichTrials)/nBins;
   nTrain = round(nOrigTrials*trainP);
   nDecode = nOrigTrials-nTrain;
  
   inds = randperm(nOrigTrials,nOrigTrials);
   origTrainInds = inds(1:nTrain);
   origDecodeInds = inds(nTrain+1:end);
   
   comparison = zeros(nOrigTrials,nBins);
   count = 1;
   for nn=1:nOrigTrials
      comparison(nn,:) = count:count+nBins-1;
      count = count+nBins;
   end
   
   Ztrain(:,trainCounter:trainCounter+nTrain*nBins-1) = ...
       Z(:,whichTrials(comparison(origTrainInds,:)'));
   Zdecode(:,decodeCounter:decodeCounter+nDecode*nBins-1) = ...
       Z(:,whichTrials(comparison(origDecodeInds,:)'));
   
   trainTrials(:,trainCounter:trainCounter+nTrain*nBins-1) = ...
       neuronTrials(:,whichTrials(comparison(origTrainInds,:)'));
   decodeTrials(:,decodeCounter:decodeCounter+nDecode*nBins-1) = ...
       neuronTrials(:,whichTrials(comparison(origDecodeInds,:)'));
       
   for pp=1:P
       Xtrain{pp}(trainCounter:trainCounter+nTrain*nBins-1,:) = ...
           X{pp}(whichTrials(comparison(origTrainInds,:)'),:);
       Xdecode{pp}(decodeCounter:decodeCounter+nDecode*nBins-1,:) = ...
           X{pp}(whichTrials(comparison(origDecodeInds,:)'),:);
   end
   
   trainCounter = trainCounter+nTrain*nBins;
   decodeCounter = decodeCounter+nDecode*nBins;
end

Mdecode = Mdecode*nBins;

fullX = X{1};

for pp=2:P
   fullX = [fullX,X{pp}]; 
end

fullXCov = cov(fullX);
fullXCov(abs(fullXCov)<1e-6) = 0;

save(sprintf('SeqRFExp_DataForMbTDRDecoder-%dNeurons-%dmsBins.mat',N,binSize*1000),...
    'Zdecode','Xdecode','Mdecode','nBins',...
    'trialDivision','trialCodes','P','totDays','N','decodeTrials','decodeP',...
    'elementLen','binSize','binsPerElement','binStartPoint','minimalX','fullXCov','abcdOrients');

clear Zdecode Xdecode Mdecode trialDivision trialCodes totDays decodeTrials ...
    trainCounter decodeCounter nPreds origTestInds origTrainInds ...
    origDecodeInds inds ii pp nn nOrigTrials count comparison origTrials whichTrials ...
    nTrain nDecode X fullX fullXCov abcdOrients;

NR = zeros(P,1);
for pp=1:P
    tmp = Xtrain{pp};
    tmp = sum(abs(tmp),2);
    matrix = repmat(tmp'>0,[N,1]);
    check = trainTrials.*matrix;
    NR(pp) = length(find(sum(check,2)>0));
end

Mtrain = Mtrain*nBins;

R = zeros(1,P);R(1) = 1;

[W,S,B,D,numParams,CnInv] = MbTDR_ECM(Ztrain,Xtrain,stimDim,R,trainTrials,NR,S);
[testLikelihood] = MbTDRMarginal_Likelihood(S,B,D,N,Xtrain,Ztrain,P,R,trainTrials);

AIC = 2*numParams-2*testLikelihood;
aicDiff = Inf;
tolerance = 1e-6;

parpool(32);
iter = 1;
while aicDiff>tolerance
    allAICs = zeros(P,1);
    allW = cell(P,1);
    allS = cell(P,1);
    allB = cell(P,1);
    allD = cell(P,1);
    allR = cell(P,1);
    parfor ii=1:P
        tmpR = R;tmpR(ii) = tmpR(ii)+1;
        
        try
            [tmpW,tmpS,tmpB,tmpD,numParams] = MbTDR_ECM(Ztrain,Xtrain,stimDim,tmpR,trainTrials,NR,S);
            [compareLikelihood] = MbTDRMarginal_Likelihood(tmpS,tmpB,tmpD,N,Xtrain,Ztrain,P,tmpR,trainTrials);
            %             [tmpW,tmpS,tmpB,tmpD,numParams] = MbTDR_ECM(Ztrain,Xtrain,stimDim,tmpR,trainTrials,NR,S);
            %             [compareLikelihood2] = MbTDRMarginal_Likelihood(tmpS,tmpB,tmpD,N,Xtrain,Ztrain,P,tmpR,trainTrials);
            %
            %             compareLikelihood = max(compareLikelihood,compareLikelihood2);
            allW{ii} = tmpW;
            allS{ii} = tmpS;
            allB{ii} = tmpB;
            allD{ii} = tmpD;
            allR{ii} = tmpR;
            
        catch
            compareLikelihood = -Inf;
            numParams = 0;
        end
        allAICs(ii) = 2*numParams-2*compareLikelihood;
    end
    
    [currentAIC,index] = min(allAICs);

    aicDiff = AIC-currentAIC;
    if aicDiff>tolerance
        W = allW{index};
        S = allS{index};
        B = allB{index};
        D = allD{index};
        R = allR{index};
        AIC = currentAIC;
    end
    iter = iter+1;
    fprintf('\nAIC Improvement: %3.2f\n',aicDiff);
    fprintf('Current Iter: %d\n\n',iter);
end

delete(gcp('nocreate'))

% [W,S,B,D,numParams,CnInv] = MbTDR_ECM(Ztrain,Xtrain,stimDim,R,trainTrials,NR);
% [Likelihood] = MbTDRMarginal_Likelihood(S,B,D,N,Xtrain,Ztrain,P,R,trainTrials);
% 
% AIC2 = 2*numParams-Likelihood;
% 
% fprintf('AIC1: %3.2f ... AIC2: %3.2f\n',AIC,AIC2);
% AIC = AIC2;
save(fullfile(sprintf('SeqRFExp_ModelFitAIC-%dNeurons-%dmsBins.mat',N,binSize*1000)),...
    'W','S','B','D','R','P','AIC','stimDim','nBins','binSize','N',...
    'M','Mtrain','elementLen','trainP','totalStimTime',...
    'neuronTrials','binStartPoint','minimalX','CnInv');
