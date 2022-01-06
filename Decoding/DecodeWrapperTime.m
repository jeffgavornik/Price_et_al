% DecodeWrapperTime.m

N = 140;binSize = 25;

load(sprintf('SeqRFExp_ModelFitAIC-%dNeurons-%dmsBins.mat',N,binSize),'B','D',...
    'W','S','R','P','stimDim');


load(sprintf('SeqRFExp_DataForMbTDRDecoder-%dNeurons-%dmsBins.mat',N,binSize),...
    'Xdecode','Zdecode','minimalX','nBins','decodeTrials','binStartPoint','abcdOrients','binsPerElement');


load(sprintf('SeqRFExp_DataForMbTDR-%dNeurons-%dmsBins.mat',N,binSize),...
    'evokedStd','spontaneousBase');   

decodeTrials = logical(decodeTrials);

[timeDecode,timeDecodeCond,neuronsPerTrial,elemDecodeCond] = RunDecoderTime(Zdecode,...
    Xdecode,B,S,W,D,R,decodeTrials,nBins,binStartPoint);

Mdecode = max(size(Xdecode{1}))/nBins;

trueX = zeros(Mdecode,5);

origCount = 0;
for mm=1:Mdecode
   trueX(mm,1) = round(exp(Xdecode{2}(origCount+nBins,end)));
   trueX(mm,2) = abcdOrients(2)-Xdecode{4}(origCount+binsPerElement+1,1); % add B orientation back in
   trueX(mm,3) = Xdecode{16}(origCount+2*binsPerElement+1,1); % B held
   trueX(mm,4) = Xdecode{24}(origCount+1,1); % E stim
   trueX(mm,5) = exp(-Xdecode{29}(origCount+nBins,end)); % trial
   origCount = origCount+nBins;
end

trueX(:,1) = round(trueX(:,1));

confusionMatrix = zeros(nBins,nBins,4);

for mm=1:Mdecode
    day = trueX(mm,1);
    for tt=1:nBins
        trueTime = tt;
        decodedTime = timeDecode(tt,mm);
        
        confusionMatrix(trueTime,decodedTime,day) = confusionMatrix(trueTime,decodedTime,day)+1;
    end
end

for ii=1:nBins
    for jj=1:4
        confusionMatrix(ii,:,jj) = confusionMatrix(ii,:,jj)./sum(squeeze(confusionMatrix(ii,:,jj)));
    end
end

time = 25/2:25:750;
figure;
for jj=1:4
   subplot(2,2,jj);
   imagesc(time,time,squeeze(confusionMatrix(:,:,jj)));caxis([0 1]);colormap('viridis');
   colorbar;
   xlabel('Decoded Time');ylabel('True Time');title(sprintf('Day %d',jj));
   xticks([0,150,300,450,600]);yticks([0,150,300,450,600]);
   xtickangle(45);
end

timeTrue = repmat((1:30)',[1,Mdecode]);

for jj=1:4
    inds = find(trueX(:,1)==jj);
    tmpTrue = timeTrue(:,inds);tmpDecode = timeDecode(:,inds);
    accuracy = sum(abs(tmpDecode(:)-tmpTrue(:))<=1)./length(tmpTrue(:));

    fprintf('Day %d - Time Decode Accuracy: %3.3f\n',jj,accuracy);
end

% conditional time decoding
confusionMatrix = zeros(nBins,nBins,4);

for mm=1:Mdecode
    day = trueX(mm,1);
    for tt=1:nBins
        trueTime = tt;
        decodedTime = timeDecodeCond(tt,mm);
        
        confusionMatrix(trueTime,decodedTime,day) = confusionMatrix(trueTime,decodedTime,day)+1;
    end
end

for ii=1:nBins
    for jj=1:4
        confusionMatrix(ii,:,jj) = confusionMatrix(ii,:,jj)./sum(squeeze(confusionMatrix(ii,:,jj)));
    end
end

time = 25/2:25:750;
figure;
for jj=1:4
   subplot(2,2,jj);
   imagesc(time,time,squeeze(confusionMatrix(:,:,jj)));caxis([0 1]);colormap('viridis');
   colorbar;
   xlabel('Decoded Time');ylabel('True Time');title(sprintf('Day %d',jj));
   xticks([0,150,300,450,600]);yticks([0,150,300,450,600]);
   xtickangle(45);
end

timeTrue = repmat((1:30)',[1,Mdecode]);

for jj=1:4
    inds = find(trueX(:,1)==jj);
    tmpTrue = timeTrue(:,inds);tmpDecode = timeDecodeCond(:,inds);
    accuracy = sum(abs(tmpDecode(:)-tmpTrue(:))<=1)./length(tmpTrue(:));

    fprintf('Day %d - Conditional Time Decode Accuracy: %3.3f\n',jj,accuracy);
end

% conditional time decoding, element timing
binsPerElem = 6;
confusionMatrix = zeros(binsPerElem,binsPerElem,4);

for mm=1:Mdecode
    day = trueX(mm,1);
    for tt=1:nBins
        trueTime = mod(tt-1,binsPerElem)+1;
        decodedTime = elemDecodeCond(tt,mm);
        
        confusionMatrix(trueTime,decodedTime,day) = confusionMatrix(trueTime,decodedTime,day)+1;
    end
end

for ii=1:binsPerElem
    for jj=1:4
        confusionMatrix(ii,:,jj) = confusionMatrix(ii,:,jj)./sum(squeeze(confusionMatrix(ii,:,jj)));
    end
end

time = 25/2:25:150;
figure;
for jj=1:4
   subplot(2,2,jj);
   imagesc(time,time,squeeze(confusionMatrix(:,:,jj)));caxis([0 1]);colormap('viridis');
   colorbar;
   xlabel('Decoded Time');ylabel('True Time');title(sprintf('Day %d',jj));
   xticks([0,50,100,150]);yticks([0,50,100,150]);
   xtickangle(45);
end

time = (1:nBins)';
timeTrue = repmat(mod(time-1,binsPerElem)+1,[1,Mdecode]);

for jj=1:4
    inds = find(trueX(:,1)==jj);
    tmpTrue = timeTrue(:,inds);tmpDecode = elemDecodeCond(:,inds);
    accuracy = sum(tmpDecode(:)==tmpTrue(:))./length(tmpTrue(:));

    fprintf('Day %d - Conditional Element Time Decode Accuracy: %3.3f\n',jj,accuracy);
end

% time decode (marginalizing X)
timeTrue = repmat((1:30)',[1,Mdecode]);
accuracy = zeros(4,1);
for jj=1:4
    inds = find(trueX(:,1)==jj);
    tmpTrue = timeTrue(:,inds);tmpDecode = timeDecode(:,inds);
    accuracy(jj) = sum(abs(tmpDecode(:)-tmpTrue(:))<=1)./length(tmpTrue(:));
end

bootstrapAcc = zeros(4,5000);
figure;
for jj=1:4
    inds = find(trueX(:,1)==jj);
    tmpTrue = timeTrue(:,inds);tmpDecode = timeDecode(:,inds);
    [nB,nT] = size(tmpDecode);
    for ii=1:5000
        newDecode = zeros(size(tmpDecode));
        for tt=1:nB
            inds = ceil(rand([1,nT])*nT);
            newDecode(tt,:) = tmpDecode(tt,inds);
        end
        bootstrapAcc(jj,ii) = sum(abs(newDecode(:)-tmpTrue(:))<=1)./length(tmpTrue(:));
    end
    histogram(bootstrapAcc(jj,:));hold on;
end

figure;
for jj=1:4
    confInt = quantile(bootstrapAcc(jj,:),[0.05/2,1-0.05/2]);
    errorbar(jj,accuracy(jj),accuracy(jj)-confInt(1),confInt(2)-accuracy(jj),'d',...
        'LineWidth',5,'CapSize',12);
    hold on;
end
plot(linspace(0,5,10),0.078*ones(10,1),'--k','LineWidth',2);
plot(linspace(0,5,10),0.118*ones(10,1),'--k','LineWidth',2);
axis([0 5 0 0.35]);
box('off');

% time decode, conditional on true value of X
timeTrue = repmat((1:30)',[1,Mdecode]);
accuracy = zeros(4,1);
for jj=1:4
    inds = find(trueX(:,1)==jj);
    tmpTrue = timeTrue(:,inds);tmpDecode = timeDecodeCond(:,inds);
    accuracy(jj) = sum(abs(tmpDecode(:)-tmpTrue(:))<=1)./length(tmpTrue(:));
end

bootstrapAcc = zeros(4,5000);
figure;
for jj=1:4
    inds = find(trueX(:,1)==jj);
    tmpTrue = timeTrue(:,inds);tmpDecode = timeDecodeCond(:,inds);
    [nB,nT] = size(tmpDecode);
    for ii=1:5000
        newDecode = zeros(size(tmpDecode));
        for tt=1:nB
            inds = ceil(rand([1,nT])*nT);
            newDecode(tt,:) = tmpDecode(tt,inds);
        end
        bootstrapAcc(jj,ii) = sum(abs(newDecode(:)-tmpTrue(:))<=1)./length(tmpTrue(:));
    end
    histogram(bootstrapAcc(jj,:));hold on;
end

figure;
for jj=1:4
    confInt = quantile(bootstrapAcc(jj,:),[0.05/2,1-0.05/2]);
    errorbar(jj,accuracy(jj),accuracy(jj)-confInt(1),confInt(2)-accuracy(jj),'d',...
        'LineWidth',5,'CapSize',12);
    hold on;
end
plot(linspace(0,5,10),0.078*ones(10,1),'--k','LineWidth',2);
plot(linspace(0,5,10),0.118*ones(10,1),'--k','LineWidth',2);
axis([0 5 0 0.4]);
box('off');
