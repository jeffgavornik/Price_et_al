% DecodeWrapper.m

N = 140;binSize = 25;

load(sprintf('SeqRFExp_ModelFitAIC-%dNeurons-%dmsBins.mat',N,binSize),'B','D',...
    'W','S','R','P','stimDim','CnInv');

load(sprintf('SeqRFExp_DataForMbTDRDecoder-%dNeurons-%dmsBins.mat',N,binSize),...
    'Xdecode','Zdecode','minimalX','nBins','decodeTrials','binStartPoint','abcdOrients','binsPerElement');

load(sprintf('SeqRFExp_DataForMbTDR-%dNeurons-%dmsBins.mat',N,binSize),...
    'evokedStd','spontaneousBase','expDay');    

%% get explained variance on held-out decoder set
  % proportion of explained variance in training set (estimate of
  % explainable variance by the PSTH when seeing the same stimulus
  % repeatedly) 
decodeTrials = logical(decodeTrials);
tmp0 = Zdecode(decodeTrials);

tmpData = [];tmpModel = [];FR = zeros(N,1);MI = zeros(N,1);
expVar = zeros(N,1);day = zeros(N,1);
for nn=1:N
    tmpData = [tmpData;Zdecode(nn,decodeTrials(nn,:))'];
    tmp2 = Zdecode(nn,decodeTrials(nn,:))'-B(nn);var0 = var(tmp2);
    day(nn) = round(exp(Xdecode{2}(find(decodeTrials(nn,:),1,'first'),1)));
    for pp=1:P
        if R(pp)>0
            tmp2 = tmp2-Xdecode{pp}(decodeTrials(nn,:),:)*S{pp}*W{pp}(nn,:)';
        end
    end
    tmpModel = [tmpModel;Zdecode(nn,decodeTrials(nn,:))'-tmp2];expVar(nn) = 1-var(tmp2)/var0;
    
    tmp = Zdecode(nn,decodeTrials(nn,:))'*evokedStd(nn)+spontaneousBase(nn);
    FR(nn) = mean(tmp);
    
    MI(nn) = max(0.5*log2(var0)-0.5*log2(var(tmp2)),0);
    
    tmpD = reshape(Zdecode(nn,decodeTrials(nn,:))',[nBins,60]);
    tmpM = reshape(Zdecode(nn,decodeTrials(nn,:))'-tmp2,[nBins,60]);
    times = linspace(0,750,nBins);
%     if ismember(nn,inds) %expVar(nn)<=0 || expVar(nn)>=0.1
%         figure;plot(times,mean(tmpD,2));hold on;plot(times,mean(tmpM,2));
%         legend('Data','Model');
%         title(sprintf('Unit %d : Exp Var %3.2f',nn,expVar(nn)));
%     end
end
FR = FR./(binSize/1000);
explainedVariance = 1-var(tmpModel-tmpData)/var(tmpData);

figure;plot(day,expVar,'.','MarkerSize',20);xlabel('Experimental Day');ylabel('Var Exp');
title('Explained Variance Across Days');

fprintf('Model Explained Variance: %3.2f %%\n',explainedVariance.*100);
colors = cell(4,1);
colors{1} = [169,209,142]./255;
colors{2} = [244,177,131]./255;
colors{3} = [143,170,220]./255;
colors{4} = [211,139,166]./255;
figure;
for ii=1:4
    inds = find(day==ii);
    [f,x] = ecdf(expVar(inds));
    plot(x,f,'LineWidth',5,'Color',colors{ii});hold on;
    hold on;
end

plot(0.067*ones(10,1),linspace(0,1,10),'k-','LineWidth',5); 
plot(0.0556*ones(10,1),linspace(0,1,10),'k--','LineWidth',4);
plot(0.0803*ones(10,1),linspace(0,1,10),'k--','LineWidth',4);
axis([-0.025 0.25 0 1]);xlabel('Held-Out Proportion of Variance Explained');
ylabel('CDF');title('Explained Variance Distribution');

% mutual information across days
MIperDay = zeros(4,1);
figure;
for ii=1:4
    tot = MI(day==ii);
    scatter(ii+normrnd(0,0.05,[length(tot),1]),tot,75,colors{ii},'filled'); hold on;
    MIperDay(ii) = mean(tot);
    meanVal = mean(tot);
    plot(linspace(ii-0.2,ii+0.2,10),meanVal*ones(10,1),'Color',colors{ii},'LineWidth',5);
end
xlabel('Exp Day');
ylabel('Mutual Information (bits)');
title('Stimulus/Neural Mutual Information');

MIspikePerDay = zeros(4,1);
MI2 = MI./FR;
figure;
for ii=1:4
    tot = MI2(day==ii);
    scatter(ii+normrnd(0,0.05,[length(tot),1]),tot,75,colors{ii},'filled'); hold on;
    MIspikePerDay(ii) = mean(tot);
    meanVal = mean(tot);
    plot(linspace(ii-0.2,ii+0.2,10),meanVal*ones(10,1),'Color',colors{ii},'LineWidth',5);
end
xlabel('Exp Day');
ylabel('Mutual Information (bits/spike)');
title('Stimulus/Neural Mutual Information');

% how can I get MI specifically for ABCD and EBCD?
figure;
MI_ae = zeros(N,2);
for nn=1:N
    aPSTH = B(nn)+S{1}*W{1}(nn,:)';
    ePSTH = aPSTH+[S{24}*W{24}(nn,:)';zeros(nBins-length(S{24}),1)];
    aPSTH = aPSTH*evokedStd(nn)+spontaneousBase(nn); % times evokedStd(nn)
    ePSTH = ePSTH*evokedStd(nn)+spontaneousBase(nn);
    infoa = InfoEstimate(aPSTH,spontaneousBase(nn),false);
    infoe = InfoEstimate(ePSTH,spontaneousBase(nn),false);
    scatter(day(nn)+normrnd(0,0.05),infoa-infoe,75,colors{day(nn)},'filled'); hold on;
    MI_ae(nn,1) = infoa;
    MI_ae(nn,2) = infoe;
end

for ii=1:4
    meanVal = mean(MI_ae(day==ii,1)-MI_ae(day==ii,2));
    plot(linspace(ii-0.2,ii+0.2,10),meanVal*ones(10,1),'Color',colors{ii},'LineWidth',5);
end
xlabel('Exp Day');
ylabel('Relative Information');
title('A vs. E Information');
%% run decoder

[XD,neuronsPerTrial] = RunDecoder(Zdecode,B,S,W,D,R,decodeTrials,nBins,binStartPoint);

%% check decoder accuracy

% there are several variables that are more fundamental than the ones
%   used in the model fit, and we want to recover those
%  1) experimental day
%  2) stim B angle
%  3) stim B held on
%  4) starts with E

Mdecode = max(size(XD));pDecode = min(size(XD));

decodedX = XD';

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

figure;
for ii=1:4
    scatter(trueX(trueX(:,1)==ii,2).*180/pi,decodedX(trueX(:,1)==ii,2).*180/pi,...
        [],colors{ii},'filled');
    hold on;
end
xlabel('True Angle (degs)');ylabel('Decoded Angle (degs)');title('Stimulus 2 Angle');

figure;
Violin(decodedX(trueX(:,1)==1,1),1,'ViolinColor',colors{1});
hold on;Violin(decodedX(trueX(:,1)==2,1),2,'ViolinColor',colors{2});
hold on;Violin(decodedX(trueX(:,1)==3,1),3,'ViolinColor',colors{3});
hold on;Violin(decodedX(trueX(:,1)==4,1),4,'ViolinColor',colors{4});
xlabel('Experimental Day');ylabel('Decoded Day');
title('Exp Day');
figure;
orientDiff = 10;
edges = 60:orientDiff:180;binCens = (edges(1:end-1)+edges(2:end))/2;
[~,~,bin] = histcounts(trueX(:,2).*180/pi,edges);
for ii=1:length(binCens)
    Violin(decodedX(bin==ii,2).*180/pi,binCens(ii));hold on;
end
xlabel('True Angle (degs)');ylabel('Decoded Angle');
title('Stimulus 2 Angle');

figure;
for ii=1:4
    scatter(trueX(trueX(:,1)==ii,5),decodedX(trueX(:,1)==ii,5),...
        [],colors{ii},'filled');
    hold on;
end
xlabel('True Trial Count');ylabel('Decoded Count');
title('Trial Count');

%% decoder-ground truth correlations 

rho = zeros(5,1);pval = zeros(5,1);

for ii=1:5
   [rho(ii),pval(ii)] = corr(trueX(:,ii),decodedX(:,ii),'Type','Spearman'); 
   fprintf('Rho-%d: %3.4f\n',ii,rho(ii));
   fprintf('P-%d: %3.4e\n',ii,pval(ii));
end

%% confusion matrices for decoding stimulus ID

% decode starts with A versus starts with E
confusionMatrix = zeros(2,2,4);

for mm=1:Mdecode
    day = round(trueX(mm,1));
    if trueX(mm,4)==0
        trueStim = 1;
    elseif trueX(mm,4)==1
        trueStim = 2;
    end
    
    if decodedX(mm,4)<=0.5
        decodedStim = 1;
    elseif decodedX(mm,4)>0.5
        decodedStim = 2;
    end
    
    confusionMatrix(trueStim,decodedStim,day) = ...
        confusionMatrix(trueStim,decodedStim,day)+1;
end

for ii=1:2
    for jj=1:4
        confusionMatrix(ii,:,jj) = confusionMatrix(ii,:,jj)./sum(squeeze(confusionMatrix(ii,:,jj)));
    end
end

figure;
for jj=1:4
   subplot(2,2,jj);
   imagesc(squeeze(confusionMatrix(:,:,jj)));caxis([0 1]);colormap('viridis');
   colorbar;
   xlabel('Decoded Stim');ylabel('True Stim');title(sprintf('Day %d',jj));
   xticks([1,2]);yticks([1,2]);
   xticklabels({'A_','E_'});
   yticklabels({'A_','E_'});
   xtickangle(45);
end

% decode B held versus not held
confusionMatrix = zeros(2,2,4);

for mm=1:Mdecode
    day = round(trueX(mm,1));
    if trueX(mm,3)==0
        trueStim = 1;
    elseif trueX(mm,3)==1
        trueStim = 2;
    end
    
    if decodedX(mm,3)<=0.5
        decodedStim = 1;
    elseif decodedX(mm,3)>0.5
        decodedStim = 2;
    end
    
    confusionMatrix(trueStim,decodedStim,day) = ...
        confusionMatrix(trueStim,decodedStim,day)+1;
end

for ii=1:2
    for jj=1:4
        confusionMatrix(ii,:,jj) = confusionMatrix(ii,:,jj)./sum(squeeze(confusionMatrix(ii,:,jj)));
    end
end

figure;
for jj=1:4
   subplot(2,2,jj);
   imagesc(squeeze(confusionMatrix(:,:,jj)));caxis([0 1]);colormap('viridis');
   colorbar;
   xlabel('Decoded Stim');ylabel('True Stim');title(sprintf('Day %d',jj));
   xticks([1,2]);yticks([1,2]);
   xticklabels({'XCD','XXD'});
   yticklabels({'XCD','XXD'});
   xtickangle(45);
end


% all four stimulus types
confusionMatrix = zeros(4,4,4);
trueStimulus = zeros(Mdecode,1);expDay = zeros(Mdecode,1);
decodedStimulus = zeros(Mdecode,1);
for mm=1:Mdecode
   day = round(trueX(mm,1));expDay(mm) = day;
   if trueX(mm,4)==0 && trueX(mm,3)==0
       trueStim = 1;
   elseif trueX(mm,4)==0 && trueX(mm,3)==1
       trueStim = 2;
   elseif trueX(mm,4)==1 && trueX(mm,3)==0
       trueStim = 3;
   elseif trueX(mm,4)==1 && trueX(mm,3)==1
       trueStim = 4;
   end
   trueStimulus(mm) = trueStim;
   
   if decodedX(mm,4)<=0.5 && decodedX(mm,3)<=0.5
       decodedStim = 1;
   elseif decodedX(mm,4)<=0.5 && decodedX(mm,3)>0.5
       decodedStim = 2;
   elseif decodedX(mm,4)>0.5 && decodedX(mm,3)<=0.5
       decodedStim = 3;
   elseif decodedX(mm,4)>0.5 && decodedX(mm,3)>0.5
       decodedStim = 4;
   end
   decodedStimulus(mm) = decodedStim;
   
   confusionMatrix(trueStim,decodedStim,day) = confusionMatrix(trueStim,decodedStim,day)+1;
end

for ii=1:4
    for jj=1:4
        confusionMatrix(ii,:,jj) = confusionMatrix(ii,:,jj)./sum(squeeze(confusionMatrix(ii,:,jj)));
    end
end

figure;
for jj=1:4
   subplot(2,2,jj);
   imagesc(squeeze(confusionMatrix(:,:,jj)));caxis([0 1]);colormap('viridis');
   colorbar;
   xlabel('Decoded Stim');ylabel('True Stim');title(sprintf('Day %d',jj));
   xticks([1,2,3,4]);yticks([1,2,3,4]);
   xticklabels({'AXCD','AXXD','EXCD','EXXD'});
   yticklabels({'AXCD','AXXD','EXCD','EXXD'});
   xtickangle(45);
end

newconfusion = mean(confusionMatrix,3);
figure;imagesc(newconfusion);caxis([0 1]);colormap('magma');
colorbar;
xlabel('Decoded Stim');ylabel('True Stim');
xticks([1,2,3,4]);yticks([1,2,3,4]);
xticklabels({'AxCD','Ax\rightarrowD','ExCD','Ex\rightarrowD'});
yticklabels({'AxCD','Ax\rightarrowD','ExCD','Ex\rightarrowD'});
xtickangle(45);
