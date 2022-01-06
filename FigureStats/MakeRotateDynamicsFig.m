% MakeRotateDynamicsFig.m
%  Price et al. 2022 visualization of low-D latent dynamics
%    (from Figure 3)

colors = cell(4,1);
colors{1} = [169,209,142]./255;
colors{2} = [244,177,131]./255;
colors{3} = [143,170,220]./255;
colors{4} = [211,139,166]./255;

load('SeqRFExp_ModelFitAIC-140Neurons-25msBins.mat');
load('SeqRFExp_DataForMbTDR-140Neurons-25msBins.mat','expDay');

myModel = zeros(140,nBins*4);
% first 30 bins are ABCD, next 30 are ABBD, then EBCD, and EBBD
for ii=1:N
    tmp = B(ii)+S{1}*W{1}(ii,:)'+log(expDay(ii)).*S{2}*W{2}(ii,:)';
    myModel(ii,1:nBins) = tmp;
    tmp2 = tmp+[zeros(12,1);S{16}*W{16}(ii,:)']+...
        log(expDay(ii)).*[zeros(12,1);S{17}*W{17}(ii,:)'];
    myModel(ii,nBins+1:2*nBins) = tmp2;
    tmp3 = tmp+[S{24}*W{24}(ii,:)']+log(expDay(ii)).*[S{25}*W{25}(ii,:)'];
    myModel(ii,2*nBins+1:3*nBins) = tmp3;
    tmp4 = tmp2+tmp3-tmp;
    myModel(ii,3*nBins+1:end) = tmp4;
end


[U1,SS1,V1] = svd(myModel(expDay==1,:)','econ');
[U4,SS4,V4] = svd(myModel(expDay==4,:)','econ');

rank1 = sum(diag(SS1)>1e-5);
rank4 = sum(diag(SS4)>1e-5);

for ii=1:rank1
    if median(V1(:,ii))<=0
        U1(:,ii) = -U1(:,ii);
    end
end
for ii=1:rank4
    if median(V4(:,ii))<=0
        U4(:,ii) = -U4(:,ii);
    end
end

days = [1,4];
mybootstraps = zeros(1000,nBins,3,length(days));
time = 0:binSize*1000:725;
whichBins = 1:nBins;varExp = zeros(1000,3);
for ii=1:length(days)
    data = myModel(expDay==days(ii),:);currN = size(data,1); % run SVD on original data
    [UU,~,VV] = svd(data','econ');
    if median(VV(:,1))<=0
        UU(:,1) = -UU(:,1);
    end
    if median(VV(:,2))<=0
        UU(:,2) = -UU(:,2);
    end
    if median(VV(:,3))<=0
        UU(:,3) = -UU(:,3);
    end
    if ii==1
        UU(:,2) = -UU(:,2);
        UU(:,3) = -UU(:,3);
    end
    meanVal = [UU(whichBins,1),UU(whichBins,2),UU(whichBins,3)];
    for jj=1:1000
        inds = ceil(rand([currN,1])*currN); % resample data, with replacement
        repeatData = data(inds,:);
        [U,TT,~] = svd(repeatData','econ');
        for tt=1:3
            varExp(jj,tt) = sum(TT(tt,tt).^2)./sum(diag(TT).^2);
            if tt==3
                varExp(jj,tt) = sum(TT(tt,tt).^2+TT(2,2).^2)./sum(diag(TT).^2);
            end
        end
        if abs(U(whichBins,3)'*meanVal(:,3))<abs(U(whichBins,4)'*meanVal(:,3))
            U(:,3) = U(:,4);
        end
        if U(whichBins,1)'*meanVal(:,1)<=0
            U(:,1) = -U(:,1);
        end
        if U(whichBins,2)'*meanVal(:,2)<=0
            U(:,2) = -U(:,2);
        end
        if U(whichBins,3)'*meanVal(:,3)<=0
            U(:,3) = -U(:,3);
        end
        mybootstraps(jj,:,1,ii) = U(whichBins,1);
        mybootstraps(jj,:,2,ii) = U(whichBins,2);
        mybootstraps(jj,:,3,ii) = U(whichBins,3);
    end
    conf95low = quantile(squeeze(mybootstraps(:,:,1,ii)),0.05/2,1)';
    conf95high = quantile(squeeze(mybootstraps(:,:,1,ii)),1-0.05/2,1)';

    figure;boundedline(time,meanVal(:,1),[meanVal(:,1)-conf95low,conf95high-meanVal(:,1)],'alpha',...
        'cmap',colors{days(ii)},'transparency',0.5);
    axis([0 725 -0.5 0.4]);xticks([0,150,300,450,600]);
    xlabel('Time from Stimulus Onset (ms)');
    ylabel('FR Modulation');
    
    conf95low = quantile(squeeze(mybootstraps(:,:,2,ii)),0.05/2,1)';
    conf95high = quantile(squeeze(mybootstraps(:,:,2,ii)),1-0.05/2,1)';
%     meanVal(:,2) = mean(squeeze(mybootstraps(:,:,2,ii)),1);%U(nBins+1:2*nBins,2);
    
    hold on;boundedline(time,meanVal(:,2),[meanVal(:,2)-conf95low,conf95high-meanVal(:,2)],'alpha',...
        'cmap',colors{days(ii)}*0.7,'transparency',0.5);
    
    conf95low = quantile(squeeze(mybootstraps(:,:,3,ii)),0.05/2,1)';
    conf95high = quantile(squeeze(mybootstraps(:,:,3,ii)),1-0.05/2,1)';
%     meanVal(:,2) = mean(squeeze(mybootstraps(:,:,2,ii)),1);%U(nBins+1:2*nBins,2);
    
    hold on;boundedline(time,meanVal(:,3),[meanVal(:,3)-conf95low,conf95high-meanVal(:,3)],'alpha',...
        'cmap',colors{days(ii)}*0.4,'transparency',0.5);
    axis([0 725 -0.5 0.4]);xticks([0,150,300,450,600]);
    xlabel('Time from Stimulus Onset (ms)');
    ylabel('FR Modulation');
    title(sprintf('Day %d',days(ii)));
end

T = 750;
myBasis = zeros(T,nBins);
for ii=1:nBins
x = 0:1:(T-1);
centers = (binSize*1000/2):binSize*1000:T;
myBasis(:,ii) = exp(-(x-centers(ii)).^2./(2*15*15));
end

mymap4 = zeros(T,3);
mymap1 = zeros(T,3);
for ii=1:3
    mymap4(:,ii) = linspace(colors{4}(ii)*0.2,colors{4}(ii)*1.2,T);
    mymap1(:,ii) = linspace(colors{1}(ii)*0.2,colors{1}(ii)*1.2,T);
end
mymap1 = flipud(mymap1);
mymap4 = flipud(mymap4);
U1(:,3) = -U1(:,3);

figure;scatter3(myBasis*U1(whichBins,1),myBasis*U1(whichBins,2),myBasis*U1(whichBins,3),[],mymap1,'filled');
% axis([-0.3 0.35 -0.4 0.35]);
c = colorbar;
c.Ticks = [0,0.2,0.4,0.6,0.8];c.TickLabels = [0,150,300,450,600];
c.Label.String = 'Time from Stimulus Onset (ms)';
xlabel('PC 1');ylabel('PC 2');zlabel('PC 3');
colormap(mymap1);
title('Day 1');

figure;scatter3(myBasis*U4(whichBins,1),myBasis*U4(whichBins,2),myBasis*U4(whichBins,3),[],mymap4,'filled');
% axis([-0.3 0.35 -0.4 0.35]);
c = colorbar;
c.Ticks = [0,0.2,0.4,0.6,0.8];c.TickLabels = [0,150,300,450,600];
c.Label.String = 'Time from Stimulus Onset (ms)';
xlabel('PC 1');ylabel('PC 2');zlabel('PC 3');
colormap(mymap4);
title('Day 4');
