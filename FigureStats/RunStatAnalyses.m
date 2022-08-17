% RunStatAnalyses.m
% Price et al. data analysis
   % make dot plots and compute permutation test p-values for Figures 3, 4, 5, 6 
   %   (Day 1 vs. Days 2,3,4
   %   comparisons in early (51-100ms) and late (101-150ms) windows
   %   after stimulus onset)

N = 140;binSize = 25;

load(sprintf('SeqRFExp_DataForMbTDR-%dNeurons-%dmsBins.mat',N,binSize),'binStartPoint',...
    'expDay','X','Z','neuronTrials','totalStimTime','nBins','binsPerElement');

    % Z is the neural data
    % X is the design matrix, set up for MbTDR
    
    
colors = cell(4,1); % for visualization
colors{1} = [169,209,142]./255;
colors{2} = [244,177,131]./255;
colors{3} = [143,170,220]./255;
colors{4} = [211,139,166]./255;

EInd = 24;
BheldInd = 16;
AngleInd = 4;
angDist = 5*pi/180; % plus or minus 5 degrees from B
targetAngle = 0*pi/180;

neuronTrials = logical(neuronTrials);

% store neural data averaged in different windows, and for different trial
%   types
a = [];
e = [];
aeDiff = [];
BheldDiff = [];
eb = [];
ab = [];

dayPSTH = zeros(nBins,4,2); % store PSTHs for starts with A vs. starts with E

PSTH = cell(N,1); 

binStarts = 1:6:nBins; % start of each element in the sequence
indsEarly = [3,4];indsLate = [5,6]; % early and late windows after onset, in bins

for nn=1:N
    % organize neural data and extract trial info from design matrix
    currNeuralData = Z(nn,neuronTrials(nn,:))';
    Angles = X{AngleInd}(neuronTrials(nn,:),:);
    Estarts = sum(X{EInd}(neuronTrials(nn,:),:),2);
    Bheld = sum(X{BheldInd}(neuronTrials(nn,:),:),2);
    
    trials = length(currNeuralData)/nBins;
    Angles = Angles(7:nBins:end,1);
    currNeuralData = reshape(currNeuralData,[nBins,trials])';%
    Estarts = reshape(Estarts,[nBins,trials])';
    Bheld = reshape(Bheld,[nBins,trials])';
    
    % for figures 3 and 5, A and E
    axcdTrials = find(sum(Estarts,2)==0); % starts with A
    excdTrials = find(sum(Estarts,2)>0); % starts with E
    
    axcd = currNeuralData(axcdTrials,:);
    excd = currNeuralData(excdTrials,:);
    
    meanaxcd = mean(axcd,1);
    meanexcd = mean(excd,1);
    
    PSTH{nn} = [meanaxcd;meanexcd];
    
    for jj=1:4
        if expDay(nn)==jj
            dayPSTH(:,jj,1) = dayPSTH(:,jj,1)+meanaxcd'./sum(expDay==jj);
            dayPSTH(:,jj,2) = dayPSTH(:,jj,2)+meanexcd'./sum(expDay==jj);
        end
    end
    
    currA = [mean(meanaxcd(binStarts(1)+indsEarly(1)-1:binStarts(1)+indsEarly(2)-1)),...
        mean(meanaxcd(binStarts(1)+indsLate(1)-1:binStarts(1)+indsLate(2)-1))];
    currE = [mean(meanexcd(binStarts(1)+indsEarly(1)-1:binStarts(1)+indsEarly(2)-1)),...
        mean(meanexcd(binStarts(1)+indsLate(1)-1:binStarts(1)+indsLate(2)-1))];
    
    a = [a;currA];
    e = [e;currE];
    
    aeDiff = [aeDiff;currE-currA];
    
    % for figure 4, second element held
    axcdTrials = find(sum(Bheld,2)==0); % second and third elements displayed
    axxdTrials = find(sum(Bheld,2)>0); % second element held
    
    axcd = currNeuralData(axcdTrials,:);
    axxd = currNeuralData(axxdTrials,:);
    
    meanaxcd = mean(axcd,1);
    meanaxxd = mean(axxd,1);
    
    BheldDiff = [BheldDiff;mean(meanaxxd(binStarts(3)+indsEarly(1)-1:binStarts(3)+indsEarly(2)-1))-...
        mean(meanaxcd(binStarts(3)+indsEarly(1)-1:binStarts(3)+indsEarly(2)-1)),...
        mean(meanaxxd(binStarts(3)+indsLate(1)-1:binStarts(3)+indsLate(2)-1))-...
        mean(meanaxcd(binStarts(3)+indsLate(1)-1:binStarts(3)+indsLate(2)-1))];
    
    % for figure 6, B after A, and B after E (B meaning angles close to trained B)
    abxdTrials = find(sum(Estarts,2)==0 & abs(Angles-targetAngle)<=angDist);
    ebxdTrials = find(sum(Estarts,2)>0 & abs(Angles-targetAngle)<=angDist);
    
    abxd = currNeuralData(abxdTrials,:);
    ebxd = currNeuralData(ebxdTrials,:);
    
    meanabxd = mean(abxd,1);
    meanebxd = mean(ebxd,1);
    
    ab = [ab;mean(meanabxd(binStarts(2)+indsEarly(1)-1:binStarts(2)+indsEarly(2)-1)),...
        mean(meanabxd(binStarts(2)+indsLate(1)-1:binStarts(2)+indsLate(2)-1))];
    eb = [eb;mean(meanebxd(binStarts(2)+indsEarly(1)-1:binStarts(2)+indsEarly(2)-1)),...
        mean(meanebxd(binStarts(2)+indsLate(1)-1:binStarts(2)+indsLate(2)-1))];
end

%% Figure 3b
allData = cell(4,1);
for dd=1:4
    allData{dd} = [a(expDay==dd,1),a(expDay==dd,2)];
end

results_pvals_3b = zeros(2,3); % rows are A early, A late
                               % columns are days 2, 3, 4

for ii=1:2
    for jj=2:4
        data = [allData{1}(:,ii);allData{jj}(:,ii)]; % stack day 1 data with days 2, 3, or 4
        day1N = sum(expDay==1);
        results_pvals_3b(ii,jj-1) = RunPermutationTest(data,day1N);
    end
end

spots = [1,2,3,4;6,7,8,9];

figure;
for ss=1:2
    for day=1:4
        dat = allData{day}(:,ss);
        scatter(spots(ss,day)+normrnd(0,0.1,[length(dat),1]),dat,75,colors{day},'filled');
        hold on;plot([spots(ss,day)-0.2,spots(ss,day)+0.2],mean(dat)*ones(1,2),'Color',[0,0,0],'LineWidth',3);
    end
end
ylabel('FR (AU)');title('Figure 3b');

%% Figure 4d
allData = cell(4,1);
for dd=1:4
    allData{dd} = [BheldDiff(expDay==dd,1),BheldDiff(expDay==dd,2)];
end

results_pvals_4d = zeros(2,3); % rows are B held minus C early, B held minus C late
                               % columns are days 2, 3, 4

for ii=1:2
    for jj=2:4
        data = [allData{1}(:,ii);allData{jj}(:,ii)]; % stack day 1 data with days 2, 3, or 4
        day1N = sum(expDay==1);
        results_pvals_4d(ii,jj-1) = RunPermutationTest(data,day1N);
    end
end

spots = [1,2,3,4;6,7,8,9];

figure;
for ss=1:2
    for day=1:4
        dat = allData{day}(:,ss);
        scatter(spots(ss,day)+normrnd(0,0.1,[length(dat),1]),dat,75,colors{day},'filled');
        hold on;plot([spots(ss,day)-0.2,spots(ss,day)+0.2],mean(dat)*ones(1,2),'Color',[0,0,0],'LineWidth',3);
    end
end
ylabel('FR Difference (AU)');title('Figure 4d');

%% Figure 5d
allData = cell(4,1);
for dd=1:4
    allData{dd} = [aeDiff(expDay==dd,1),aeDiff(expDay==dd,2)];
end

results_pvals_5d = zeros(2,3); % rows are E minus A early, E minus A late
                               % columns are days 2, 3, 4

for ii=1:2
    for jj=2:4
        data = [allData{1}(:,ii);allData{jj}(:,ii)]; % stack day 1 data with days 2, 3, or 4
        day1N = sum(expDay==1);
        results_pvals_5d(ii,jj-1) = RunPermutationTest(data,day1N);
    end
end

spots = [1,2,3,4;6,7,8,9];

figure;
for ss=1:2
    for day=1:4
        dat = allData{day}(:,ss);
        scatter(spots(ss,day)+normrnd(0,0.1,[length(dat),1]),dat,75,colors{day},'filled');
        hold on;plot([spots(ss,day)-0.2,spots(ss,day)+0.2],mean(dat)*ones(1,2),'Color',[0,0,0],'LineWidth',3);
    end
end
ylabel('FR Difference (AU)');title('Figure 5d');

% %% previous Figure 6b (from older version of the paper)
% allData = cell(4,1);
% for dd=1:4
%     allData{dd} = [ab(expDay==dd,1),eb(expDay==dd,1),ab(expDay==dd,2),eb(expDay==dd,2)];
% end

% results_pvals_6b = zeros(4,3); % rows are AB early, EB early, AB late, EB late
%                                % columns are days 2, 3, 4

% for ii=1:4
%     for jj=2:4
%         data = [allData{1}(:,ii);allData{jj}(:,ii)]; % stack day 1 data with days 2, 3, or 4
%         day1N = sum(expDay==1);
%         results_pvals_6b(ii,jj-1) = RunPermutationTest(data,day1N);
%     end
% end

% spots = [1,2,3,4;11,12,13,14;6,7,8,9;16,17,18,19];

% figure;
% for ss=1:4
%     for day=1:4
%         dat = allData{day}(:,ss);
%         scatter(spots(ss,day)+normrnd(0,0.1,[length(dat),1]),dat,75,colors{day},'filled');
%         hold on;plot([spots(ss,day)-0.2,spots(ss,day)+0.2],mean(dat)*ones(1,2),'Color',[0,0,0],'LineWidth',3);
%     end
% end
% ylabel('FR (AU)');title('Figure 6b');
