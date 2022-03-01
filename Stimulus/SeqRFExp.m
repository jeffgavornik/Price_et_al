function [] = SeqRFExp(AnimalName,Day,holdTime)
%SeqRFExp.m
%  Display static full-field sinusoidal gratings for 150ms
%   run a randomized 4-day experiment attempting to look for changes in
%   population orientation tuning and evidence of predictive coding

% could also try with the Berry Patch stimulus

% INPUT: Obligatory-
%        AnimalName - animal's unique identifier as a number, e.g. 45602
%        Day - experimental day
%
%        Optional- 
%        holdTime - amount of time to wait between blocks of about 50 stimuli
% 
%
% OUTPUT: a file with stimulus parameters named SeqStimDate_AnimalName
%           e.g. SeqStim20160708_12345.mat to be saved in CloudStation's 
%           Seq folder under '~/CloudStation/ByronExp/SEQ'
% Created: 2018/03/03 at 24 Cummington, Boston
%  Byron Price
% Updated: 2020/07/06
%  By: Byron Price

load('SequenceVars.mat');
degreeRadius = 100;
radianRadius = degreeRadius*pi/180;

directory = '~/Documents/MATLAB/Sequence-RF-Exp';
%directory = '~/CloudStation/ByronExp/SEQ';

if nargin < 3
    holdTime = 60;
end

Date = datetime('today','Format','yyyy-MM-dd');
Date = char(Date); Date = strrep(Date,'-','');Date = str2double(Date);
% Acquire a handle to OpenGL, so we can use OpenGL commands in our code:
global GL;

% Make sure this is running on OpenGL Psychtoolbox:
AssertOpenGL;

usb = usb1208FSPlusClass;
display(usb);

WaitSecs(1);

% Choose screen with maximum id - the secondary display:
screenid = max(Screen('Screens'));

% 
% % Open a fullscreen onscreen window on that display, choose a background
% % color of 127 = gray with 50% max intensity; 0 = black;255 = white
background = 127;
[win,~] = Screen('OpenWindow', screenid,background);

gammaTable = makeGrayscaleGammaTable(gama,0,255);
Screen('LoadNormalizedGammaTable',win,gammaTable);

% Switch color specification to use the 0.0 - 1.0 range
Screen('ColorRange', win, 1);

% Query window size in pixels
[w_pixels, h_pixels] = Screen('WindowSize', win);

% Retrieve monitor refresh duration
ifi = Screen('GetFlipInterval', win);

dgshader = [directory '/SequenceStim.vert.txt'];
GratingShader = LoadGLSLProgramFromFiles({ dgshader, [directory '/SequenceStim.frag.txt'] }, 1);
gratingTex = Screen('SetOpenGLTexture', win, [], 0, GL.TEXTURE_3D,w_pixels,...
    h_pixels, 1, GratingShader);

% screen size in millimeters and a conversion factor to get from mm to pixels
[w_mm,h_mm] = Screen('DisplaySize',screenid);
conv_factor = (w_mm/w_pixels+h_mm/h_pixels)/2;
mmPerPixel = conv_factor;

spatFreq = spatFreq*180/pi;
DistToScreenPix = DistToScreen*10/mmPerPixel;

centerVals = [w_pixels/2,85/mmPerPixel];
centerPos = [0,0].*pi/180;

tmp = binornd(1,stimTypeProbs(Day));

if tmp==1
   X = X{Day,1};
   reps = type1Reps;
   blocks = type1Blocks;
   stimType = 1;
elseif tmp==0
   X = X{Day,2};
   reps = type2Reps;
   blocks = type2Blocks;
   stimType = 2;
end


repsPerBlock = round(reps/blocks);

waitTimes = ISI(1)+(ISI(2)-ISI(1)).*rand([reps,1]);

stimParams = zeros(reps,numElements*2);
stimEventCodes = zeros(reps,numElements);

for ii=1:reps
    index = find(X(ii,1:4));
    
    if index==1
        stimParams(ii,1:4) = [abcdOrients(1),X(ii,5),abcdOrients(3),abcdOrients(4)];
        stimParams(ii,5:end) = [Contrast,X(ii,6),Contrast,Contrast];
        stimEventCodes(ii,:) = 20:20+numElements-1;
    elseif index==2
        stimParams(ii,1:4) = [abcdOrients(1),X(ii,5),X(ii,5),abcdOrients(4)];
        stimParams(ii,5:end) = [Contrast,X(ii,6),X(ii,6),Contrast];
        stimEventCodes(ii,:) = 30:30+numElements-1;
    elseif index==3
        stimParams(ii,1:4) = [ebcdOrients(1),X(ii,5),ebcdOrients(3),ebcdOrients(4)];
        stimParams(ii,5:end) = [Contrast,X(ii,6),Contrast,Contrast];
        stimEventCodes(ii,:) = 40:40+numElements-1;
    elseif index==4
        stimParams(ii,1:4) = [ebcdOrients(1),X(ii,5),X(ii,5),ebcdOrients(4)];
        stimParams(ii,5:end) = [Contrast,X(ii,6),X(ii,6),Contrast];
        stimEventCodes(ii,:) = 50:50+numElements-1;
    end
end

fprintf('\nStimulus Type: %d\n',stimType);

estimatedTime = ((mean(waitTimes)+stimTime*4)*reps+blocks*holdTime/2+holdTime+2)/60;
fprintf('\nEstimated time: %3.2f minutes\n',estimatedTime);

% Define first and second ring color as RGBA vector with normalized color
% component range between 0.0 and 1.0, based on Contrast between 0 and 1
% create all textures in the same window (win), each of the appropriate
% size
Grey = 0.5;
Black = 0;
White = 1;

offsetGrey = 6;

% Screen('BlendFunction',win,GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);

% Perform initial flip to gray background and sync us to the retrace:
Priority(9);

usb.startRecording;WaitSecs(1);usb.strobeEventWordOE(1);
WaitSecs(holdTime);

% Animation loop
count = 1;
for yy=1:blocks
    for zz = 1:repsPerBlock
        currentOrients = stimParams(count,1:4).*pi/180;
        currentContrasts = stimParams(count,5:8);
        vbl = Screen('Flip',win);
        ww = 0;
        while ww<numElements
            % ELEMENT on
            Screen('DrawTexture', win,gratingTex, [],[],...
                [],[],[],[Grey Grey Grey Grey],...
                [], [],[White,Black,...
                radianRadius,centerVals(1),centerVals(2),spatFreq,currentOrients(ww+1),...
                phase(ww+1),DistToScreenPix,centerPos(1),centerPos(2),currentContrasts(ww+1)]);
            % Request stimulus onset
            vbl = Screen('Flip',win,vbl-ifi/2+stimTime); % +ifi/2+(currentPause-stimOnTime)
            usb.strobeEventWordOE(stimEventCodes(count,ww+1));
            ww = ww+1;
        end
        vbl = Screen('Flip',win,vbl-ifi/2+stimTime);
        usb.strobeEventWordOE(offsetGrey);
        %             vbl = Screen('Flip',win,vbl-ifi/2+stimOnTime);
        vbl = Screen('Flip',win,vbl-ifi/2+waitTimes(count));
        count = count+1;
    end

    timeIncrement = 1;
    totalTime = timeIncrement;
    while totalTime<=holdTime/2
        usb.strobeEventWordOE(1);
        vbl = Screen('Flip',win,vbl-ifi/2+timeIncrement);
        totalTime = totalTime+timeIncrement;
    end
end
WaitSecs(1);
usb.stopRecording;
Priority(0);
    

spatFreq = spatFreq*pi/180;

fileName = sprintf('SeqRFStim_Day%d_Type%d-%d-%d.mat',Day,stimType,Date,AnimalName);
save(fileName,'stimParams','stimEventCodes','reps','stimTime','ISI','gama',...
    'w_pixels','h_pixels','spatFreq','mmPerPixel','waitTimes','holdTime',...
    'DistToScreen','offsetGrey','Day','Contrast','stimType','X','phase');
% Close window
Screen('CloseAll');

end

function gammaTable = makeGrayscaleGammaTable(gamma,blackSetPoint,whiteSetPoint)
% Generates a 256x3 gamma lookup table suitable for use with the
% psychtoolbox Screen('LoadNormalizedGammaTable',win,gammaTable) command
% 
% gammaTable = makeGrayscaleGammaTable(gamma,blackSetPoint,whiteSetPoint)
%
%   gamma defines the level of gamma correction (1.8 or 2.2 common)
%   blackSetPoint should be the highest value that results in a non-unique
%   luminance value on the monitor being used (sometimes values 0,1,2, all
%   produce the same black pixel value; set to zero if this is not a
%   concern)
%   whiteSetPoint should be the lowest value that returns a non-unique
%   luminance value (deal with any saturation at the high end)
% 
%   Both black and white set points should be defined on a 0:255 scale

gamma = max([gamma 1e-4]); % handle zero gamma case
gammaVals = linspace(blackSetPoint/255,whiteSetPoint/255,256).^(1./gamma);
gammaTable = repmat(gammaVals(:),1,3);
end
