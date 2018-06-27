%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ASVspoof 2017 CHALLENGE:
% Audio replay detection challenge for automatic speaker verification anti-spoofing
% 
% http://www.spoofingchallenge.org/
% 
% ====================================================================================
% Matlab implementation of the baseline system for replay detection based
% on constant Q cepstral coefficients (CQCC) features + Gaussian Mixture Models (GMMs)
% ====================================================================================
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear; close all; clc;

% add required libraries to the path
addpath(genpath('utility'));
addpath(genpath('CQCC_v1.0'));
addpath(genpath('bosaris_toolkit'));
addpath(genpath('rastamat'));

% set paths to the wave files and protocols
pathToDatabase = fullfile('..', 'wav'); %Goes to main directory
trainProtocolFile = fullfile('..','protocol','CM_protocol', 'toy_train.trn');
devProtocolFile = fullfile('..','protocol','CM_protocol', 'toy_dev.ndx');

% read train protocol
fileID = fopen(trainProtocolFile);
protocol = textscan(fileID, '%s%s%s%s%s%s%s');
fclose(fileID);

% get file and label lists
filelist = protocol{2};
labels = protocol{4};
spoof_type = protocol{3};

% get indices of genuine and spoof files
genuineIdx = find(strcmp(labels,'human'));
spoofIdx = find(strcmp(labels,'spoof'));

%%Added functionality, indexes each spoofing attack (S1-S10)
for i=1:10
    s1 = int2str(i)
    s2 = 'S'
spoof_typeId{i} = strcat(s2,s1);
end
flip_spoof_typeId = transpose(spoof_typeId);

for i=1:length(flip_spoof_typeId)
    class(flip_spoof_typeId{i});
    spoof_typeIdx{i} = find(strcmp(spoof_type,flip_spoof_typeId{i}))
    
        
        
end


%% Get features for different spoof types(For Dev and Eval)

for i=1:10
    s1 = int2str(i)
    s2 = 'S'
spoof_typeId{i} = strcat(s2,s1);
end
flip_spoof_typeId = transpose(spoof_typeId);

for i=1:length(flip_spoof_typeId)
    spoof_typeIdx{i} = strcmp(spoof_type,flip_spoof_typeId{i}); 
end


%% Feature extraction for training data

% extract features for GENUINE training data and store in cell array
disp('Extracting features for GENUINE training data...');
genuineFeatureCell = cell(size(genuineIdx));
s2 = '.wav'
parfor i=1:length(genuineIdx)
    s1 = filelist{genuineIdx(i)};
    filename = strcat(s1,s2)
    filePath = fullfile(pathToDatabase,'train',filename);
    [x,fs] = audioread(filePath);
    genuineFeatureCell{i} = melfcc(x, fs,'maxfreq',8000);
end
disp('Done!');

% extract features for SPOOF training data and store in cell array
disp('Extracting features for SPOOF training data...');
spoofFeatureCell = cell(size(spoofIdx));
parfor i=1:length(spoofIdx)
    s1 = filelist{spoofIdx(i)};
    filename = strcat(s1,s2)
    filePath = fullfile(pathToDatabase,'train',filename);
    [x,fs] = audioread(filePath);
    spoofFeatureCell{i} = melfcc(x, fs, 'maxfreq',8000);
end
disp('Done!');

%% GMM training

% train GMM for GENUINE data
disp('Training GMM for GENUINE...');
[genuineGMM.m, genuineGMM.s, genuineGMM.w] = vl_gmm([genuineFeatureCell{:}], 512, 'verbose', 'MaxNumIterations',100);
disp('Done!');

% train GMM for SPOOF data
disp('Training GMM for SPOOF...');
[spoofGMM.m, spoofGMM.s, spoofGMM.w] = vl_gmm([spoofFeatureCell{:}], 512, 'verbose', 'MaxNumIterations',100);
disp('Done!');


%% Feature extraction and scoring of development data

% read development protocol
fileID = fopen(devProtocolFile);
protocol = textscan(fileID, '%s%s%s%s%s%s%s');
fclose(fileID);

% get file and label lists
filelist_dev = protocol{2};
labels_dev = protocol{4};
spoof_type_dev = protocol{3};

%%Added functionality, indexes each spoofing attack (S1-S10)
for i=1:10
    s1 = int2str(i)
    s2 = 'S'
spoof_typeId{i} = strcat(s2,s1);
end
flip_spoof_typeId = transpose(spoof_typeId);

for i=1:length(flip_spoof_typeId)
    spoof_typeIdx{i} = find(strcmp(spoof_type_dev,flip_spoof_typeId{i}))   
end
% spoof_typeIdx = transpose(spoof_typeIdx);
%% process each development trial: feature extraction and scoring

disp('Computing scores for development trials...');

% Merges human and synthetic labels together based on Synth attack type
for i=1:length(genuineIdx)
    s1 = labels_dev{genuineIdx(i)};
    s2 = filelist_dev{genuineIdx(i)};
    s3 = '.wav';
    labelList_human(i) = cellstr(s1);
    fileList_human(i) = cellstr(strcat(s2,s3));
end

for j=1:5
    cell = spoof_typeIdx{j};
    for i=1:length(cell)
        s1 = labels_dev{cell(i)};
        s2 = filelist_dev{cell(i)};
        s3 = '.wav';
        labelList_synth(i) = cellstr(s1);
        fileList_synth(i) = cellstr(strcat(s2,s3));
        if i == length(cell)
            LabelList_all{j} = horzcat(labelList_human,labelList_synth);
            fileList_all{j} = horzcat(fileList_human,fileList_synth);
        end
    end
end
% genuineIdx = transpose(genuineIdx);


for j=1:5
    scores = zeros(size(fileList_all{j}));    
    parfor i=1:length(fileList_all{j});
        tmp_fileList = fileList_all{j};
        filename = char(tmp_fileList(i));
        filePath = fullfile(pathToDatabase,'toy-dev',filename);
        disp(filePath);
        [x,fs] = audioread(filePath);
        %featrue extraction
        x_mfcc = melfcc(x, fs, 'maxfreq', 8000);

        %score computation
        llk_genuine = mean(compute_llk(x_mfcc,genuineGMM.m,genuineGMM.s,genuineGMM.w));
        llk_spoof = mean(compute_llk(x_mfcc,spoofGMM.m,spoofGMM.s,spoofGMM.w));
        disp(llk_genuine)
        disp(llk_spoof)
        %compute log-likelihood ratio
        scores(i) = llk_genuine - llk_spoof;
        disp(scores(i));
        
    end
    [Pmiss,Pfa] = rocch(scores(strcmp(LabelList_all{j},'human')),scores(strcmp(LabelList_all{j},'spoof')));
    EER(j) = rocch2eer(Pmiss,Pfa) * 100; 
    fprintf('EER for spoof type is %.2f\n', EER);
    
    
    disp('Done!');
end
% compute performance
% [Pmiss,Pfa] = rocch(scores(strcmp(labels,'genuine')),scores(strcmp(labels,'spoof')));
% EER = rocch2eer(Pmiss,Pfa) * 100; 
% fprintf('EER is %.2f\n', EER);
