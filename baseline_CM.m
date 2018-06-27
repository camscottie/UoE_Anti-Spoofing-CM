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

% set paths to the wave files and protocols
pathToDatabase = fullfile('..', 'data'); %Goes to main directory
trainProtocolFile = fullfile('..','protocol_V2', 'ASVspoof2017_V2_train.trn.txt');
devProtocolFile = fullfile('..','protocol_V2', 'ASVspoof2017_V2_dev.trl.txt');

% read train protocol
fileID = fopen(trainProtocolFile);
protocol = textscan(fileID, '%s%s%s%s%s%s%s');
fclose(fileID);

% get file and label lists
filelist = protocol{1};
labels = protocol{2};

% get indices of genuine and spoof files
genuineIdx = find(strcmp(labels,'genuine'));
spoofIdx = find(strcmp(labels,'spoof'));

%% Feature extraction for training data

% extract features for GENUINE training data and store in cell array
disp('Extracting features for GENUINE training data...');
genuineFeatureCell = cell(size(genuineIdx));
parfor i=1:length(genuineIdx)
    filePath = fullfile(pathToDatabase,'train',filelist{genuineIdx(i)});
    [x,fs] = audioread(filePath);
    genuineFeatureCell{i} = cqcc(x, fs, 96, fs/2, fs/2^10, 16, 29, 'ZsdD');
end
disp('Done!');

% extract features for SPOOF training data and store in cell array
disp('Extracting features for SPOOF training data...');
spoofFeatureCell = cell(size(spoofIdx));
parfor i=1:length(spoofIdx)
    filePath = fullfile(pathToDatabase,'train',filelist{spoofIdx(i)});
    [x,fs] = audioread(filePath);
    spoofFeatureCell{i} = cqcc(x, fs, 96, fs/2, fs/2^10, 16, 29, 'ZsdD');
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
filelist = protocol{1};
labels = protocol{2};

% process each development trial: feature extraction and scoring
scores = zeros(size(filelist));
disp('Computing scores for development trials...');
parfor i=1:length(filelist)
    filePath = fullfile(pathToDatabase,'dev',filelist{i});
    [x,fs] = audioread(filePath);
    % featrue extraction
    x_cqcc = cqcc(x, fs, 96, fs/2, fs/2^10, 16, 29, 'ZsdD');

    %score computation
    llk_genuine = mean(compute_llk(x_cqcc,genuineGMM.m,genuineGMM.s,genuineGMM.w));
    llk_spoof = mean(compute_llk(x_cqcc,spoofGMM.m,spoofGMM.s,spoofGMM.w));
    % compute log-likelihood ratio
    scores(i) = llk_genuine - llk_spoof;
end
disp('Done!');

% compute performance
[Pmiss,Pfa] = rocch(scores(strcmp(labels,'genuine')),scores(strcmp(labels,'spoof')));
EER = rocch2eer(Pmiss,Pfa) * 100; 
fprintf('EER is %.2f\n', EER);
