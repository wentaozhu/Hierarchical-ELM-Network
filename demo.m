clear all; close all; clc; 
addpath('./Utils');
addpath('./Liblinear');
addpath('./common');
TrnSize = 12000; 
%TrnSize = 60000; 
ImgSize = 28; 
ImgFormat = 'gray'; %'color' or 'gray'

%% Loading data from MNIST Basic (10000 training, 2000 validation, 50000 testing) 
load('F:\Data\MNISTdata\mnist_basic');
% load('F:\Data\MNISTdata\mnist_train'); 
% load('F:\Data\MNISTdata\mnist_test'); 

% mnist_train = load('F:\Data\Data Set\mnist_rotation_new\mnist_all_rotation_normalized_float_train_valid.amat');
% mnist_test = load('F:\Data\Data Set\mnist_rotation_new\mnist_all_rotation_normalized_float_test.amat');

% mnist_train = load('F:\Data\Data Set\rectangles\rectangles_train.amat');
% mnist_test = load('F:\Data\Data Set\rectangles\rectangles_test.amat');

%load('./MNISTdata/mnist_train');
%mnist_train = [train_X train_labels];
%clear trainX train_labels;
%load('./MNISTdata/mnist_test');
%mnist_test = [test_X test_labels];
%clear testX test_labels;
% ===== Reshuffle the training data =====
Randnidx = randperm(size(mnist_train,1)); 
mnist_train = mnist_train(Randnidx,:); 
% =======================================

TrnData = mnist_train(1:TrnSize,1:end-1)';  % partition the data into training set and validation set
TrnLabels = mnist_train(1:TrnSize,end);
ValData = mnist_train(TrnSize+1:end,1:end-1)';
ValLabels = mnist_train(TrnSize+1:end,end);
clear mnist_train;

TestData = mnist_test(:,1:end-1)';
TestLabels = mnist_test(:,end);
clear mnist_test;


% ==== Subsampling the Training and Testing sets ============
% (comment out the following four lines for a complete test) 
% TrnData = TrnData(:,1:4:end);  % sample around 2500 training samples
% TrnLabels = TrnLabels(1:4:end); % 
% TestData = TestData(:,1:50:end);  % sample around 1000 test samples  
% TestLabels = TestLabels(1:50:end); 
% ===========================================================

nTestImg = length(TestLabels);

%% Net parameters (they should be funed based on validation set; i.e., ValData & ValLabel)
% We use the parameters in our IEEE TPAMI submission
Net.NumStages = 2;
Net.PatchSize = 7;
Net.NumFilters = [8 8];  %[8 8]
Net.HistBlockSize = [7 7]; % if Net.HistBlockSize < 1, stands for block size is the ratio of the image size,[14 14]
Net.BlkOverLapRatio = 0.5;
Net.Poolingsize = 2;
Net.PoolingStride = 2;    % if stride == poolingsize, then no pad
Net.Pooling = 0;  % 0 stands for no pooling, 1 stands for max pooling, 2 stands for average pooling
Net.Whitten = 1;
Net.SignSqrtNorm = 0;
Net.Type = 'ELM';
Net.NormClassifier = 0;
Net.ResolutionFlag = 3; % 0 stands for standard net type; 1 stands for Laplacian Pyramid; 2 stands for multi scale version; 
%%% 3 stands for concatinates the first layer output to the last to classify.
Net.ResolutionNum = 2; % stands for how many scale used
Net.WPCA = 0; % stands for the dimensions that use wpca recuded, 0 stands for no wpca

if Net.ResolutionFlag == 0
    Net.ResolutionNum = 0;
end

fprintf('\n ====== Net Parameters ======= \n')
Net

%% Net Training with 10000 samples

fprintf('\n ====== Net Training ======= \n')
TrnData_ImgCell = mat2imgcell(TrnData,ImgSize,ImgSize,ImgFormat); % convert columns in TrnData to cells 
clear TrnData; 
tic;
[ftrain, V, M, P, BlkIdx] = Net_train(TrnData_ImgCell,Net,1); % BlkIdx serves the purpose of learning block-wise DR projection matrix; e.g., WPCA
if Net.WPCA ~= 0
    block_dim = 2 ^ Net.NumFilters(2);
    DR_WPCA = cell(size(ftrain,1)/block_dim,1);
    ftrain_DR = zeros(DR_WPCA*Net.WPCA,TrnSize);
    for i = 1 : length(DR_WPCA)
        [wcoeff,~,latent,~,explained] = pca(ftrain((i-1)*block_dim+1:i*block_dim,:)','VariableWeights','variance');
        coefforth = diag(std(ingredients)) \ wcoeff;
        DR_WPCA{i,1} = coefforth(:,1:Net.WPCA);
        ftrain_DR = DR_WPCA{i,1}' * ftrain((i-1)*block_dim+1:i*block_dim,:);
    end
    ftrain = ftrain_DR;
    clear ftrain_DR;
end
Net_TrnTime = toc;
clear TrnData_ImgCell; 

fprintf('\n ====== Training Linear SVM Classifier ======= \n')

% standardize data
ftrain = ftrain';
if Net.NormClassifier == 1
    trainXC_mean = mean(ftrain);
    % Incrementally calculate the var
    ind = ones(500,1); varftrain = zeros(1,size(ftrain,2));
    for i = 1 : 1000 : TrnSize
        if i+1000 >= TrnSize
            ftrain(i:end,:) = ftrain(i:end,:) - trainXC_mean(ones(TrnSize-i+1,1),:);
            varftrain = varftrain + sum(ftrain(i:end,:) .* ftrain(i:end,:));
        else
            ftrain(i:i+1000-1,:) = ftrain(i:i+1000-1,:) - trainXC_mean(ind,:);
            varftrain = varftrain + sum(ftrain(i:i+1000-1,:) .* ftrain(i:i+1000-1,:));
        end
    end
    varftrain = varftrain ./ (TrnSize-1);
    trainXC_sd = sqrt(varftrain+0.01);
    clear varftrain ind;
    ftrain = bsxfun(@rdivide, ftrain, trainXC_sd);
end

tic;
models = train(TrnLabels, ftrain, '-s 1 -B 1 -q'); % we use linear SVM classifier (C = 1), calling libsvm library
LinearSVM_TrnTime = toc;
clear ftrain; 

%% Net Feature Extraction and Testing 

TestData_ImgCell = mat2imgcell(TestData,ImgSize,ImgSize,ImgFormat); % convert columns in TestData to cells 
clear TestData; 

fprintf('\n ====== Net Testing ======= \n')

nCorrRecog = 0;
RecHistory = zeros(nTestImg,1);

tic; 
for idx = 1:1:nTestImg
    
    ftest = Net_FeaExt(TestData_ImgCell(idx),V,M,P,Net); % extract a test feature using trained Net model 
    if Net.WPCA ~= 0
        for i = 1 : length(DR_WPCA)
            ftest_DR = DR_WPCA{i,1}' * ftest((i-1)*block_dim+1:i*block_dim,:);
        end
        ftest = ftest_DR;
        clear ftest_DR;
    end
    ftest = ftest';
    if Net.NormClassifier == 1
        ftest = bsxfun(@rdivide, bsxfun(@minus, ftest, trainXC_mean), trainXC_sd);
    end
    [xLabel_est, accuracy, decision_values] = predict(TestLabels(idx),...
        sparse(ftest), models, '-q'); % label predictoin by libsvm
   
    if xLabel_est == TestLabels(idx)
        RecHistory(idx) = 1;
        nCorrRecog = nCorrRecog + 1;
    end
    
    if 0==mod(idx,nTestImg/10); 
        fprintf('Accuracy up to %d tests is %.2f%%; taking %.2f secs per testing sample on average. \n',...
            [idx 100*nCorrRecog/idx toc/idx]); 
    end 
    
    TestData_ImgCell{idx} = [];
    
end
Averaged_TimeperTest = toc/nTestImg;
Accuracy = nCorrRecog/nTestImg; 
ErRate = 1 - Accuracy;

%% Results display
fprintf('\n ===== Results of Net, followed by a linear SVM classifier =====');
fprintf('\n     Net training time: %.2f secs.', Net_TrnTime);
fprintf('\n     Linear SVM training time: %.2f secs.', LinearSVM_TrnTime);
fprintf('\n     Testing error rate: %.2f%%', 100*ErRate);
fprintf('\n     Average testing time %.2f secs per test sample. \n\n',Averaged_TimeperTest);