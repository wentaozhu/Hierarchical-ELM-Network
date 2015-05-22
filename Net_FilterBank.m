function [V, M, P] = Net_FilterBank(InImg, PatchSize, NumFilters, Whitten, Sigparain, SigC)
% =======INPUT=============
% InImg            Input images (cell structure)
% InImgIdx         Image index for InImg (column vector)
% PatchSize        the patch size, asumed to an odd number.
% NumFilters       the number of Net filters in the bank.
% givenV           the Net filters are given.
% =======OUTPUT============
% V                Net filter banks, arranged in column-by-column manner
% OutImg           filter output (cell structure)
% OutImgIdx        Image index for OutImg (column vector)

addpath('./Utils')
% to efficiently cope with the large training samples, we randomly subsample 100000 training subset to learn Net filter banks
ImgZ = length(InImg);
MaxSamples = 10000;
NumRSamples = min(ImgZ, MaxSamples);
RandIdx = randperm(ImgZ);
RandIdx = RandIdx(1:NumRSamples);
%% Learning Net filters (V)
% NumChls = size(InImg{1},3);
im = im2col_general(double(real(InImg{1})),[PatchSize PatchSize]); % collect all the patches of the ith image in a matrix 36 * 529
PatchperNum = size(im,2); NumPatches = PatchperNum*NumRSamples;
% Patches = zeros(NumPatches,PatchSize*PatchSize);    %%% Caltech 101 must
% delete it
PatchesFlag = 0;
for i = 1:NumRSamples
    im = im2col_general(double(real(InImg{RandIdx(i)})),[PatchSize PatchSize]); % collect all the patches of the ith image in a matrix 36 * 529
    % normalize for contrast
% %     im = bsxfun(@rdivide, bsxfun(@minus, im, mean(im)), sqrt(var(im,[],1)+10));
    %             im = bsxfun(@minus, im, mean(im)); % patch-mean removal
    %     Patches((i-1)*PatchperNum+1:i*PatchperNum,:) = im';
    imrand = randperm(size(im,2));
    im = im(:,imrand(1:ceil(size(im,2)/50)));
    Patches(PatchesFlag+1:PatchesFlag+size(im,2),:) = im';
    PatchesFlag = PatchesFlag+size(im,2);
    if PatchesFlag+size(im,2) > 100000
        NumPatches = PatchesFlag;     %%%%%%%%%%%%% Attentions!!!!
        break;
    end
    clear im;
    % patches = bsxfun(@rdivide, bsxfun(@minus, im, mean(im)), sqrt(var(im)));
end
if size(Patches,1) > 100000
    Idx = randperm(size(Patches,1));
    Patches = Patches(Idx(1:100000),:);
    NumPatches = 100000;
end
NumPatches = size(Patches,1);
clear Idx;
% normalize for contrast
Patches = bsxfun(@rdivide, bsxfun(@minus, Patches, mean(Patches,2)), sqrt(var(Patches,[],2)+10));
if Whitten == 0
    M = zeros(1, PatchSize*PatchSize);
    P = eye(PatchSize*PatchSize,PatchSize*PatchSize);
elseif Whitten == 1
    %     C = cov(Patches);
    M = mean(Patches);
    % Incrementally calculate the cov
    C = zeros(PatchSize*PatchSize, PatchSize*PatchSize);
    %         ind = ones(10000,1);
    for i = 1 : 10000 : NumPatches
        if i+10000 >= NumPatches
            Patches(i:end,:) = bsxfun(@minus,Patches(i:end,:), M);
            C = C + Patches(i:end,:)' * Patches(i:end,:);
            break;
        else
            Patches(i:i+10000-1,:) = bsxfun(@minus,Patches(i:i+10000-1,:), M);
            C = C + Patches(i:i+10000-1,:)' * Patches(i:i+10000-1,:);
        end
    end
    C = C ./ (NumPatches-1);
    [V,D] = eig(C);
    P = V * diag(sqrt(1./(diag(D) + diag(0.1)))) * V';
    clear ind V D C;
    Patches = Patches * P;
end
V = ELM_AE(Patches', NumFilters, 10^SigC, Sigparain);  %%% Patches 10000 * m
end