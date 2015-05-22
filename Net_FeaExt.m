function [f, BlkIdx] = Net_FeaExt(InImg,V,M,P,Net)
% =======INPUT=============
% InImg     Input images (cell)
% V         given Net filter banks (cell)
% Net    Net parameters (struct)
%       .Net.NumStages
%           the number of stages in Net; e.g., 2
%       .PatchSize
%           the patch size (filter size) for square patches; e.g., 3, 5, 7
%           only a odd number allowed
%       .NumFilters
%           the number of filters in each stage; e.g., [16 8] means 16 and
%           8 filters in the first stage and second stage, respectively
%       .HistBlockSize
%           the size of each block for local histogram; e.g., [10 10]
%       .BlkOverLapRatio
%           overlapped block region ratio; e.g., 0 means no overlapped
%           between blocks, and 0.3 means 30% of blocksize is overlapped
% =======OUTPUT============
% f         Net features (each column corresponds to feature of each image)
% BlkIdx    index of local block from which the histogram is compuated

addpath('./Utils')

if length(Net.NumFilters)~= Net.NumStages;
    display('Length(Net.NumFilters)~=Net.NumStages')
    return
end

NumImg = length(InImg);

OutImg = InImg;
ImgIdx = (1:NumImg)';
clear InImg;
g = cell(Net.NumStages-1,1);
for stage = 1:Net.NumStages
    [OutImg, ImgIdx] = Net_output(OutImg, ImgIdx, ...
        Net.PatchSize, Net.NumFilters(stage), V{stage}, M{stage}, P{stage}, Net.Whitten, Net.SignSqrtNorm, Net.SigPara(stage,2));
    
    [g{stage,1}, ~] = HashingHist(Net,ImgIdx,OutImg);
end

[f, BlkIdx] = HashingHist(Net,ImgIdx,OutImg);

f = [f ; g{1,1}];

end