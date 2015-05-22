function [f, BlkIdx] = HashingHist(Net,ImgIdx,OutImg)
% Output layer of Net (Hashing plus local histogram)
% ========= INPUT ============
% Net  Net parameters (struct)
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
% ImgIdx  Image index for OutImg (column vector)
% OutImg  Net filter output before the last stage (cell structure)
% ========= OUTPUT ===========
% f       Net features (each column corresponds to feature of each image)
% BlkIdx  index of local block from which the histogram is compuated

addpath('./Utils')

NumImg = max(ImgIdx);
f = cell(NumImg,1);
map_weights = 2.^((Net.NumFilters(end)-1):-1:0); % weights for binary to decimal conversion

for Idx = 1:NumImg
  
    Idx_span = find(ImgIdx == Idx);
    Bhist = cell(length(Idx_span),1);
    NumImginO = length(Idx_span)/Net.NumFilters(end); % the number of feature maps in "\cal O"
    
    for i = 1:NumImginO 
        
        T = zeros(size(OutImg(Idx_span(1))));
        if Net.HistBlockSize(1) < 1    %%% Only used for Caltech
            BlockSize = [floor(Net.HistBlockSize(1) * size(OutImg{Idx_span(Net.NumFilters(end)*(i-1)+1)},1)), ...
                floor(Net.HistBlockSize(2) * size(OutImg{Idx_span(Net.NumFilters(end)*(i-1)+1)},2))];
            for j = 1:Net.NumFilters(end)
                T = T + map_weights(j)*Heaviside(OutImg{Idx_span(Net.NumFilters(end)*(i-1)+j)});
                % weighted combination; hashing codes to decimal number conversion 
                OutImg{Idx_span(Net.NumFilters(end)*(i-1)+j)} = [];
            end
            %%% This is will cause dimension not consistent
            %%% Calculate the minimun numbers of patches
            X_Size = 1 / (Net.HistBlockSize(1) * Net.BlkOverLapRatio);  % 8
            X_Size = X_Size - X_Size * Net.HistBlockSize(1) + 1;    % 7
            Y_Size = 1 / (Net.HistBlockSize(2) * Net.BlkOverLapRatio);
            Y_Size = Y_Size - Y_Size * Net.HistBlockSize(2) + 1;
            %%% Check x direction number of patches
            ResidualX = floor((size(T,1) - BlockSize(1)) / floor(BlockSize(1)*(1-Net.BlkOverLapRatio))) + 1;
            if ResidualX > X_Size
                T = T(1:end-(ResidualX-X_Size)*floor(BlockSize(1)*(1-Net.BlkOverLapRatio)),:);
            elseif ResidualX < X_Size
                ResidualX
                X_Size
                size(T)
                BlockSize
            end
            %%% Check y direction number of patches
            ResidualY = floor((size(T,2) - BlockSize(2)) / floor(BlockSize(2)*(1-Net.BlkOverLapRatio))) + 1;
            if ResidualY > Y_Size
                T = T(:,1:end-(ResidualY-Y_Size)*floor(BlockSize(2)*(1-Net.BlkOverLapRatio)));
            elseif ResidualY < Y_Size
                ResidualX
                X_Size
                size(T)
                BlockSize
            end
            Bhist{i} = sparse(histc(im2col_general(double(T),BlockSize,...
                floor((1-Net.BlkOverLapRatio)*BlockSize)),(0:2^Net.NumFilters(end)-1)'));
            % calculate histogram for each local block in "T"
        else
            for j = 1:Net.NumFilters(end)
                T = T + map_weights(j)*Heaviside(real(OutImg{Idx_span(Net.NumFilters(end)*(i-1)+j)}));
                % weighted combination; hashing codes to decimal number conversion
                OutImg{Idx_span(Net.NumFilters(end)*(i-1)+j)} = [];
            end
            Bhist{i} = sparse(histc(im2col_general(double(T),Net.HistBlockSize,...
                round((1-Net.BlkOverLapRatio)*Net.HistBlockSize)),(0:2^Net.NumFilters(end)-1)'));
            % calculate histogram for each local block in "T"
        end
        Bhist{i} = bsxfun(@times, Bhist{i}, ...
            2^Net.NumFilters(end)./sum(Bhist{i})); % to ensure that sum of each block-wise histogram is equal 
    end           
    f{Idx} = vec([Bhist{:}]);
end
f = [f{:}];
BlkIdx = kron(ones(NumImginO,1),kron((1:size(Bhist{1},2))',ones(size(Bhist{1},1),1)));

%-------------------------------
function X = Heaviside(X) % binary quantization
X = sign(X);
X(X<=0) = 0;

function x = vec(X) % vectorization
x = X(:);