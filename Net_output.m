function [OutImg, OutImgIdx] = Net_output(InImg, InImgIdx, PatchSize, NumFilters, V, M, P, Whitten, SigSqrtNorm, sigscale)
% Computing PCA filter outputs
% ======== INPUT ============
% InImg         Input images (cell structure); each cell can be either a matrix (Gray) or a 3D tensor (RGB)
% InImgIdx      Image index for InImg (column vector)
% PatchSize     Patch size (or filter size); the patch is set to be sqaure
% NumFilters    Number of filters at the stage right before the output layer
% V             Net filter banks (cell structure); V{i} for filter bank in the ith stage
% ======== OUTPUT ===========
% OutImg           filter output (cell structure)
% OutImgIdx        Image index for OutImg (column vector)


addpath('./Utils')

ImgZ = length(InImg);
mag = (PatchSize-1)/2;
OutImg = cell(NumFilters*ImgZ,1);
cnt = 0;
for i = 1:ImgZ
    [ImgX, ImgY, NumChls] = size(InImg{i});
    img = zeros(round(ImgX+PatchSize-1),round(ImgY+PatchSize-1), round(NumChls));
    img((mag+1):end-mag,(mag+1):end-mag,:) = InImg{i};
    im = im2col_general(double(real(img)),[PatchSize PatchSize]); % collect all the patches of the ith image in a matrix
    %     im = bsxfun(@minus, im, mean(im)); % patch-mean removal
    % normalize for contrast
    patchestmp = im';
    patchestmp = bsxfun(@rdivide, bsxfun(@minus, patchestmp, mean(patchestmp,2)), sqrt(var(patchestmp,[],2)+10));
    if (Whitten == 1)
        patchestmp = bsxfun(@minus, patchestmp, M) * P;
    end
    im = patchestmp';
    clear patchestmp;
    
    for j = 1:NumFilters
        cnt = cnt + 1;
        OutImg{cnt} = reshape(tanh(sigscale * V(:,j)'*im),ImgX,ImgY);
%         OutImg{cnt} = sigscale * reshape(V(:,j)'*im,ImgX,ImgY);  % convolution output
        %OutImg{cnt} = reshape( 1 ./ (1 + exp(-sigscale * V(:,j)'*im)),ImgX,ImgY);  % convolution output
        % Sined square root normalization
        if SigSqrtNorm == 1
            OutImg{cnt} = sign(OutImg{cnt}) .* sqrt(abs(OutImg{cnt}));
        end
    end
    InImg{i} = [];
    %            fprintf(1,'Layered Max Val %f Min Val %f\n',max(max(OutImg{:})),min(min(OutImg(:))));
end
OutImgIdx = kron(InImgIdx,ones(NumFilters,1));