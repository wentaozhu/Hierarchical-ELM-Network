function [ dst ] = cellimresizeminus( src1, src, scale, method )
%CELLIMRESIZEMINUS implements the image in the cell resize with batch mode,
%then use src1 to minus the resized image
%   Use GPU operation

dst = cell(length(src),1);
for i = 1 : length(src);
    gpuData = gpuArray(src{i,1});
    gpuDatatrans = imresize(gpuData, scale, method);
    gpuDatasrc1 = gpuArray(src1{i,1});
    gpuData = gpuDatasrc1 - gpuDatatrans;
    dst{i,1} = gather(gpuData);
end

end

