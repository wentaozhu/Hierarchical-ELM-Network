function [ dst ] = cellimresize( src, scale, method )
%CELLIMRESIZE implements the image in the cell resize with batch mode
%   Use GPU operation

dst = cell(length(src),1);
for i = 1 : length(src);
    gpuData = gpuArray(src{i,1});
    gpuDatatrans = imresize(gpuData, scale, method);
    dst{i,1} = gather(gpuDatatrans);
end

end