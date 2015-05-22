function [ FilterMap ] = VisualFilter( FilterWeight, RFsize )
%VISUALFILTER Summary of this function goes here
%   Detailed explanation goes here
%%% Normalization

figure
FilterMap = zeros(size(FilterWeight,1), RFsize(1), RFsize(2));
if length(size(FilterWeight)) == 2
    
    for num = 1 : size(FilterWeight,1)
        Cursor = 1;
        Tmp = zeros(RFsize(1), RFsize(2));
        MaxWeight = max(FilterWeight(num,:)); MinWeight = min(FilterWeight(num,:));
        for i = 1 : RFsize(1)
            for j = 1 : RFsize(2)
%                 FilterMap(num, j, i) = (FilterWeight(Cursor) - MinWeight) / (MaxWeight - MinWeight);
                FilterMap(num, j, i) = FilterWeight(num,Cursor);
                Tmp(j,i) = FilterMap(num, j, i);
                Cursor = Cursor + 1;           
            end
        end
        subplot(6,8,num);
        imshow(Tmp,[min(Tmp(:)) max(Tmp(:))]);
    end
end

end

