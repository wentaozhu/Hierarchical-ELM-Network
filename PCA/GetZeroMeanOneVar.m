function [x_mean] = GetZeroMeanOneVar(X)
% Y = double(X); % too many memory needed
x_mean = mean(X,2); %mean rows
% X = bsxfun(@minus, X, x_mean);
% x_var = zeros(size(X,1),1);
% for i = 1 : 100 : size(X,2)
%     if i + 99 > size(X,2)
%         X(:,i:end) = bsxfun(@minus, X(:,i:end), x_mean);
%         x_var = x_var + sum(X(:,i:end) .* X(:,i:end) , 2);
%         break;
%     end
%     X(:,i:i+99) = bsxfun(@minus, X(:,i:i+99), x_mean);
%     x_var = x_var + sum(X(:,i:i+99) .* X(:,i:i+99) , 2);
% end
% x_var = sqrt(x_var / size(X,2));

% x_var = std(X,0,2);