%0均值，1方差，X为n个列向量的样本，x_mean为均值，列向量。
function M = ZeroMeanOneVar(X,x_mean,x_var)
index = x_var<0.0000001;
x_var(index) = 1;
% x_mean(index) = 0;

num = size(X,2);
X_mean = repmat(x_mean, 1, num); % too many memory needed
M = X - X_mean;
clear X_mean;
X_var = repmat(x_var, 1, num);
M = M ./ X_var;
clear X_var;
flag = isnan(M);
M(flag) = 0;
end