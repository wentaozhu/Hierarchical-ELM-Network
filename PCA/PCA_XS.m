function   [x_mean W_pca eig_value] = PCA_XS(X_pca, PCA_Dim, lamda)
%% ***********数据预处理********************************************
if nargin < 3 lamda = 0; end
dim = size(X_pca,1);
sample_num = size(X_pca,2);
% [x_mean x_var] = GetZeroMeanOneVar(X_pca);
x_mean = mean(X_pca,2);
% [x_mean] = GetZeroMeanOneVar(X_pca);
fprintf('\n ====== Mean has been calculated ======= \n')
% save memory
for i = 1 : 1000 : size(X_pca,2)
%     X_pca(:,i) = (X_pca(:,i) - x_mean) ./ x_var;
if i+1000 < size(X_pca,2)
 X_pca(:,i:i+1000-1) = bsxfun(@minus,X_pca(:,i:i+1000-1), x_mean);
else
    X_pca(:,i:end) = bsxfun(@minus,X_pca(:,i:end), x_mean);
    break;
end
end
save('PCA_XS_x_mean.mat','x_mean','-v7.3');
clear x_mean;
% X_pca = bsxfun(@minus,X_pca,x_mean);
fprintf('\n ====== Mean removal has done ======= \n')
%X_pca = ZeroMeanOneVar(X_pca, x_mean,x_var);

pca_dim = PCA_Dim;
%% ***********  PCA ***********************************************
if dim<sample_num  
    St = X_pca * X_pca'+ lamda*eye(dim);
    clear X_pca;
    fprintf('\n ====== Covariance has been calculated ======= \n')
    [Vt Dt v] = svd(St);
    fprintf('\n ====== SVD has been calculated ======= \n')
    eig_value_unsort = diag(Dt);
    [eig_value  eig_index]  = sort(eig_value_unsort,'descend');
    energy = sum(eig_value);
    
    for i=1:dim
        kept_energy = sum(eig_value(1:i));
        if kept_energy > energy*0.98
            fprintf('%d dims can preserve 0.98 energy',i);
            break;
        end
    end
    
    W_pca = zeros(dim,PCA_Dim);  %W_pca的列向量为主成分
    for i = 1: PCA_Dim
        W_pca(:, i) = Vt(:,eig_index(i));
    end
else 
    St = X_pca' * X_pca + lamda*eye(sample_num);
    save('PCA_XS_X_pca.mat','X_pca','-v7.3');
    clear X_pca;
    fprintf('\n ====== Covariance has been calculated ======= \n')
    [Vt Dt v] = svd(St);
    fprintf('\n ====== SVD has been calculated ======= \n')
    eig_value_unsort = diag(Dt);
    [eig_value  eig_index]  = sort(eig_value_unsort,'descend');
    energy = sum(eig_value);
    
    for i=1:sample_num
        kept_energy = sum(eig_value(1:i));
        if kept_energy > energy*0.98
            fprintf('%d dims can preserve 0.98 energy',i);
            break;
        end
    end
    fprintf('PCA dim:%d\n',pca_dim);
    W_pca = zeros(dim, PCA_Dim);  %W_pca的列向量为主成分
    load PCA_XS_X_pca.mat;
    for i = 1:PCA_Dim
        W_pca(:, i) = X_pca*Vt(:,eig_index(i));
    end
    clear X_pca;
end

for i = 1: PCA_Dim
    v = W_pca(:, i);
    v = v/norm(v);
    W_pca(:, i) = v;
end
% save(model_path, 'x_mean', 'x_var', 'W_pca');
fprintf('Trainset PCA Model Finished\n');
save PCA_XS_x_mean.mat;