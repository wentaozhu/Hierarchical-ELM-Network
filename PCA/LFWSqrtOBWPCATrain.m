% set data
CsPattern = {'Middle','Inner','Exterior'};
tic
[LFWOBFeature] = LoadOBFeatureNew( 'D:/Experiment/Object_Bank/MATLAB_release/150_80FeatureMiddle/', 44604);
toc

for s = 1  :  1
    sPattern = CsPattern{s};
    load('LFWURID.mat');
    % rerange the LFW feature according to the unrestricted settings
    [OBPos, OBNeg] = SetLFWView2Data(LFWOBFeature);
    View2Pos = sqrt(abs(OBPos));
    View2Neg = sqrt(abs(OBNeg));
    
    clear LFWOBFeature;
    clear FaceIndex.mat;
    
    nSize = 13233;

    for PCA_Dim = 1000
        LFWSMatrix = zeros(600,1,10);
        % 10 folder test
        for folder = 1 : 10
           fprintf('Cal %d-th folder\n', folder);
           % Set training data and validation/test data
           % set test data
           test_start = (folder-1)*600+1;
           test_end   = folder*600;
           test_Pos = View2Pos(:,test_start:test_end);
           test_Neg = View2Neg(:,test_start:test_end);
           
           train_Pos = 1:6000;
           train_Neg = 1:6000;
           train_Pos(test_start:test_end) = [];
           train_Neg(test_start:test_end) = [];
           
           train_data =[View2Pos(:,train_Pos),View2Neg(:,train_Neg)];
           
           tic
            [x_mean, x_var, W_pca,eig_value] = PCA_XS(train_data, 3000);
           toc
           save(['./150_80MiddleR/PCASqrt_' num2str(folder) '.mat'], 'W_pca','x_mean','x_var','eig_value');
           
           
           W_pca = W_pca(:,1:PCA_Dim);
           eig_matrix = 1./sqrt(eig_value(1:PCA_Dim));
           eig_matrix = diag(eig_matrix);

           % test the model
           test_PosPCA = eig_matrix * W_pca' * ZeroMeanOneVar(test_Pos,x_mean,x_var);
           test_NegPCA = eig_matrix * W_pca' * ZeroMeanOneVar(test_Neg,x_mean,x_var); 
           sim_Pos = zeros(300,1);
           sim_Neg = zeros(300,1);
           for i = 1 : 300
                sim_Pos(i) = CalcCosineDist(test_PosPCA(:,2*i-1),test_PosPCA(:,2*i));
                sim_Neg(i) = CalcCosineDist(test_NegPCA(:,2*i-1),test_NegPCA(:,2*i));
           end

           sim_NegSort = sort(sim_Neg, 'descend');

           fprintf('PatchVR@0.1 = %f \n',length(find(sim_Pos>sim_NegSort(30)))/300);
           LFWSMatrix(:,1,folder) = [sim_Pos;sim_Neg];
        end

        save(sprintf('.\\LFWSqrt150_80OB%sWPCA%d.mat',sPattern,PCA_Dim),'LFWSMatrix');
        % Calc the accuray of current scale
        fprintf('Mean Accuracy of current scale = %f\n',LFWCAccuracy(LFWSMatrix));
    end
end

