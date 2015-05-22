function Filters = ELM_AE(data, nHidden, C, sigscale)
%ELM_AE Summary of this function goes here
%   Detailed explanation goes here
%% Run the extreme learning machine auto encoder (ELM_AE)
InputWeight=rand(nHidden,size(data,1))*2 -1;
if nHidden > size(data,1)
    InputWeight = orth(InputWeight);
else
    InputWeight = orth(InputWeight')';
end
BiasofHiddenNeurons=rand(nHidden,1)*2 -1;
BiasofHiddenNeurons=orth(BiasofHiddenNeurons);
tempH=InputWeight*data;                                           %   Release input of training data
clear InputWeight;
ind=ones(1,size(data,2));
BiasMatrix=BiasofHiddenNeurons(:,ind);              %   Extend the bias matrix BiasofHiddenNeurons to match the demention of H
tempH=tempH+BiasMatrix;

clear BiasMatrix;
clear BiasofHiddenNeurons;
% fprintf(1,'AutoEncorder Max Val %f Min Val %f\n',max(tempH(:)),min(tempH(:)));
H = tanh(sigscale*tempH);%1 ./ (1 + exp(-sigscale*tempH));
clear tempH sigscale;                                        %   Release the temparary array for calculation of hidden neuron output matrix H
if nHidden == size(data,1)
    [~,Filterstmp,~] = procrustNew( data',H');
else
    if C == 0
        Filterstmp =pinv(H') * data';                        % implementation without regularization factor //refer to 2006 Neurocomputing paper
    else
        rhohats = mean(H,2);
        rho = 0.05;                                                 %%%%%%%%%%
        KLsum = sum(rho * log(rho ./ rhohats) + (1-rho) * log((1-rho) ./ (1-rhohats)));
        
        Hsquare =  H * H';
        HsquareL = diag(max(Hsquare,[],2));
        Filterstmp=( ( eye(size(H,1)).*KLsum +HsquareL )*(1/C)+Hsquare) \ (H * data');
        clear Hsquare;
        clear HsquareL;
    end
end
Filters = Filterstmp';
end
