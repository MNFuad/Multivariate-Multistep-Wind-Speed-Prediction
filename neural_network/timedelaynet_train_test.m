
clearvars

ExType = 'H1_18';
%% Trian feedforwardnet
data = importdata(['../CV_Partition_Train/Norm_Train_Val_data_lag10_' ExType '.mat']);
[d1,d2,d3] = size(data.XTrain);
X = reshape(data.XTrain,d1,d2*d3);
T = data.YTrain;
Nt = size(X,1);
[d1,d2,d3] = size(data.XVal);
Xv = reshape(data.XVal,d1,d2*d3);
Tv = data.YVal;
Nv = size(Xv,1);

% combine train and validation
X = [X;Xv];
T = [T;Tv];

% prepare data for TDNN
X = tonndata(X,false,false);
T = tonndata(T,false,false);

% create the FFNet 
net = timedelaynet(1:1,10);% neurons = 10

%% split the combined data again to train-validation according to given
% indices
[trainInd,valInd,testInd] = ...
divideind(Nt+Nv,1:Nt,Nt+1:Nt+Nv,[]);
% assign the customized net parameters
net.trainFcn    = 'trainlm';% 'trainlm' 'trainbr'
net.divideFcn   = 'divideind';
net.divideParam.trainInd = trainInd;
net.divideParam.valInd = valInd;
net.divideParam.testInd = [];
net.divideMode  = 'sampletime';
%% train the network
[Xs,Xi,Ai,Ts] = preparets(net,X,T);
[net,tr] = train(net,Xs,Ts,Xi,Ai);

% save NN model to disk
saveTo = [pwd '\Gamal_NN\TDNN\'];
if ~exist(saveTo,'dir')
    mkdir(saveTo)
end
save([saveTo ExType '.mat'],'net')
%% Validation
% prepare data for TDNN
Xv = tonndata(Xv,false,false);
Tv = tonndata(Tv,false,false);
[Xs,Xi,Ai,Ts] = preparets(net,Xv,Tv);
PredVal = net(Xs,Xi);

figure;plot(cell2mat(Ts)');hold on;
plot(cell2mat(PredVal)'); title('TDNN - Val')
%% performance on validation data
truth = cell2mat(Tv(2:end));
pred = cell2mat(PredVal);
MSE  = mean(mean((truth' - pred').^2));   % Mean Squared Error
RMSE = mean(MSE.^0.5);
MAPE = mean(mean(abs(truth'-pred')./abs(truth')))*100;
MAE  = mean(mean(abs(truth'-pred')));

str = sprintf('Validation Results ---> MAE =%0.4f, MSE =%0.4f, RMSE =%0.4f, MAPE =%0.4f',...
    MAE,MSE,RMSE,MAPE);
disp(str);
%% Testing
data = importdata(['../CV_Partition_Test/Norm_Test_data_lag10_' ExType '.mat']);
[d1,d2,d3] = size(data.XTest);
Xs = reshape(data.XTest,d1,d2*d3);
Ts = data.YTest;

% prepare data for TDNN
Xs = tonndata(Xs,false,false);
Ts = tonndata(Ts,false,false);
[Xs,Xi] = preparets(net,Xs,Ts);
PredTest = net(Xs,Xi);

figure;plot(cell2mat(Ts(2:end))');hold on;
plot(cell2mat(PredTest)'); title('TDNN - Test')
%% performance on Testing data
truth = cell2mat(Ts(2:end));
pred = cell2mat(PredTest);
MSE  = mean(mean((truth' - pred').^2));   % Mean Squared Error
RMSE = mean(MSE.^0.5);
MAPE = mean(mean(abs(truth'-pred')./abs(truth')))*100;
MAE  = mean(mean(abs(truth'-pred')));

str = sprintf('Testing Results    ---> MAE =%0.4f, MSE =%0.4f, RMSE =%0.4f, MAPE =%0.4f',...
    MAE,MSE,RMSE,MAPE);
disp(str);