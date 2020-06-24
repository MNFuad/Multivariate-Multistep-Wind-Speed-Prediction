clearvars

ExType = 'H1_18';
%%
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

% prepare data for NARX
X = tonndata(X,false,false);
T = tonndata(T,false,false);

% creat narx net
net = narxnet(1:1,1:1,10);% neurons=10

%% split the combined data again to train-validation according to given
% indices
[trainInd,valInd,testInd] = ...
divideind(Nt+Nv,1:Nt,Nt+1:Nt+Nv,[]);
% shuffle (randperf could replace divideind)
% indTr     = randperm(Nt); 
% trainInd = trainInd(indTr);
% assign the customized net parameters
net.trainFcn    = 'trainlm';% 'trainlm' 'trainbr'
net.divideFcn   = 'divideind';
net.divideParam.trainInd = trainInd;
net.divideParam.valInd = valInd;
net.divideParam.testInd = [];
net.divideMode  = 'sampletime';
net.trainParam.goal=1e-6;

%% train the network
[Xs,Xi,Ai,Ts] = preparets(net,X,{},T);
[net,tr] = train(net,Xs,Ts,Xi,Ai);

% save NN model to disk
saveTo = [pwd '\Gamal_NN\NARX\'];
if ~exist(saveTo,'dir')
    mkdir(saveTo)
end
save([saveTo ExType '.mat'],'net')

%% Close loop (bad performance on train data itself??)
% net2 = closeloop(net);
% view(net2)
% [Xs,Xi,Ai,Ts] = preparets(net2,X,{},T);
% Yclosed = net2(Xs,Xi,Ai);
% figure;plot(cell2mat(T(1:end-1)));hold on;
% plot(cell2mat(Yclosed)); title('Closed loop')

%% Validation
[d1,d2,d3] = size(data.XVal);
Xv = tonndata(reshape(data.XVal,d1,d2*d3),false,false);
Tv = tonndata(data.YVal,false,false);
[Xs,Xi,Ai,Tss] = preparets(net,Xv,{},Tv);
PredVal = net(Xs,Xi,Ai);

figure;plot(cell2mat(Tss)');hold on;
plot(cell2mat(PredVal)'); title('narx - Val')
%% performance on validation data/ we never used validation during net training
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
Xs = tonndata(reshape(data.XTest,d1,d2*d3),false,false);
Ts = tonndata(data.YTest,false,false);
[Xs,Xi,Ai,Tss] = preparets(net,Xs,{},Ts);

PredTest = net(Xs);

figure;plot(cell2mat(Tss)');hold on;
plot(cell2mat(PredTest)'); title('narx - Test')
% figure;scatter(cell2mat(Tss),cell2mat(PredTest))
%% performance on Testing data
truth = cell2mat(Tss);
pred = cell2mat(PredTest);
MSE  = mean(mean((truth' - pred').^2));   % Mean Squared Error
RMSE = mean(MSE.^0.5);
MAPE = mean(mean(abs(truth'-pred')./abs(truth')))*100;
MAE  = mean(mean(abs(truth'-pred')));

str = sprintf('Testing Results    ---> MAE =%0.4f, MSE =%0.4f, RMSE =%0.4f, MAPE =%0.4f',...
    MAE,MSE,RMSE,MAPE);
disp(str);

