
clearvars

ExType = 'H1_18';
%% Trian feedforwardnet
data = importdata(['../CV_Partition_Train/Norm_Train_Val_data_lag10_' ExType '.mat']);
[d1,d2,d3] = size(data.XTrain);
X = reshape(data.XTrain,d1,d2*d3)';
T = data.YTrain';
Nt = size(X,2);
[d1,d2,d3] = size(data.XVal);
Xv = reshape(data.XVal,d1,d2*d3)';
Tv = data.YVal';
Nv = size(Xv,2);

% combine train and validation
X = [X Xv];
T = [T Tv];



% create the FFNet 
net = feedforwardnet(10);% neurons = 10

% split the combined data again to train-validation according to given
% indices
[trainInd,valInd,testInd] = ...
divideind(Nt+Nv,1:Nt,Nt+1:Nt+Nv,[]);
% shuffle (randperf could replace divideind)
indTr     = randperm(Nt); 
trainInd = trainInd(indTr);
% assign the customized net parameters
net.trainFcn    = 'trainlm';% 'trainlm' 'trainbr'
net.divideFcn   = 'divideind';
net.divideParam.trainInd = trainInd;
net.divideParam.valInd = valInd;
net.divideParam.testInd = [];
net.divideMode  = 'sampletime';
net.trainParam.goal=1e-6;
% net.efficiency.memoryReduction=2;

% scale data
% net.inputConnect(1,1) = true;
% net.inputs{1}.processFcns = {'mapminmax'};
% % Outputs
% net.outputConnect(2) = true;
% net.outputs{2}.processFcns = {'mapminmax'};

% train the network
[net,tr] = train(net,X,T);%,'useParallel','yes');

% save NN model to disk
saveTo = [pwd '\Gamal_NN\FFNN\'];
if ~exist(saveTo,'dir')
    mkdir(saveTo)
end
save([saveTo ExType '.mat'],'net')
%% Validation
PredVal = net(Xv);

figure;plot(Tv');hold on;
plot(PredVal'); title('feedforwardnet - Val')
%% performance on validation data
truth = Tv;
pred = PredVal;
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
Xs = reshape(data.XTest,d1,d2*d3)';
Ts = data.YTest';

PredTest = net(Xs);

figure;plot(Ts');hold on;
plot(PredTest'); title('feedforwardnet - Test')
%% performance on Testing data
truth = Ts;
pred = PredTest;
MSE  = mean(mean((truth' - pred').^2));   % Mean Squared Error
RMSE = mean(MSE.^0.5);
MAPE = mean(mean(abs(truth'-pred')./abs(truth')))*100;
MAE  = mean(mean(abs(truth'-pred')));

str = sprintf('Testing Results    ---> MAE =%0.4f, MSE =%0.4f, RMSE =%0.4f, MAPE =%0.4f',...
    MAE,MSE,RMSE,MAPE);
disp(str);