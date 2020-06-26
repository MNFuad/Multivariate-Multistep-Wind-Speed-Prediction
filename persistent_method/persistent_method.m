
clearvars

ExType = 'H12';
lag = 10;
%%
data  = readtable('../../data_imputed_2018_10Min.csv');
wind_speed10m = data.Avg_Wind_Speed_10m;
if strcmp(ExType,'H1')
    wind_speed10m = wind_speed10m(10:end-1);
elseif strcmp(ExType,'H3')
    wind_speed10m = wind_speed10m(10:end-3);
elseif strcmp(ExType,'H6')
    wind_speed10m = wind_speed10m(10:end-6);
elseif strcmp(ExType,'H12')
    wind_speed10m = wind_speed10m(10:end-12);
elseif strcmp(ExType,'H18')
    wind_speed10m = wind_speed10m(10:end-18);
elseif strcmp(ExType,'H1_18')
    wind_speed10m = wind_speed10m(10:end-18);
    wind_speed10m = repmat(wind_speed10m,1,5);
end

%% Testing
data = importdata(['../CV_Partition_Test/Norm_Test_data_lag10_' ExType '.mat']);
Ts = data.YTest;
wind_speed10m = (wind_speed10m-data.YmuTest)./data.YsigTest;

figure;plot(Ts);hold on;
plot(wind_speed10m); title('feedforwardnet - Test')
%% performance on Testing data
truth = wind_speed10m;
pred = Ts;
MSE  = mean(mean((truth - pred).^2));   % Mean Squared Error
RMSE = mean(MSE.^0.5);
MAPE = mean(mean(abs(truth-pred)./abs(truth)))*100;
MAE  = mean(mean(abs(truth-pred)));

str = sprintf('Testing Results    ---> MAE =%0.4f, MSE =%0.4f, RMSE =%0.4f, MAPE =%0.4f',...
    MAE,MSE,RMSE,MAPE);
disp(str);