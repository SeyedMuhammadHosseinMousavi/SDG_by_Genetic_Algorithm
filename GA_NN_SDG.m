%% Synthetic Data Generation by Genetic Algorithm (GA)
clc;
clear;
close all;
warning('off');
%% Loading The Dataset
load fisheriris.mat;
Target(1:50)=1;Target(51:100)=2;Target(101:150)=3;Target=Target'; % Original labels

%% Value of new samples
SyntheticGen=7; % 2 means generate synthetic samples, double the original samples.
% GA Parameters
VarLow = -5; % Lower bound of vars
VarHigh = 10; % Upper bound of vars

PopSize = 8; % Population size
MaxGenerations = 18; % Iterations
% Making dataset ready
Data=reshape(meas,1,[]); % Preprocessing - convert matrix to vector
Data=Data';
X=Data;
Y = round(Data)*rand;
DataNum = size(X,1);
InputNum = size(X,2);
OutputNum = size(Y,2);
% Normalization
MinX = min(X); MaxX = max(X);
MinY = min(Y); MaxY = max(Y);
XN = X; YN = Y;
for ii = 1:InputNum
XN(:,ii) = Normalize_Fcn(X(:,ii),MinX(ii),MaxX(ii));end
for ii = 1:OutputNum
YN(:,ii) = Normalize_Fcn(Y(:,ii),MinY(ii),MaxY(ii))*rand;end

%% Training  
TrPercent = 100;
TrNum = round(DataNum * TrPercent / 100);
R = sort(randperm(DataNum));
trIndex = R(1 : TrNum);
Xtr = XN(trIndex,:);
Ytr = YN(trIndex,:);
% Network Structure
pr = [-1 1];
PR = repmat(pr,InputNum,1);
Network = newff(PR,[50 OutputNum],{'tansig' 'tansig'});

%% Train GA
for i = 1:SyntheticGen
Network = TrainUsing_GA_Fcn(Network,Xtr,Ytr,VarLow,VarHigh,PopSize,MaxGenerations);
% Validation
YtrNet = sim(Network,Xtr')';
MSEtr = mse(YtrNet - Ytr);
% Create final matrix
Synthetic{i}=abs(YtrNet.*Data);
disp([' Generating Pack Number "',num2str(i)]);
end
FinalMSE = mse(Data,Synthetic{i})
% Converting cell to matrix (the last time)
Synthetic2 = cell2mat(Synthetic);
% Converting matrix to cell
P = size(Data); P = P (1,1);
S = size(Synthetic{i}); SO = size (meas); SF = SO (1,2); SO = SO (1,1); SS = S (1,2); 
for i = 1 : SyntheticGen
Generated1{i}=reshape(Synthetic2(:,i),[SO,SF]);
Generated1{i}(:,end+1)=Target; end
% Converting cell to matrix (the last time)
Synthetic3 = cell2mat(Generated1');
SyntheticData=Synthetic3(:,1:end-1);
SyntheticLbl=Synthetic3(:,end);

%% Plot data and classes
Feature1=2;
Feature2=3;
f1=meas(:,Feature1); % feature1
f2=meas(:,Feature2); % feature 2
ff1=SyntheticData(:,Feature1); % feature1
ff2=SyntheticData(:,Feature2); % feature 2
figure('units','normalized','outerposition',[0 0 1 1])
subplot(2,2,1)
plot(meas, 'linewidth',1); title('Original Data');
ax = gca; ax.FontSize = 12; ax.FontWeight='bold'; grid on;
subplot(2,2,2)
plot(SyntheticData, 'linewidth',1); title('Synthetic Data');
ax = gca; ax.FontSize = 12; ax.FontWeight='bold'; grid on;
subplot(2,2,3)
gscatter(f1,f2,Target,'rkgb','.',20); title('Original');
ax = gca; ax.FontSize = 12; ax.FontWeight='bold'; grid on;
subplot(2,2,4)
gscatter(ff1,ff2,SyntheticLbl,'rkgb','.',20); title('Synthetic');
ax = gca; ax.FontSize = 12; ax.FontWeight='bold'; grid on;

%% Train and Test
% Training Synthetic dataset by SVM
Mdlsvm  = fitcecoc(SyntheticData,SyntheticLbl); CVMdlsvm = crossval(Mdlsvm); 
SVMError = kfoldLoss(CVMdlsvm); SVMAccAugTrain = (1 - SVMError)*100;
% Predict new samples (the whole original dataset)
[label5,score5,cost5] = predict(Mdlsvm,meas);
% Test error and accuracy calculations
sizlbl=size(Target); sizlbl=sizlbl(1,1);
countersvm=0; % Misclassifications places
misindexsvm=0; % Misclassifications indexes
for i=1:sizlbl
if Target(i)~=label5(i)
misindex(i)=i; countersvm=countersvm+1; end; end
% Testing the accuracy
TestErrAugsvm = countersvm*100/sizlbl; SVMAccAugTest = 100 - TestErrAugsvm;
% Result SVM
AugResSVM = [' Synthetic Train SVM "',num2str(SVMAccAugTrain),'" Test on Original Dataset"', num2str(SVMAccAugTest),'"'];
disp(AugResSVM);

