% function [] =main(dataset)
% load(dataset)
load(['ml_yeast.mat' ]);
param = importdata('param.mat');


param.tooloptions.maxiter = 30;
param.tooloptions.gradnorm = 1e-3;
param.tooloptions.stopfun = @mystopfun;     


[num_examples,~] = size(data);
train_ratio = 0.7;
num_train = round(num_examples * train_ratio);
num_round = 5;
results = zeros(num_round,5);

j=3;
s = RandStream.create('mt19937ar','seed',1);  
RandStream.setGlobalStream(s);

for i = 1 : num_round
    indexperm = randperm(num_examples);
    train_index = indexperm(1,1:num_train);
    test_index = indexperm(1,num_train + 1 : end);
    X1 = data(train_index,:);
    X2 = data(test_index,:);
    Y0 = target(:,train_index);
    Y1 = partial_labels(:,train_index);
    Y2 = target(:,test_index);
    [J] = genObv( Y1', 0.1*j);
    Y1=Y1';
    [W] = PML_LCTrain(J,Y1,X1,param);
    [HL,RL,OError,Coverage,AP] = evalt(W,X2,Y2);
    results(i,:) = [HL,RL,OError,Coverage,AP];
end
AVG = mean(results);
standard_deviation = std(results);
% evalstr = ['save results_new',filesep,dataset,'PML-LC.mat results AVG standard_deviation'];
% eval(evalstr);

function stopnow = mystopfun(~, ~, info, last)
    if last < 5 
        stopnow = 0;
        return;
    end
    flag = 1;
    for i = 1:3
        flag = flag & abs(info(last-i).cost-info(last-i-1).cost) < 1e-5;
    end
    stopnow = flag;
end