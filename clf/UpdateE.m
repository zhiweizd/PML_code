function [ E ] = UpdateE(Y,W,X,param)
  lambdaE = param.lambda5;
  Q=2;
  E1 = Y - X*W;
  E = max(E1 - lambdaE/ Q,0);
end