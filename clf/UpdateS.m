function [Lm,S] = UpdateS(L,Lm,betag,XW,XGW,param)
   [n,k] = size(Lm);
  
   manifold = elliptopefactory(n,k);   
   problem.M = manifold;
    
    problem.cost = @(x) LCost(L,x,betag,XW,XGW,param);
    problem.grad = @(x) LGrad(Lm,XGW,param);

    options = param.tooloptions;
    [x,~,~] = steepestdescent(problem,Lm,options);  
    Lm = x;   
    S = Lm*Lm';  
end

function cost = LCost(L,Lm,betag, XW,XGW,param)
    lambda1=param.lambda1;
    lambda2=param.lambda2;
    cost = 0.5*lambda2*betag*trace(XW'*L*XW)+0.5*lambda1*trace(XGW'*Lm*(Lm')*XGW);
end
function grad = LGrad(Lm,XGW,param)
    lambda1=param.lambda1;
    grad = lambda1*XGW*(XGW')*Lm;
end