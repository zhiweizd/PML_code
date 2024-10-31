function [W,XW,XGWPool] = UpdateW(L,Q,Y,E,W,X,betav,XGPool,MgPool,C,param)
     [d,l] = size(W);
     manifold = euclideanfactory(d, l);  
     problem.M = manifold;
     problem.cost = @(x) Wcost(L,Y,E,x,X,betav,XGPool,MgPool,C,param);  
     problem.grad = @(x) Wgrad(Q,Y,E,x,X,betav,XGPool,MgPool,C,param);  
     options = param.tooloptions;
     [x,~,~] = steepestdescent(problem,W,options);   
     W = x;       
     XW = X*W;   
     XGWPool = cell(size(XGPool));   
     for i=1:length(XGPool)
         XGWPool{i} = XGPool{i}*x;   
     end
end

function cost = Wcost(L,Y,E,x,X,betav,XGPool,MgPool,C,param)
  lambda = param.lambda;
  lambda1 = param.lambda1;
  lambda2 = param.lambda2;
  lambdaw = param.lambda4;
  
  cost = lambda*norm(Y-E-X*x)^2 + lambdaw*trace(x*pinv(C)*x');  
  XW = X*x;   
  for i=1:length(MgPool)
      Mg = MgPool{i};    
      XGW = XGPool{i}*x;    
      cost = cost + lambda2*betav(i)*trace(XW'*L*XW) + lambda1*trace(XGW'*Mg*XGW);   
  end
end
function grad = Wgrad(Q,Y,E,x,X,betav,XGPool,MgPool,C,param)
  lambda = param.lambda;
  lambda1 = param.lambda1;
  lambda2 = param.lambda2;
  lambdaw = param.lambda4;

  grad = lambda*(X'*X * x - X'*(Y-E))+ lambdaw*(x*pinv(C)+x*(pinv(C))');
  for i=1:length(MgPool)
      Mg = MgPool{i};   
      Xg = XGPool{i};
      grad = grad + lambda2*betav(i)*(Q*x+Q'*x) + lambda1*((Xg')*Mg*Xg*x+(Xg'*Mg*Xg)'*x);
                                             
  end
end
