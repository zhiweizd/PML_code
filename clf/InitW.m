function [W] = InitW(Y,W,X,E,C,param)
     [d,l] = size(W);  
     obj_old = [];
     last = 0;
     
     for i=1:param.maxIter
        W = upw(Y,E,W,X,C,param,l,d);  
        obj = 0.5*norm((Y-E-X*W),'fro')^2;
         disp(obj);  
         last = last + 1;
         obj_old = [obj_old;obj];
      
      
          if last < 5
             continue;
          end 
          stopnow = 1;
          for ii=1:3
             stopnow = stopnow & (abs(obj-obj_old(last-1-ii)) < 1e-3);
          end
          if stopnow
             break;
          end
     end

end
function W =upw(Y,E,W,X,C,param,l,d)
          manifold = euclideanfactory(d, l);  
          problem.M = manifold;
          problem.cost = @(x) Wcost(Y,E,W,X,C,param);
          problem.grad = @(x) Wgrad(Y,E,W,X,C,param);
          options = param.tooloptions;
          [x,~,~] = steepestdescent(problem,W,options); 
          W = x;    
end

function cost = Wcost(Y,E,W,X,C,param)
     
     lambda = param.lambda;
     lambdaw = param.lambda4;
          
     cost = 0.5*lambda*norm(Y-E-X*W,'fro')^2 + 0.5*lambdaw*trace(W*pinv(C)*W');
end

function grad = Wgrad(Y,E,W,X,C,param)  
     
     lambda = param.lambda;
     lambdaw = param.lambda4;
     grad = lambda*(X'*X * W - X'*(Y-E)) + lambdaw*(W*pinv(C)+W*(pinv(C))');
end

