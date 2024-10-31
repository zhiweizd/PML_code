function [W] = PML_LCTrain(J,Y,X,param)
  [~,d] = size(X);   
  [n,l] = size(Y);   
  W = zeros(d,l);   
  E = zeros(n,l);
  C = eye(l);     
  o = sum(J(:));
  
   param.lambda = 1;
   param.lambda1 = param.lambda1*o/((n/param.g)^2); 
   param.lambda2 = param.lambda2*o/(n^2);  
   param.lambda4 = param.lambda4*o/(d*l); 
   param.lambda5 = 0.8;  

%   param.lambda = 1;   
%   param.lambda5 = 0.4;  
%   param.lambda4 = 0.4; 
%   param.lambda2 = 0.0001;  
%   param.lambda1 = 0.1;  
   % param.g = 15;

  [T, ~] = kmeans(X,param.g,'emptyaction','drop');

  [betav,XGPool,LgPool,param] = InitGroup(Y,X,T,param);  
  
  g = param.g;
  MgPool = cell(g,1);
  param.maxIter = 15;
  param.tooloptions.maxiter = 60;
  param.tooloptions.gradnorm = 1e-5;
  [ W ] = InitW(Y,W,X,E,C,param); 
  [ C ] = UpdateC(W);
  
  obj_old = [];
  last = 0;   
  XGWPool = cell(g,1);
  for i = 1:g
      XGWPool{i} = XGPool{i}*W;    
  end
  
  XW = X*W;
  tic;
  S = ConstructMST(X);
  D = diag(sum(S,2));
  L = D - S;
  Q = (X')*L*X;
  for i=1:param.maxIter 
      disp(i);  
      for gr=1:g
          XGW = XGWPool{gr};  
          [Lg] = UpdateS(L,LgPool{gr},betav(gr),XW,XGW,param);  
          LgPool{gr} = Lg;   
          MgPool{gr} = Lg*Lg';  
      end  
      [E] = UpdateE(Y,W,X,param);
      [W,XW,XGWPool] = UpdateW(L,Q,Y,E,W,X,betav,XGPool,MgPool,C,param);
      [C] = UpdateC(W);
      obj = 0.5*norm((Y-E-X*W),'fro')^2 ;
     
     
      disp(obj);
      last = last + 1;
      obj_old = [obj_old;obj];
       
      if last < 5
          continue;
      end
      stopnow = 1;
      for ii=1:3
         stopnow = stopnow & (abs(obj-obj_old(last-1-ii)) < 1e-6);
      end
      if stopnow
          break;
      end
  end

end

