function [betav,XGPool,LgPool,param] = InitGroup(Y,X,T,param)
     [n,~] = size(Y);
     k2 = param.k2;
     gp = unique(T);   
     g = length(gp);
     param.g = g;     
     LgPool = cell(g,1);
     XGPool = cell(g,1);
     betav = ones(g,1);  
     for i=1:g
       ii = T==gp(i);   
       Xg = X(ii,:);    
       [N,~] = size(Xg);
       LgPool{i} = rand(N,k2);  
       XGPool{i,1} = Xg;
       betav(i) = betav(i)*sum(ii)/n;   
     end   
end