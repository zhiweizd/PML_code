function [ C ] = UpdateC(W)

     o=W'*W;
     %W_sq = sqrtm(o); 
     %C = W_sq/trace(W_sq);
    [U, S, V] = svd(o);
    S_sqrt = sqrt(S);

    B = U * S_sqrt * V';
    %W_sq = real(sqrtm(o));  
    C = B/trace(B);
end