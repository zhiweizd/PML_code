function [S] = ConstructMST(train_data)   %train_data:nxd

D = pdist2(train_data,train_data);  
[Tree,Dis] = MinSpanTree(D);  
[a,~] = size(train_data);
[b,~] = size(Tree);
W = zeros(a,a);
S = zeros(a,a);
for i=1:b
    W(Tree(i,1),Tree(i,2))=Dis(i,1);
end
W=W+W';
for i=1:a
    for j=1:a
       if i==j
           W(i,j)=0;
       end
       if i~=j&&W(i,j)==0
           W(i,j)=inf;
       end
    end
end

[D]=floyd(W);

for i=1:a
    for j=1:a
       if i==j
           S(i,j)=0;
       end
       if D(i,j)>0
           S(i,j)=1./D(i,j);
       end
    end
end

function [D]=floyd(a)   
D=a;n=size(D,1);path=zeros(n,n); 
for i=1:n
    for j=1:n
        if D(i,j)~=inf
            path(i,j)=j;
        end  
    end
end

for k=1:n
    for i=1:n
        for j=1:n
            if D(i,k)+D(k,j)<D(i,j)
                D(i,j)=D(i,k)+D(k,j);
                path(i,j)=path(i,k);
            end 
        end
    end
end
end


function [Tree,Dis]=MinSpanTree(adjMat,sPoint)
Mat=adjMat;        
Mat(Mat==0)=inf;   
if nargin<=1
    MinNum=min(min(Mat));
    [H,~]=find(Mat==MinNum);
    H=H(1);
else
    H=sPoint;          
                       
end
N=size(adjMat,1);  
Tree=zeros(N-1,2); 
Dis=zeros(N-1,1);  
for i=1:N-1
    Mat(:,H)=inf;                
    tempMat=Mat(H,:);            
    MinNum=min(min(tempMat));    
    [m,n]=find(tempMat==MinNum);
    Tree(i,:)=[H(m(1)),n(1)];    
    Dis(i)=adjMat(H(m(1)),n(1)); 
    H=[H,n(1)];                  
end 
end
end