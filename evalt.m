function [ HL,RL,OError,Coverage,AP ] = evalt(W,X2,Y2)

Y2(Y2 == 0) = -1;
PV = X2*W;
PV1 = PV;
PV1(PV1 >= 0.5)= 1;
PV1(PV1 < 0.5) = -1;
HL = Hamming_loss(PV1', Y2);
RL = Ranking_loss(PV', Y2);
OError = One_error(PV',Y2);
Coverage = coverage(PV', Y2);
AP = Average_precision(PV', Y2);

%% Haming_loss
function HammingLoss=Hamming_loss(Pre_Labels,test_target)

    [num_class,num_instance]=size(Pre_Labels);
    miss_pairs=sum(sum(Pre_Labels~=test_target));
    HammingLoss=miss_pairs/(num_class*num_instance);

%% Ranking Loss
function RankingLoss=Ranking_loss(Outputs,test_target)

    [num_class,num_instance]=size(Outputs);
    temp_Outputs=[];
    temp_test_target=[];
    for i=1:num_instance
        temp=test_target(:,i);
        if((sum(temp)~=num_class)&(sum(temp)~=-num_class))
            temp_Outputs=[temp_Outputs,Outputs(:,i)];
            temp_test_target=[temp_test_target,temp];
        end
    end
    Outputs=temp_Outputs;
    test_target=temp_test_target;     
    [num_class,num_instance]=size(Outputs);
    
    Label=cell(num_instance,1);
    not_Label=cell(num_instance,1);
    Label_size=zeros(1,num_instance);
    for i=1:num_instance
        temp=test_target(:,i);
        Label_size(1,i)=sum(temp==ones(num_class,1));
        for j=1:num_class
            if(temp(j)==1)
                Label{i,1}=[Label{i,1},j];
            else
                not_Label{i,1}=[not_Label{i,1},j];
            end
        end
    end
    
    rankloss=0;
    for i=1:num_instance
        temp=0;
        for m=1:Label_size(i)
            for n=1:(num_class-Label_size(i))
                if(Outputs(Label{i,1}(m),i)<=Outputs(not_Label{i,1}(n),i))
                    temp=temp+1;
                end
            end
        end
        rl_binary(i)=temp./(m*n);
        rankloss=rankloss+temp./(m*n);
    end
    RankingLoss=rankloss./num_instance;

%% One_error
function OneError=One_error(Outputs,test_target)

    [num_class,num_instance]=size(Outputs);
    temp_Outputs=[];
    temp_test_target=[];
    for i=1:num_instance
        temp=test_target(:,i);
        if((sum(temp)~=num_class)&(sum(temp)~=0))
            temp_Outputs=[temp_Outputs,Outputs(:,i)];
            temp_test_target=[temp_test_target,temp];
        end
    end
    Outputs=temp_Outputs;
    test_target=temp_test_target;     
    [num_class,num_instance]=size(Outputs);
    
    Label=cell(num_instance,1);
    not_Label=cell(num_instance,1);
    Label_size=zeros(1,num_instance);
    for i=1:num_instance
        temp=test_target(:,i);
        Label_size(1,i)=sum(temp==ones(num_class,1));
        for j=1:num_class
            if(temp(j)==1)
                Label{i,1}=[Label{i,1},j];
            else
                not_Label{i,1}=[not_Label{i,1},j];
            end
        end
    end
    
    oneerr=0;
    for i=1:num_instance
        indicator=0;
        temp=Outputs(:,i);
        [maximum,index]=max(temp);
        for j=1:num_class
            if(temp(j)==maximum)                
                if(ismember(j,Label{i,1}))
                    indicator=1;
                    break;
                end
            end
        end
        if(indicator==0)
            oneerr=oneerr+1;
        end
    end
    OneError=oneerr/num_instance;

%% Coverage
function Coverage=coverage(Outputs,test_target)

       [num_class,num_instance]=size(Outputs);
    
       Label=cell(num_instance,1);
       not_Label=cell(num_instance,1);
       Label_size=zeros(1,num_instance);
       for i=1:num_instance
           temp=test_target(:,i);
           Label_size(1,i)=sum(temp==ones(num_class,1));
           for j=1:num_class
               if(temp(j)==1)
                   Label{i,1}=[Label{i,1},j];
               else
                   not_Label{i,1}=[not_Label{i,1},j];
               end
           end
       end

       cover=0;
       for i=1:num_instance
           temp=Outputs(:,i);
           [tempvalue,index]=sort(temp);
           temp_min=num_class+1;
           for m=1:Label_size(i)
               [tempvalue,loc]=ismember(Label{i,1}(m),index);
               if(loc<temp_min)
                   temp_min=loc;
               end
           end
           cover=cover+(num_class-temp_min+1);
       end
       Coverage=((cover./num_instance)-1)./num_class;

%% Average Precision
function Average_Precision=Average_precision(Outputs,test_target)

    [num_class,num_instance]=size(Outputs);
    temp_Outputs=[];
    temp_test_target=[];
    for i=1:num_instance
        temp=test_target(:,i);
        if((sum(temp)~=num_class)&(sum(temp)~=-num_class))
            temp_Outputs=[temp_Outputs,Outputs(:,i)];
            temp_test_target=[temp_test_target,temp];
        end
    end
    Outputs=temp_Outputs;
    test_target=temp_test_target;     
    [num_class,num_instance]=size(Outputs);
    
    Label=cell(num_instance,1);
    not_Label=cell(num_instance,1);
    Label_size=zeros(1,num_instance);
    for i=1:num_instance
        temp=test_target(:,i);
        Label_size(1,i)=sum(temp==ones(num_class,1));
        for j=1:num_class
            if(temp(j)==1)
                Label{i,1}=[Label{i,1},j];
            else
                not_Label{i,1}=[not_Label{i,1},j];
            end
        end
    end
    
    aveprec=0;
    for i=1:num_instance
        temp=Outputs(:,i);
        [tempvalue,index]=sort(temp);
        indicator=zeros(1,num_class);
        for m=1:Label_size(i)
            [tempvalue,loc]=ismember(Label{i,1}(m),index);
            indicator(1,loc)=1;
        end
        summary=0;
        for m=1:Label_size(i)
            [tempvalue,loc]=ismember(Label{i,1}(m),index);
            summary=summary+sum(indicator(loc:num_class))./(num_class-loc+1);
        end
        ap_binary(i)=summary./Label_size(i);
        aveprec=aveprec+summary./Label_size(i);
    end
    Average_Precision=aveprec./num_instance;
