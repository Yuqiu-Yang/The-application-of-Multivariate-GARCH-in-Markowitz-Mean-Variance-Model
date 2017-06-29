function g
% Using real data to compare these model
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
zzhangye=xlsread('D:\Shanghai University of Finance and Economics\DISSERTATION\DATA\finaldata\hangyezz.xlsx','B3:K1526');
rev=diff(zzhangye);
[zzr,zzc]=size(rev);
stationaryp=zeros(1,zzc);
for i=1:zzc
    [~,stationaryp(i)]=augdf(rev(:,i),1,5);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
windowlength=zzr-60;
ARlag=floor(log(zzr-2));
maerev=cell(zzr,4);
Aeq=ones(2,zzc);
for k=1:(zzr-windowlength)
    res=zeros(windowlength,2*zzc);  %contain univariate AR and VAR
    ARexpectation=zeros(1,zzc);
    for i=1:zzc
        besterrors=zeros(windowlength,1);bestAIC=inf;bestq=0;bestp=0;
        for q=0:1%MA
            for p=1:2%AR
                if q==0
                   [ARparameters,~,ARerrors, ~,diagnostics]=armaxfilter(rev(k:(k+windowlength-1),i),1,[1:p],0);
                else              
                   [ARparameters,~,ARerrors, ~,diagnostics]=armaxfilter(rev(k:(k+windowlength-1),i),1,[1:p],[1:q]); 
                end
                [~,lmpv]=lmtest1(ARerrors,ARlag);
                ARh=zeros(ARlag,1);
                for j=1:ARlag
                    if(lmpv(j)<0.05)
                        ARh(j)=1;
                    end
                end
                if(1<=sum(ARh))
                    if(bestAIC>diagnostics.AIC)
                        bestAIC=diagnostics.AIC;
                        bestARparameters=ARparameters;
                        besterrors=ARerrors;
                        bestq=q;bestp=p;
                    end
                end
            end
        end
        res(:,i)=besterrors;   
        ARexpectation(i)=[1,transpose(rev((k+windowlength-bestq-bestp):(k+windowlength-1),i))]*bestARparameters;
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    VARAIC=inf;
    for lags =1:3
        [VARparameters,~,~,~,VARconst,~,~,VARerrors,s2]=vectorar(rev(k:(k+windowlength-1),:),1,lags);
        AIC1=log(det(s2))+2*lags*zzc*zzc/windowlength;
        VARh=multiportest(VARerrors,s2,lags);
        if(1<=sum(VARh))
            if(AIC1<VARAIC)
                step=lags;
                extpara=VARparameters;
                timeconst=VARconst;
                res(2:end,(zzc+1):end)=VARerrors;
                VARAIC=AIC1;
            end
        end
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    VARexpectation=timeconst;
    for j=1:step
        VARexpectation=VARexpectation+extpara{j}*transpose(rev(k+windowlength-j,:));
    end
    VARexpectation=transpose(VARexpectation);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    minARexp=min(ARexpectation);
    maxARexp=max(ARexpectation);
    rAR=[(maxARexp-minARexp)*0.25+minARexp,(minARexp+maxARexp)/2,...
        (maxARexp-minARexp)*0.75+minARexp];
    minVARexp=min(VARexpectation);
    maxVARexp=max(VARexpectation);
    rVAR=[(maxVARexp-minVARexp)*0.25+minVARexp,(minVARexp+maxVARexp)/2,...
        (maxVARexp-minVARexp)*0.75+minVARexp];
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for l=1:2
        [~, ~, H] = bekk(res(:,(1+zzc*(l-1)):(zzc*l)),[],1,0,1,'Scalar');
        bekkHt=H(:,:,end);
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        if l==1
            maerev{k+windowlength,l}=zeros(1,length(rAR));
            Aeq(2,:)=ARexpectation;
            for m=1:length(rAR)
                x=quadprog(bekkHt,[],[],[],Aeq,[1;rAR(m)],zeros(1,zzc),[]);
                maerev{k+windowlength,l}(m)=abs(rev(k+windowlength,:)*x-rAR(m));
            end
        else
            maerev{k+windowlength,l}=zeros(1,length(rVAR));
            Aeq(2,:)=VARexpectation;
            for m=1:length(rVAR)
                x=quadprog(bekkHt,[],[],[],Aeq,[1;rVAR(m)],zeros(1,zzc),[]);
                maerev{k+windowlength,l}(m)=abs(rev(k+windowlength,:)*x-rVAR(m));
            end
        end
    end
    for l=1:2
        [~, ~,H]=dcc(res(:,(1+zzc*(l-1)):(zzc*l)),[],1,0,1,1,0,1);
        dccHt=H(:,:,end);
        if l==1
            maerev{k+windowlength,l+2}=zeros(1,length(rAR));
            Aeq(2,:)=ARexpectation;
            for m=1:length(rAR)
                x=quadprog(dccHt,[],[],[],Aeq,[1;rAR(m)],zeros(1,zzc),[]);
                maerev{k+windowlength,l+2}(m)=abs(rev(k+windowlength,:)*x-rAR(m));
            end
        else
            maerev{k+windowlength,l+2}=zeros(1,length(rVAR));
            Aeq(2,:)=VARexpectation;
            for m=1:length(rVAR)
                x=quadprog(dccHt,[],[],[],Aeq,[1;rVAR(m)],zeros(1,zzc),[]);
                maerev{k+windowlength,l+2}(m)=abs(rev(k+windowlength,:)*x-rVAR(m));
            end
        end
    end  
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
smaerev1=zeros(zzr,4);
summaerev1=zeros(1,4);
for j=1:4
    for i=(1+windowlength):zzr
        smaerev1(i,j)=sum(maerev{i,j});
    end
    summaerev1=sum(smaerev1);
end
summaerev1=summaerev1./180;
plot(summaerev)
set(gca,'XTick',1:4)
set(gca,'XTickLabel',{'ARMA-bekk','VAR-bekk','ARMA-dcc', 'VAR-dcc'})
plot(summaerev1)
set(gca,'XTick',1:4)
set(gca,'XTickLabel',{'ARMA-bekk','VAR-bekk','ARMA-dcc', 'VAR-dcc'})







