function f
phi=10000*sind(180/10000);
pho=zeros(300,8);
[numofobs,numofcorr]=size(pho);
ran=rand(10,1);
for i=1:numofobs
    pho(i,1)=0;
    pho(i,2)=0.9;
    pho(i,3)=1.8*(1/(1+exp(-(i-numofobs/2)/40))-0.5);
    pho(i,4)=0.9*cos(2*phi*i/50);
    pho(i,5)=0.5*sign(sin(i*phi/10));
    pho(i,6)=1.8*(mod(i,20)/19-0.5);
    pho(i,7)=0.9*exp(-mod((i-1),50)/20)*cos(2*phi*i/5);
    pho(i,8)=0.9*ran(floor((i-1)/30)+1)*sign(sin(i*phi/10));
end
for i=1:numofcorr
    subplot(4,2,i)
    plot(pho(:,i),'-o','LineWidth',2)
    ylim([-1 1])
    xlim([1 numofobs])
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
e=cell(50,numofcorr);
[repeattimes,numofcorr]=size(e);
epislon=cell(repeattimes,numofcorr);Ht=cell(repeattimes,numofcorr);
simulatedata=cell(repeattimes,numofcorr);
VARlag1=[5/8,0.5;0.25,5/8];VARconst=[0.01;0.5];
for j=1:numofcorr
    for i=1:repeattimes
        e{i,j}=zeros(numofobs,2);epislon{i,j}=zeros(numofobs,2);Ht{i,j}=zeros(numofobs,3);
        simulatedata{i,j}=zeros(numofobs,4);
        for k=1:numofobs
            SIGMA=[1,pho(k,j);pho(k,j),1];
            e{i,j}(k,:)=mvnrnd([0,0],SIGMA,1);
        end
        [epislon{i,j}(:,1),Ht{i,j}(:,1)]=tarch_simulate(e{i,j}(:,1),[0.01;0.05;0.9],1,0,1);
        [epislon{i,j}(:,2),Ht{i,j}(:,3)]=tarch_simulate(e{i,j}(:,2),[0.5;0.2;0.5],1,0,1);
        [simulatedata{i,j}(:,1)]=armaxfilter_simulate(epislon{i,j}(:,1),0,1,0.9);
        [simulatedata{i,j}(:,2)]=armaxfilter_simulate(epislon{i,j}(:,2),0,1,0.7);
        simulatedata{i,j}(1,3:end)=(eye(2)-VARlag1)\VARconst;
        for k=2:numofobs
            simulatedata{i,j}(k,3:end)=transpose(VARconst+VARlag1*transpose(simulatedata{i,j}(k-1,3:end))+transpose(epislon{i,j}(k,:)));
        end
        for k=1:numofobs
            Ht{i,j}(k,2)=pho(k,j)*sqrt(Ht{i,j}(k,1)*Ht{i,j}(k,3));
        end
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
windowlength=285;
esdata=cell(repeattimes,numofcorr);% compare simple mean, univariate ar and var
esHt=cell(repeattimes,numofcorr);% compare cov risk metrics bekk and dcc
ARnumofcompare=6;
GARCHnumofcompare=(ARnumofcompare)*4*3;
ARlag=1;
for j=1:numofcorr
    for i=1:repeattimes
        esHt{i,j}=zeros(numofobs,GARCHnumofcompare);
        esdata{i,j}=zeros(numofobs,2*ARnumofcompare);
        errors=zeros(windowlength,2*ARnumofcompare);% the first 2 columns of errors are errors of simple mean
        % 3rd to 4th columns are univerate AR 5th to the end are VAR
        for o=1:2
            for k=1:(numofobs-windowlength)
                esdata{i,j}(k+windowlength,1+6*(o-1):2+6*(o-1))=mean(simulatedata{i,j}(k:(k+windowlength-1),(1+2*(o-1)):(2*o)));
                errors(:,1+6*(o-1):2+6*(o-1))=simulatedata{i,j}(k:(k+windowlength-1),(1+2*(o-1)):(2*o))...
                    -ones(windowlength,2)*diag(esdata{i,j}(k+windowlength,(1+2*(o-1)):(2*o)));
                %%%%%%%%%%%%%%%%%%%%%%
                for l=1:2
                    [ARparameters, ~, errors(:,2+l+6*(o-1))]=armaxfilter(simulatedata{i,j}(k:(k+windowlength-1),l+2*(o-1)),1,ARlag);
                    esdata{i,j}(k+windowlength,2+l+6*(o-1))=[1,simulatedata{i,j}((k+windowlength-ARlag):(k+windowlength-1),l+2*(o-1))]*ARparameters;  
                end
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                [VARparameters,~,~,~,VARconstent,~,~,errors(2:end,5+6*(o-1):6+6*(o-1))]=vectorar(simulatedata{i,j}(k:(k+windowlength-1),(1+2*(o-1)):(2*o)),1,ARlag);
                esdata{i,j}(k+windowlength,5+6*(o-1):6+6*(o-1))=simulatedata{i,j}((k+windowlength-ARlag):(k+windowlength-1),(1+2*(o-1)):(2*o))...
                    *transpose(VARparameters{ARlag})+transpose(VARconstent);
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                for l=1:(ARnumofcompare/2)
                    cvHt=cov(errors(:,(1+2*(l-1)+6*(o-1)):(2*l+6*(o-1))));
                    esHt{i,j}(k+windowlength,(1+12*(l-1)+36*(o-1)):(3+12*(l-1)+36*(o-1)))=transpose(cvHt(tril(cvHt)~=0));
                    %%%%%%%%%%%%%%%
                    riskHt=riskmetrics(errors(:,(1+2*(l-1)+6*(o-1)):(2*l+6*(o-1))),0.94);
                    riskHt=riskHt(:,:,end);
                    esHt{i,j}(k+windowlength,(4+12*(l-1)+36*(o-1)):(6+12*(l-1)+36*(o-1)))=transpose(riskHt(tril(riskHt)~=0));
                    %%%%%%%%%%%%%%%%%%%
                    [~, ~, bekkHt] = bekk(errors(:,(1+2*(l-1)+6*(o-1)):(2*l+6*(o-1))),[],1,0,1,'Scalar');
                    bekkHt=bekkHt(:,:,end);
                    esHt{i,j}(k+windowlength,(7+12*(l-1)+36*(o-1)):(9+12*(l-1)+36*(o-1)))=transpose(bekkHt(tril(bekkHt)~=0));
                    %%%%%%%%%%%%%%%%
                    [~, ~, dccHt]=dcc(errors(:,(1+2*(l-1)+6*(o-1)):(2*l+6*(o-1))),[],1,0,1,1,0,1);
                    dccHt=dccHt(:,:,end);
                    esHt{i,j}(k+windowlength,(10+12*(l-1)+36*(o-1)):(12*l)+36*(o-1))=transpose(dccHt(tril(dccHt)~=0));
                end
            end
        end
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%The first 3 columns of a certern cell in esHt are simple mean combined
%with simple COV, the next 3 contain simple mean combined with riskmetrics
%from the 13th column univerate AR combined with the corresponding method
%to estimate covariance matrix
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
maeHt=cell(repeattimes,numofcorr);
for j=1:numofcorr
    for i=1:repeattimes
        maeHt{i,j}=zeros(numofobs-windowlength,GARCHnumofcompare);
        for k=1:(GARCHnumofcompare/3)
            maeHt{i,j}(:,(1+3*(k-1)):3*k)=abs(esHt{i,j}((1+windowlength):end,(1+3*(k-1)):3*k)...
                -Ht{i,j}((1+windowlength):end,:));
        end
    end
end
smaeHt=cell(repeattimes,numofcorr);summaeHt=cell(1,numofcorr);
for j=1:numofcorr
    for i=1:repeattimes
        smaeHt{i,j}=sum(maeHt{i,j});
    end
    summaeHt{j}=zeros(1,GARCHnumofcompare);
    for i=1:repeattimes
        summaeHt{j}=summaeHt{j}+smaeHt{i,j};
    end
end
for i=1:8
    summaeHt{i}=summaeHt{i}./(repeattimes*(numofobs-windowlength));
end
SUMHt=zeros(numofcorr,GARCHnumofcompare/3);
for i=1:numofcorr
    for j=1:(GARCHnumofcompare/3)
        SUMHt(i,j)=sum(summaeHt{i}((1+3*(j-1)):(3*j)));
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
plot(SUMHt(:,5:GARCHnumofcompare/6)')
set(gca,'XTick',1:GARCHnumofcompare/6-4)
set(gca,'XTickLabel',{'ARMA-cov', 'ARMA-riskmetric', 'ARMA-bekk', 'ARMA-dcc', 'VAR-cov',...
    'VAR-riskmetric', 'VAR-bekk', 'VAR-dcc'})
title('ARMA')
legend('uncorrelated','highly correlated','logistic function','cos','step','modulous','exponential cos','random step')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
plot(SUMHt(:,GARCHnumofcompare/6+5:end)')
set(gca,'XTick',1:GARCHnumofcompare/6-4)
set(gca,'XTickLabel',{'ARMA-cov', 'ARMA-riskmetric', 'ARMA-bekk', 'ARMA-dcc', 'VAR-cov',...
    'VAR-riskmetric', 'VAR-bekk', 'VAR-dcc'})
title('VAR')
legend('uncorrelated','highly correlated','logistic function','cos','step','modulous','exponential cos','random step')
%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% THE END OF THE SIMULATED DATA
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
