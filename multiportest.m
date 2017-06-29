function [h,pval]=multiportest(errors,s2,lags)
[T,col]=size(errors);
m=mean(errors);
mu=zeros(T,col);
for i=1:T
    mu(i,:)=m;
end
h=zeros(lags,1);pval=zeros(lags,1);
for i=1:lags
    Qm=0;
    for l=1:i
        errors1=errors((l+1):end,:)-mu(1:(T-l),:);errors2=errors(1:(T-l),:)-mu(1:(T-l),:);
        gammal=(1/(T-l))*(transpose(errors1)*errors2);
        Qm=Qm+(1/(T-l))*trace((transpose(gammal)/s2)*(gammal/s2));
    end
    Qm=T*T*Qm;
    degoffree=col*col*i;
    pval(i)=1-chi2cdf(Qm,degoffree);
    if(pval(i)<0.05)
        h(i)=1;
    end
end