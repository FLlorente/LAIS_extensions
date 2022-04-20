function [f,f1,f2,f3,f4,f5,logf]=evaluate_target_Gaussian_2D(x,tipo)


if tipo==8
    mu1=[-10 -10];
    SIGMA1 = [2 0.6; 0.6 1];
    mu2=[0 16];
    SIGMA2 = [2 -0.4;-0.4 2];
    mu3=[13 8];
    SIGMA3 = [2 0.8;0.8 2];
    mu4=[-9 7];
    SIGMA4 = [3 0; 0 0.5];
    mu5=[14 -14];
    SIGMA5 = [2 -0.1;-0.1 2];
    
    f1=1/5*mvnpdf(x',mu1,SIGMA1);
    f2=1/5*mvnpdf(x',mu2,SIGMA2);
    f3=1/5*mvnpdf(x',mu3,SIGMA3);
    f4=1/5*mvnpdf(x',mu4,SIGMA4);
    f5=1/5*mvnpdf(x',mu5,SIGMA5);
    
    f=f1+f2+f3+f4+f5;
    f=f';
    logf=log(f);

end