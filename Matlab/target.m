function [logtarget,target,DIM,mu_true,Marglike_true]=target(x,typeT)
%%%% target definition - posterior %%%%

switch typeT
    case 1
        
      if isnan(x)
         DIM=2;
         logtarget=[];
         target=[];
         mu_true=[1.6 1.4]';
         Marglike_true=1;
      else
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
      
       f1=1/5*mvnpdf(x,mu1,SIGMA1);
       f2=1/5*mvnpdf(x,mu2,SIGMA2);
       f3=1/5*mvnpdf(x,mu3,SIGMA3);
       f4=1/5*mvnpdf(x,mu4,SIGMA4);
       f5=1/5*mvnpdf(x,mu5,SIGMA5);
       f=f1+f2+f3+f4+f5;
       target=f';
       logtarget=log(f);
      end
  %%%%%%%%%    
  case 2
  %%%%%%%%%      
      if isnan(x)
         DIM=2;
         logtarget=[];
         target=[];
         mu_true=[0 16]';
         Marglike_true=1;
      else
           mu2=[0 16];
           SIGMA2 = [3 0;0 3];
           f=mvnpdf(x,mu2,SIGMA2);
           target=f';
           logtarget=log(f);
      end
  %%%%%%%%%    
  case 3
  %%%%%%%%%      
      if isnan(x)
         DIM=2;
         logtarget=[];
         target=[];
         mu_true=[2.5 8]';
         Marglike_true=1;
      else  
            mu1=[5 0];
            SIGMA1 = [2 0.6; 0.6 1];
           mu2=[0 16];
           SIGMA2 = [3 0;0 3];
           f1=1/2*mvnpdf(x,mu1,SIGMA1);
           f2=1/2*mvnpdf(x,mu2,SIGMA2);
           f=f1+f2;
           target=f';
           logtarget=log(f);
      end
  %%%%%%%%    
  case 4
  %%%%%%%%%      
      if isnan(x)
         DIM=4;
         logtarget=[];
         target=[];
         mu_true=[0 16 5 -5]';
         Marglike_true=1;
      else
           DIM=4;
           mu2=[0 16 5 -5];
           sig=4;
           SIGMA2 = sig*eye(DIM);
           f=mvnpdf(x,mu2,SIGMA2);
           target=f';
           logtarget=log(f);
      end  
 %%%%%%%%    
  case 5
  %%%%%%%%%      
      if isnan(x)
         DIM=10;
         logtarget=[];
         target=[];
         mu_true=5*ones(DIM,1);
         Marglike_true=1;
      else
           DIM=10;
           mu2=5*ones(1,DIM);
           sig=4;
           SIGMA2 = sig*eye(DIM);
           f=mvnpdf(x,mu2,SIGMA2);
           target=f';
           logtarget=log(f);
      end       
      
end

end
   