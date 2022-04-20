function [x_est,MarginalLike,x_IS,W]=Lower_Layer_IS(mu,mu_sp,mu_time,N,T,M,sig_lower_layer,typeDEN,typeTar)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%% Lower Layer of LAIS %%%%%%%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % L. Martino, V. Elvira, D. Luengo, J. Corander, 
% %"Layered Adaptive Importance Sampling", 
% % Statistics and Computing, 2016. doi:10.1007/s11222-016-9642-5
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%
%%% typeDEN  1: 'StandIS' -- 2: 'FullDM'
%%%%% %%%%% %%%%% %%%%% %%%%% %%%%% %%%%% %%%%% %%%%% %%%%% %%%%% %%%%%
Np=N*T;
%%%%% %%%%% %%%%% %%%%% %%%%% %%%%% %%%%% %%%%% %%%%% %%%%% %%%%% %%%%%
disp('-----------------------------------------------------------------------------------------')
   disp('****-****-****-****-****-****-****-****')
   disp('****       LOWER-LAYER            *****')
   disp('****-****-****-****-****-****-****-****')
   disp('-----------------------------------------------------------------------------------------')
    disp(['Number of proposal pdfs used in MIS = ' num2str(Np) ' '])
    disp(['Number of samples per proposal= ' num2str(M)])
    disp(['Total number of used samples = ' num2str(Np*M)])
    switch typeDEN
        case 1
            aux='Standard IS';
            NumDen=1;
            NumProp=1;
        case 2
             aux='Full DM';
             NumDen=1;
             NumProp=N*T;
         case 3
             aux='Partial DM - Spatial';
             NumDen=T;
             NumProp=N;
    end
    disp(' ')
    disp(['Type IS-Denominator= ' aux])
    disp(['Number of different denominators = ' num2str(NumDen)])
    disp(['Number of proposal pdfs in each denominator = ' num2str(NumProp)])
     
   disp('-----------------------------------------------------------------------------------------')
   
   



[nothing,nothing,DIM,mu_true,Marglike_true]=target(NaN,typeTar); %%%%%% DIM= dimension 
%%%%  target distribution %%% change target.m
logf=@(x) target(x,typeTar);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% normalizing constant proposal ==> 1/(sqrt((2*pi).^DIM*det(SIGMA_p)))

count=1;
t=1;
n=1;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for j=1:Np
%%%% generated samples - lower layer %%%%%%%%%%%%
SIGMA_p=sig_lower_layer(j)*eye(DIM);
x_IS(:,M*(j-1)+1:M*j)=mvnrnd(mu(:,j),SIGMA_p,M)';

%%%%%% %%%%%% %%%%%% 
if typeDEN==1 %%% 'StandIS'
%%%%% %%%% %%%% %%%% %%%% %%%% %%%% %%%% %%%%  %%%%
%%%% evaluate j-th proposal - only own samples %%%%
%%%% %%%% %%%% %%%% %%%% %%%% %%%% %%%% %%%% %%  %%
P(M*(j-1)+1:M*j)=mvnpdf(x_IS(:,M*(j-1)+1:M*j)',mu(:,j)',SIGMA_p);
end

%%%%%% %%%%%% %%%%%% 
if typeDEN==3
   x_ISsp{t}(:,M*(count-1)+1:M*count)=x_IS(:,M*(j-1)+1:M*j); 
   count=count+1;
    if mod(j,N)==0
     t=t+1;
     count=1;
   end 
end

%%%%%% %%%%%% %%%%%%  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% evaluate target         %%%%
logNUM(M*(j-1)+1:M*j)=logf(x_IS(:,M*(j-1)+1:M*j)')';
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


switch  typeDEN
    %%%%%%%%%%%%%%%%%%%%%%  
    case 1 %%% 'StandIS'
      logDEN=log(P);
    %%%%%%%%%%%%%%%%%%%%%%  
    case 2 %%% 'FullDM' (evaluating 'All-in-All')
    %%%%%%%%%%%%%%%%%%%%%%    
        for j=1:Np  
            
             if mod(j,10)==0
                      disp(['computing IS denominators...',num2str(min([fix(j*100/Np) 100])) '%'])
             end
             
            for k=1:Np
               P(k,M*(j-1)+1:M*j)=mvnpdf(x_IS(:,M*(j-1)+1:M*j)',mu(:,k)',SIGMA_p);
               %%% P size N*T \times N*T*M
               %%% row => evaluation one proposal pdf at all the samples   
               %%% (row index=> denote one proposal pdf)
               %%% (column index=> denote one sample)
            end           
        end
          P=mean(P);
          logDEN=log(P); 
      %%%%%%%%%%%%%%%%%%%%%%   
       case 3 %%% 'Partial DM- Spatial'  
      %%%%%%%%%%%%%%%%%%%%%%  
          for t=1:T
                 if mod(t,10)==0
                      disp(['computing IS denominators...',num2str(min([fix(t*100/T) 100])) '%'])
                 end
               
              for n=1:N
                  Maux(n,:)=mvnpdf(x_ISsp{t}',mu_sp{t}(n,:),SIGMA_p); 
              end   
              a(t,:)=log(mean(Maux));
          end    
           logDEN= reshape(a',1,N*T*M);
     
     
end %%% build denominator
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%% WEIGHTING %%%%%%%%%%%
W=exp(logNUM-logDEN);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% ESTIMATIONS %%%%%%%%%
MarginalLike=mean(W); %%% estimation of the marginal likelihood
disp(' ')
disp('RESULTS')
disp(' ')
disp(['Marginal likelihood- true value = ', num2str(Marglike_true)])
disp(['Marginal likelihood- estimated value = ', num2str(MarginalLike)])
SEmargLike=(Marglike_true-MarginalLike)^2;
disp(['Square Error in the estimation of marginal likelihood = ', num2str(SEmargLike)])
%%%%%%%%%%%%
Wn=W./(Np*M*MarginalLike);
x_est=sum(repmat(Wn,DIM,1).*x_IS,2);
disp(' ')
disp(['Expected Value of the posterior/target pdf - True Values = ',num2str(mu_true')])
disp(['Expected Value of the  posterior/target pdf-Estimated Values= ',num2str(x_est')])
SE_est=mean((mu_true-x_est).^2);
disp(['Square Error in the estimation of the Expected Value = ', num2str(SE_est)])
%%%%%%%%%%%%%%
%%%% plot %%%%
if typeTar==1 | typeTar==2 | typeTar==3
 if typeTar==1    
    hgload('contourGauss5modes.fig');
 end
 hold on
  plot(x_IS(1,:),x_IS(2,:),'g.','MarkerEdgeColor','g','MarkerFaceColor','g','MarkerSize',1)
 plot(mu(1,:),mu(2,:),'rs','MarkerEdgeColor','k','MarkerFaceColor','r','MarkerSize',5)
  axis([-22 22 -25 25])  
end
