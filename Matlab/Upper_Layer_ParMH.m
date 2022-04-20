function [x_tot,x,x_t]=Upper_Layer_ParMH(N,T,sigprop,typeTar)
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %% LAIS - upper level               %%%%%%%
% %% parallel MH chains               %%%%%%%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % L. Martino, V. Elvira, D. Luengo, J. Corander, 
% %"Layered Adaptive Importance Sampling", 
% % Statistics and Computing, 2016. doi:10.1007/s11222-016-9642-5
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % N= number of chain
% % sigprop = std of the proposal pdfs of the vertical chains
% % x_tot= generated samples (states of the chains)
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if N<2
    N=2;
end 
if T<1
    T=1;
end 
   disp('-----------------------------------------------------------------------------------------')
   disp('****-****-****-****-****-****-****-****')
   disp('****       UPPER-LAYER            *****')
   disp('****-****-****-****-****-****-****-****')
   disp('-----------------------------------------------------------------------------------------')
    disp(['Number of chains= ' num2str(N), ' '])
    disp(['Iterations= ' num2str(T)])
    disp(['Total number of generated mean parameters= ' num2str(N*T)])
    
   disp('-----------------------------------------------------------------------------------------')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% TARGET %%%%%%%%%%%%%%%%%%%%%
 %%%%typeTar: type of the  target distribution %%% change target.m
[nothing,nothing,DIM]=target(NaN,typeTar); %%%%%% DIM= dimension 
logf=@(x) target(x,typeTar);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% parameters %%%%%%%%%%%%%%%%%%%%%
initialPoints=-4+8*rand(N,DIM);
x{1}=initialPoints;
SIGMA=sigprop.^2*eye(DIM);
x_tot=[];
 Vbef=logf(x{1});  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%% START %%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 %%%%%%%%%%%%% PARALLEL  CHAINS                   %%%%%%%%%%%%%
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
  for t=2:T+1    
     x{t}=mvnrnd(x{t-1},SIGMA);
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   
  %%% evaluating target %%%%%%%%%%%%
   %%%% Vbef=logf(x{t-1}); %%% THIS EVALUATION CAN BE AVOIDED, storing the previous one
    Vnow=logf(x{t});
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   
    [d1,d2]=size(Vbef);
     if d1>d2
         Vbef=Vbef';
         Vnow=Vnow';
     end
    rho=exp(Vnow-Vbef);
    alpha=min([ones(1,N);rho]);
    %%% Test %%%
    u=rand(1,N);
    test=(u<=alpha);
    vec_aux=(x{t}-x{t-1}).*repmat(test',1,DIM);
    vec_aux2=(Vnow-Vbef).*test;
    x{t}=x{t-1}+vec_aux;
    Vbef=Vbef+vec_aux2; 
    Vbef=Vbef';
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      x_tot=[x_tot x{t}'];
  end %%% end for - t
 x{1}=[];
 x=x(~cellfun('isempty',x));  
    for i=1:N
     x_t{i}=x_tot(:,i:N:end);
    end
       
       






