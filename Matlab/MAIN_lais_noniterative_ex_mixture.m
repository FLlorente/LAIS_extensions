
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%   NON-ITERATIVE IMPLEMENTATION OF               %%%%%%%
%%%%%%%   LAYERED ADAPTIVE IMPORTANCE SAMPLING (LAIS)   %%%%%%%
%%%%%%%      (see reference below)                      %%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% L. Martino, V. Elvira, D. Luengo, J. Corander, 
%%%"Layered Adaptive Importance Sampling", Statistics and Computing, 2016. 
%%% doi:10.1007/s11222-016-9642-5
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all
close all
clc
help MAIN
%%%
typeTar=1;%%% typeTar: type of the  target distribution 
          %%% typeTar=1,2,3 =>DIM=2 
          %%% typeTar=4 =>DIM=4
          %%% typeTar=5 =>DIM=10    
          %%% Build your inference problem, changing target.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% upper layer: parallel MH chains  %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% number of parallel chains
N=100; %%% N>=1
%%%%  number of vertical and horizontal steps per epoch
T=20; %%% T>=1
sig_prop=5; %%std of the proposal pdfs of upper layer
[mu_tot,mu_sp,mu_time]=Upper_Layer_ParMH(N,T,sig_prop,typeTar);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% lower layer: MIS schemes         %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
M=100; %%% M>=1 %%% samples per proposal pdfs in the lower layer
%sig_lower_layer=3+7*rand(1,N*T);%% possible random selection of the scalar parameters 
sig_lower_layer=13*ones(1,N*T);
%%%%%%%%%%
%%% SUGGESTION: USE typeDEN=3 as in PI-MAIS (see article)
typeDEN=3; %%%% type of the MIS scheme
%%%% 1 - Standard IS
%%%% 2 - Full DM
%%%% 3 - Partial DM - spatial (suggested)
if typeDEN==1   
  disp(' ')
  disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% ')
  disp('We suggest to use the Partial DM approach (typeDEN=3)')
  disp('and to avoid typeDEN=1, specially for the estimation of the marginal likelihood')
  disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% ')
  disp(' ')
  pause(1)
end
[x_est,MarginalLike,x_IS,W]=Lower_Layer_IS(mu_tot,mu_sp,mu_time,N,T,M,sig_lower_layer,typeDEN,typeTar);
