%%
clc
clear all
close all

% Seed

c=clock;
rand('seed',sum(c(1:end)));
randn('seed',sum(c(1:end)));

% Algorithm parameters

ap.N = 100; % number of proposals
ap.K = 10; % samples per proposal per iteration
ap.sig_prop = 1; % sigma or proposals
ap.E = 200000; % Target evaluations (total samples)
ap.T = ap.E/ap.K/ap.N; % number of iterations
ap.dm = 1; % deterministic mixture weights [0 1];
ap.lr = 0; % local resampling
ap.rs = 0; % random sigma

% Target parameters

tp.M = 2; % Dimension
tp.type = 8; % Type target


SIMU = 10;

for i = 1:SIMU
    i
   namefile=strcat('PMC_N_',num2str(ap.N),'_T_',num2str(ap.T),'_Sig_',num2str(ap.sig_prop),'_tipoTarget_',num2str(tp.type),'_detMixt_',num2str(ap.dm),'_K_',num2str(ap.K),'_lr_',num2str(ap.lr));
    
    [Est_evolution,Zest_evolution,all_samples] = DM_PMC_v1(ap,tp);
    
    clear Est_evolution_tot  Z_evolution_tot all_samples_tot
    
    if exist(strcat(namefile,'.mat'), 'file')==2
        load(namefile)
        
        Est_evolution_tot{end+1} = Est_evolution;
        Z_evolution_tot{end+1} = Zest_evolution;
    else
        Est_evolution_tot{1} = Est_evolution;
        Z_evolution_tot{1} = Zest_evolution;
    end
    
    save(namefile,'Est_evolution_tot','Z_evolution_tot')
end
