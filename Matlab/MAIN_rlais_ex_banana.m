% Example of application of RLAIS for banana target
clc 
clear all
close all


% banana target in DIM=2
DIM=2;
B=10;etha1=4;etha2=3.5;etha3=3.5;
logtarget=@(x) -(4-B*x(1)-x(2).^2).^2./(2*etha1^2)-...
    (x(1).^2)./(2*etha2^2)-(x(2).^2)./(2*etha3^2);
% target=@(x) exp(logtarget(x(1),x(2)));
Z_TRUE = 7.9976; % true values

 
% 1. SET number of parallel chains = number of logTars to be passed to
% fun_genChains.m
N = 2; 
for n = 1 : N
    logTars{n} = logtarget; %all chains have the target as invariant distribution
end

% 2. SET Gaussian pdfs for proposals within MH and for lower layer
h=1; % standard devs of the Gaussian covariance matrix
phi = @(x_new,x_old) mvnpdf(x_new, x_old, h^2*eye(DIM));
phirnd = @(x_old) mvnrnd(x_old, h^2*eye(DIM), 1);

% 3. SET starting points and number of iterations of MH
starting_points = 10 - 20*rand(N,DIM);
T = 500;

% 4. CALL fun_genChains.m to produce location parameters of proposals and
% samples
[mu_LAIS,samples,logEvals] = fun_genChainsmod(T, logTars, starting_points, phi, phirnd);


% 5. CALL fun_lowerWeighting.m to compute weights 
denType = 2; % MIS denom: 1 = spatial; 2 = temporal; 3 = complete
w_IS = fun_lowerWeighting(samples, logEvals,mu_LAIS, phi, denType); % 


% 7. ESTIMATE Z and check error
Z_est_rlais = mean(w_IS, 'all')
error_rlais = abs(Z_est_rlais - Z_TRUE) / Z_TRUE



% END RLAIS 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% STANDARD IS WITH UNIFORM PROPOSAL FOR COMPARISON

for i = 1 : N*T
    z = 10 - 20*rand(1,DIM); % proposal uniform in [-10,10]^2
    w_unif(i) = exp(logtarget(z))*20^2; 
end

Z_unif = mean(w_unif)
error_unif = abs(Z_unif - Z_TRUE) / Z_TRUE

% END STANDARD IS



