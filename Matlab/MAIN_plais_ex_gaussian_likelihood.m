clc 
clear all
close all

% Gaussian conjugate model: inference on the mean (one-dimensional example)


D = 50; % number of data points
y = 10 + randn(D,1); % data generated from std Gaussian

sig=1; % std dev of Gaussian likelihood (considered known)
loglike = @(data,x) -0.5*sum((data-x).^2)/sig;
sig_pr=10; % std dev of Gaussian prior
logprior = @(x) -0.5*x.^2/sig_pr;
logtarget = @(x) loglike(y,x) + logprior(x);

mu_post = (1/sig_pr^2+D/sig^2)^(-1)*(sum(y)/sig^2); % TRUE VALUE
sig_post = (1/sig_pr^2+D/sig^2)^(-1); % TRUE VALUE

% K Subsets of data randomly selected
N=10; % number of chains
K = D/N; % number of (randomly selected) data per subposterior
for n = 1 : N
    y_now = datasample(y,K, 'Replace',false);
  
    logTars{n} = @(x) loglike(y_now,x) + logprior(x); % PA-LAIS
%     logTars{n} = logtarget;  % std LAIS

end

aux=-4*sig_pr:0.1:4*sig_pr;
figure
semilogy(aux,exp(logtarget(aux)),'r--','linewidth',2)
hold on
for k = 1 : length(logTars)
%     figure
    semilogy(aux,exp(logTars{k}(aux)))
    hold on
% plot(aux,exp(logTars{k}(0:0.1:20)/max(exp(logTars{k}(aux)))))
% hold on
end
legend('full posterior','partial posteriors')


% 2. SET Gaussian pdfs for proposals within MH and for lower layer
% h=1; % standard devs of the Gaussian covariance matrix
h=1; DIM=1;

phi = @(x_new,x_old) mvnpdf(x_new, x_old, h^2*eye(DIM));
phirnd = @(x_old) mvnrnd(x_old, h^2*eye(DIM), 1);

% 3. SET starting points and number of iterations of MH
starts = randn(N,1);
T = 100;

% 4. CALL fun_genChains.m to produce location parameters of proposals
mu_LAIS = fun_genChains(T, logTars, starts, phi, phirnd);



% chi=2; % standard devs of the Gaussian covariance matrix
chi=1;
q = @(x_new,x_old) mvnpdf(x_new, x_old, chi^2*eye(DIM));
qrnd = @(x_old) mvnrnd(x_old, chi^2*eye(DIM), 1);


% 5. CALL fun_lowerSampling.m to produce samples in the lower layer
M=1; % nº of samples per proposal
samples = fun_lowerSampling(mu_LAIS, qrnd,M); % a different proposal could be used!

% 6. CALL fun_lowerWeighting.m to compute weights (using spatial mixture
% denominator)
denType = 2; % 1 = spatial; 2 = temporal; 3 = complete
w_IS = fun_lowerWeightingbis(samples, mu_LAIS, logtarget, q, denType); 



% 7. ESTIMATE Z and check error
Z_est = mean(cell2mat(w_IS), 'all')

mu_post 
mu_post_est_plais = sum(cat(1,samples{:}).*cat(1,w_IS{:})) / (M*N*T*Z_est)

error = abs(mu_post_est_plais-mu_post)/mu_post


