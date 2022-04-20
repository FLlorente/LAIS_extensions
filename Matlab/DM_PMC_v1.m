function [Est_evolution,Z_evolution,all_samples] = DM_PMC_v1(ap,tp)

N = ap.N; % number of proposals
K =  ap.K; % samples per proposal per iteration
sig_prop = ap.sig_prop; % sigma or proposals
T = ap.T; % number of iterations
dm = ap.dm; % deterministic mixture weights [0 1];
lr = ap.lr; % local resampling
rs = ap.rs; % random sigma

M = tp.M;
type_tar = tp.type;

if type_tar == 8
    TrueValue(1,1)=mean([-10 0 13 -9 14]);
    TrueValue(2,1)=mean([-10 16 8 7 -14]);
end
%% Inizialization

% Mean of the proposals
if type_tar==12 %%% banana
    initial_means(1,:)=-6+3*rand(1,N);
    initial_means(2,:)=-4+8*rand(1,N);
    
elseif type_tar==8
    initial_means=-4+8*rand(M,N);
elseif type_tar==9
    initial_means=1+4*rand(M,N);
elseif type_tar ==50 || type_tar==51
    initial_means = 30*randn(M,N);
end

% Variance of the proposals
if rs == 1 % Random
    v2 = 1 + sig_prop*rand(M,N);
else % Fixed
    %     disp('fixed variance')
    v2 = sig_prop*ones(M,N);
end

% Intialized recursive estimators
Stot = 0;

%% Iterate

for i=1:T
    
    % 0. Update proposals
    
    if i == 1 % First iteration
        mu{1} = initial_means; % Initialized means
    else % Next iterations
        mu{i} = samples_resampled; % Proposal means are the resampled particles of previous iteration
    end
    
    % 1. Sampling (propagate proposals)
    
    samples{i} = kron(mu{i},ones(1,K)) + kron(v2,ones(1,K)) .* randn(M,N*K);
    
    if dm == 1 % If deterministic mixture
        [fp_mixt,~]=evaluate_proposal_multiple(samples{i},kron(mu{i},ones(1,K)),kron(v2,ones(1,K)),K*N);
        logP = log(fp_mixt + 10^(-300));
    else
        fp_sing = evaluate_proposal(samples{i},kron(mu{i},ones(1,K)),kron(v2,ones(1,K)));
        logP = log(fp_sing + 10^(-300));
    end
    
    if type_tar == 9
        [logf,~] = target(samples{i}.',type_tar); % Aqui no hace falta transponer
        logf = logf.';
    else
        [~,~,~,~,~,~,logf] = evaluate_target_Gaussian_2D(samples{i},type_tar);
        
    end
    
    % 2. Weighting
    w = exp(logf-logP) + 10^(-300); % Raw weights
    S1(i) = sum(w); % Sum of raw weights
    wn=w./S1(i); % Normalized weights
    
    if i == 1 % First iteration
        Z_evolution(i) = S1(i)/(K*N); % First Z estimator
    else % Next iterations
        Z_evolution(i) = ((i-1)*Z_evolution(i-1) + S1(i)/(K*N))/i; % Recursive Z estimator
    end
    
    wn_D=repmat(wn,M,1);
    Est_part(:,i)=sum(wn_D.*samples{i},2);
    
    
    if i == 1
        Est_evolution(:,i) = Est_part(:,i);
    else
        Est_evolution(:,i) = (Stot(i)*Est_evolution(:,i-1)+S1(i)*Est_part(:,i))/(Stot(i)+S1(i));
    end
    
    Stot(i+1) = Stot(i) + S1(i);
    
    % 3. Multinomial resampling
    
    if lr == 0 % Global resampling: N independent simulations with replacement from the set of NK samples
        pos = randsrc(1,N,[1:N*K; wn]);
    elseif lr == -1 % No resampling
        pos = 1:K:K*N;
    else % Local resampling: Exactly one sample per proposal survives with prob. proportional to its weight
        for n = 1:N
            proposal_indices = (n-1)*K+1:n*K;
            wn_n = (wn(proposal_indices) + 10^(-300))/sum(wn(proposal_indices) + 10^(-300)); %normalized weights for the n-th gaussian (length K)
            pos_n = randsrc(1,1,[proposal_indices;wn_n]); % select one of the K samples of the n-th gaussian
            pos(n) = pos_n;
        end
    end
    
    samples_resampled = samples{i}(:,pos); % Resampled population
    all_samples{i} = samples_resampled; % All samples are updated
end
