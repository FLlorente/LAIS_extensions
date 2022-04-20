function [chains,z,logEvals] = fun_genChainsmod(nIter, logTargets, starts, q, qrnd)
% Run MH algorithms for nIter iterations for each function 
% within logTargets. 
% It uses starting points within starts and all with the same proposal q.
% logTargets is an array with N cells such the n-th cell is the logtarget
% function for the n-th MH algorithm

% DEPENDS ON fun_MHmod.m

%  nChains = length(logTargets);
[nChains,DIM] = size(starts);
chains = zeros(nIter-1,DIM,nChains);
z = zeros(nIter-1,DIM,nChains);
logEvals = zeros(nIter-1,nChains);
for n = 1 : nChains 
    disp(['Running ',num2str(n),'-th MH'])
    [chains(:,:,n), z(:,:,n), logEvals(:,n)] = fun_MHmod(starts(n,:),nIter,logTargets{n},q,qrnd);
end
end