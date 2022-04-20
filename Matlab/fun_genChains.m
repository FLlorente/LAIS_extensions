function chains = fun_genChains(nIter, logTargets, starts, q, qrnd)
% Run MH algorithms for nIter iterations for each function 
% within logTargets. 
% It uses starting points within starts and all with the same proposal q.
% logTargets is an array with N cells such the n-th cell is the logtarget
% function for the n-th MH algorithm

% DEPENDS ON fun_MH.m

nChains = length(logTargets);
for n = 1 : nChains 
    disp(['Running ',num2str(n),'-th MH'])
    chains(:,:,n) = fun_MH(starts(n,:),nIter,logTargets{n},q,qrnd);
end
end