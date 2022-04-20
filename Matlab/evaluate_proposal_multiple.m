function [fp_mix,fp_sin]=evaluate_proposal_multiple(x,mu,sig,N)
% Evaluate N samples in N Gaussian isotropic proposals

% x: samples (M x N)
for i=1:N % i-th sample
    z=repmat(x(:,i),1,N); 
    tmp =1*exp(-0.5 * ((z - mu)./sig).^2) ./ (sqrt(2*pi) .* sig); % i-th sample evaluated in M dimensions x N proposals
    fp1=prod(tmp,1); % All dimensions are multiplied (1 x N)
    fp_mix(i)=1/N*sum(fp1,2); % scalar value i-th sample 
    fp_sin(i)=prod(tmp(:,i),1);
end

