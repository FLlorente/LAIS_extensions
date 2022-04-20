function samples = fun_lowerSampling(chains, qrnd, M)
% Draw M samples from qrnd's located on the vectors contained in
% chains

[T,DIM, N] = size(chains);
samples = cell(T,N);

for n = 1 : N
    disp(['Sampling n ',num2str(n),'-th chain'])
    for t = 1 : T
        samples_aux = zeros(M,DIM);
        for m = 1 : M
            samples_aux(m,:) = qrnd(chains(t,:,n));           
        end
        samples{t,n} = samples_aux;
    end
end

end