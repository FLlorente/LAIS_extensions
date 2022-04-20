function weights = fun_lowerWeighting(samples, chains, logtarget,q,denom)
% compute intermediate-cost IS weights for all samples using the proposal
% in qrnd using the locations parameters from chains

[T,~, N] = size(chains);
weights = cell(T,N);

for n = 1 : N
   disp(['Weighting ',num2str(n),'-th chain'])
    for  t = 1 : T
        samples_aux = samples{t,n};
        [M,~] = size(samples_aux);
        weights_aux = zeros(1,M);
        
        switch denom
            case 1 % spatial mixture
                AUX = zeros(M,N);
                for i = 1 : N % spatial mixture
                    AUX(:,i) = q(samples_aux,chains(t,:,i));
                end
                den = 1/N*sum(AUX,2);
                for m = 1 : M
                    weights_aux(m) = exp(logtarget(samples_aux(m,:))) / den(m);       
                end
                weights{t,n} = weights_aux;
            
            case 2 % temporal mixture
                AUX = zeros(M,T);
                for i = 1 : T 
                    AUX(:,i) = q(samples_aux,chains(i,:,n));
                end
                den = 1/T*sum(AUX,2);
                for m = 1 : M
                    weights_aux(m) = exp(logtarget(samples_aux(m,:))) / den(m);       
                end
                weights{t,n} = weights_aux;
                
            case 3 % complete mixture
                AUX = zeros(M,N*T);
                i_aux=1;
                for j = 1 : N
                    for i = 1 : T 
                        AUX(:,i_aux) = q(samples_aux,chains(i,:,j));
                        i_aux=i_aux+1;
                    end        
                end
                den = 1/(T*N)*sum(AUX,2);
                for m = 1 : M
                    weights_aux(m) = exp(logtarget(samples_aux(m,:))) / den(m);  
                end
                weights{t,n} = weights_aux;
        end
    end
end
%
end

            
