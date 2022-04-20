function weights = fun_lowerWeighting(samples, logEvals, chains,q,denom)
% compute intermediate-cost IS weights for all samples using the proposal
% in qrnd using the locations parameters from chains

[T,~, N] = size(chains);
weights = zeros(T,N);

for n = 1 : N
   disp(['Weighting ',num2str(n),'-th chain'])
    for  t = 1 : T
        
        switch denom
            case 1 % spatial mixture
                aux=0;
                for i = 1 : N 
                    aux = aux + q(samples(t,:,n),chains(t,:,i));
                end
                den = 1/N*aux;
                weights(t,n) = exp(logEvals(t,n)) / den;       
                
                
            
            case 2 % temporal mixture
                aux=0;
                for i = 1 : T 
                    aux = aux + q(samples(t,:,n),chains(i,:,n));
                end
                den = 1/T*aux;
                weights(t,n) = exp(logEvals(t,n)) / den;       
                
                
            case 3 % complete mixture
                aux = 0;
                for j = 1 : N
                    for i = 1 : T 
                        aux = aux + q(samples(t,:,n),chains(i,:,j));
                    end        
                end
                den = 1/(T*N)*aux;
                weights(t,n) = exp(logEvals(t,n)) / den;  

        end
    end
end
%
end

            
