function [samples_bis,z_aux,logEval_aux] = fun_MHmod(start, nsamples, logtarget, prop, proprnd) %Metropolis-Hastings algorithm

DIM = size(start,2);
samples = zeros(nsamples, DIM);
samples(1,:) = start;                                                            
z_aux = zeros(nsamples-1,DIM);
logEval_aux = zeros(nsamples-1,1);
    for t = 2:nsamples
        z = proprnd(samples(t-1,:));
        z_aux(t-1,:) = z;
        logtar_aux = logtarget(z);
        logEval_aux(t-1) = logtar_aux;
        
        alpha = exp(  logtar_aux + log(prop(samples(t-1,:), z))...
            - logtarget(samples(t-1,:)) -  log(prop(z, samples(t-1,:))) );
        if rand < alpha
            samples(t,:) = z; %aceptar el nuevo valor con prob alpha
        else 
            samples(t,:) = samples(t-1,:); %rechazamos con prob 1- alpha     
        end 
    end
samples_bis = samples(1:end-1,:);
end