function samples = fun_MH(start, nsamples, logtarget, prop, proprnd) %Metropolis-Hastings algorithm

DIM = size(start,2);
samples = zeros(nsamples, DIM);
samples(1,:) = start;                                                            

    for t = 2:nsamples
        z = proprnd(samples(t-1,:));
        alpha = exp(  logtarget(z) + log(prop(samples(t-1,:), z))...
            - logtarget(samples(t-1,:)) -  log(prop(z, samples(t-1,:))) );
        if rand < alpha
            samples(t,:) = z; %aceptar el nuevo valor con prob alpha
        else 
            samples(t,:) = samples(t-1,:); %rechazamos con prob 1- alpha     
        end 
    end

end