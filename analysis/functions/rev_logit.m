function A = rev_logit(A)
A = 1/(1+exp(-A));
return