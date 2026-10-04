function A = logit(A)
% catch boundary cases
if A==0
    A = A+eps;
elseif A==1
    A = A-eps;
end
% take logit
A = log(A/(1-A));
return