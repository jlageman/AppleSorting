function rep_curve = get_rep_curve(MDP)

% ACTIVE INFERENCE MODELS
%--------------------------------------------------------------------------
if length(MDP)>1
    N = length(MDP);
    seqind = [1:(N/2) 1:(N/2)];
    rbef = zeros(N-2,1);
    raft = zeros(N-2,1);
    edir = zeros(N-2,1);

    counter = 0;
    for i = 1:N
        if seqind(i)==N/2
            continue
        end
        counter = counter + 1;
        rbef(counter) = MDP(i).action;
        raft(counter) = MDP(i+1).action;
        edir(counter) = MDP(i).seqllr*(3-2*rbef(counter));
    end
    rdir = raft==rbef;
end

% BAYESIAN INFERENCE MODELS
%--------------------------------------------------------------------------
if length(MDP)==1
    rbef = MDP.rbef;
    raft = MDP.raft;
    edir = MDP.seqllr.*(3-2*rbef);
    rdir = raft==rbef;
end

% COMPUTE REPETITION CURVE
%--------------------------------------------------------------------------
emin = [-inf, -3,  -2,  -1,   0,  +1,  +2, +3];
emax = [-3, -2,  -1,   0,  +1,  +2, +3, +inf];
nbin = numel(emax);
prep_resp = zeros(nbin,1);

for ibin = 1:nbin
    ifilt = edir > emin(ibin) & edir < emax(ibin);
    prep_resp(ibin) = mean(rdir(ifilt));
end

% fit response repetition curve with 2 or 3-param sigmoid function)
x = edir(:);
y = rdir(:);

[b,~,~] = glmfit(x,y,'binomial','link','logit');
if glmval(b,-10,'logit') > 0.5
    pse_resp = -inf;
elseif glmval(b,+10,'logit') < 0.5
    pse_resp = +inf;
else
    pse_resp = -fzero(@(L)glmval(b,L,'logit')-0.5,[-10,+10]);
end
woe_resp = b(2);

prep_resp_hat = zeros(nbin,1);
pr_hat = glmval(b,x,'logit');
for ibin = 1:nbin
    ifilt = edir > emin(ibin) & edir < emax(ibin);
    prep_resp_hat(ibin) = mean(pr_hat(ifilt));
end

rep_curve = [];
rep_curve.prep_resp = prep_resp;
rep_curve.prep_resp_hat = prep_resp_hat;
rep_curve.pse = pse_resp;
rep_curve.woe = woe_resp;

return