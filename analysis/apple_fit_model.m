% Jet Lageman, 2026

% Function for fitting parameters of MDP model to apple task
% Input:
% isubj        = subject number
% itask        = task ID (0 = passive, 1 = active)
% icond        = condition ID (0 = low volatility (h = 1/12), 1 = high
%                volatility (h = 1/6))
% options      = cell array of options, containing the following fields:
%   .fields     = cell array of parameters to fit, can contain:
%       'Hr'       = subjective transition probability
%       'alpha'    = hyper-prior on precision of action selection  (higher = more precise)
%       'beta'     = hyper-prior on precision of policy selection (higher = less precise)
%       'Cr'       = reward-sensitivity parameter (>0)
%       'eta_win'  = learning rate consistent evidence (0-1)
%       'eta_loss' = learning rate inconsistent evidence (0-1)
%       'omega'    = forgetting rate for prior beliefs (0-1)
%   .outName    = name under which to save results
%   .use_obs    = use observations from participant data (1) or sample from likelihood function (0)
%   .sim        = fit using participant actions (0) or simulated actions (1)
%   .save_file  = whether to save results file (0 if used for recovery)
%   .get_LL     = whether to compute LL/AIC/BIC (0 if used for recovery)

% Output:
% results = structure containing the following fields:
%   .Hr                 = fitted belief about transition probability
%   .alpha              = fitted hyper-prior on precision of action selection  (higher = more precise)
%   .beta               = fitted hyper-prior on precision of policy selection (higher = less precise)
%   .Cr                 = fitted reward-sensitivity parameter (>0)
%   .eta_win            = fitted learning rate consistent evidence (0-1)
%   .eta_loss           = fitted learning rate inconsistent evidence (0-1)
%   .omega              = fitted forgetting rate for prior beliefs (0-1)
%   .simMDP             = simulated beliefs using fitted parameters
%   .action_probs       = probability of chosen action (2 x numTrials double)
%   .avg_action_probs   = average probability of chosen action
%   .Fe                 = model evidence
%   .LL                 = loglikelihood
%   .aic                = AIC
%   .bic                = BIC

function [results] = apple_fit_model(isubj, itask, icond, options)

% 1 Load data and define settings
%----------------------------------------------------------------------
% unpack options
try fields      = options.fields;    catch, fields    = {'alpha', 'beta'};  end
try outName     = options.outName;   catch, outName   = 'results';          end
try use_obs     = options.use_obs;   catch, use_obs   = 1;                  end
try sim         = options.sim;       catch, sim       = 0;                  end
try save_file   = options.save_file; catch, save_file = 1;                  end
try get_LL      = options.get_ll;    catch, get_LL    = 1;                  end

% determine fixed parameters
params = {};
pnam = {'Hr', 'alpha', 'beta', 'eta_win', 'eta_loss', 'omega', 'Cr'};
npar = numel(pnam);
for i = 1:npar
    if isfield(options,pnam{i}) && ~isempty(options.(pnam{i}))
        params.(pnam{i}) = options.(pnam{i});
    end
end

% load data, set params and options
fname_out = sprintf('data/matlab/dat_sub%d_task%d_cond%d.mat',isubj, itask, icond);
load(fname_out,'data');
params.states = data.seqdir_recode;
params.seqllr = data.seqllr;
responses = data.rbef_recode;
if isubj < 101
    params.condition = icond;
end

% define MDP
MDP = apple_gen_model(params);
N = MDP.numBlocks * MDP.numTrials;
[MDP(1:N)] = deal(MDP);
for i = 1:N
    % true states
    MDP(i).s = [params.states(i) params.states(i); 0 0];
    MDP(i).seqllr = params.seqllr(i);
    % observations
    if use_obs
        MDP(i).o = [1 0;
                    9 data.lvl1(i);
                    9 data.lvl2(i);
                    9 data.lvl3(i);
                    9 data.lvl4(i)];
    else
        MDP(i).o = [0 0;
                    0 0;
                    0 0;
                    0 0;
                    0 0];
    end
    % actions
    if ~sim
        MDP(i).u = [1 ; responses(i)];
    end
end

% if fitting simulated actions, run simulation first
if sim
    OPTIONS.plot = 0;
    MDP  = sim_r_apple_simple(MDP, OPTIONS);
end

% define observations and actions
U = {MDP.o};
Y = {MDP.u};


% 2 Invert model and try to recover parameters
%----------------------------------------------------------------------
DCM.MDP = apple_gen_model(params);
DCM.field = fields;               % parameter (field) names to optimise

DCM.U     = U;              % include the observations made by (real 
                                  % or simulated) participants

DCM.Y     = Y;              % include the actions made by (real or 
                                  % simulated) participants

DCM.S     = {MDP.s};              % include true hidden states

DCM       = apple_inversion(DCM); % Run the parameter estimation function


% 3 Save relevant outcomes in results structure
%----------------------------------------------------------------------
results = {};
results.DCM = DCM;
action_probs = zeros(1,numel(DCM.U));

% get fitted parameters
field = fieldnames(DCM.Ep);
for i = 1:length(field)
    if strcmp(field{i},'alpha')
        results.(field{i}) = exp(DCM.Ep.(field{i}));
        params.(field{i}) = exp(DCM.Ep.(field{i}));
    elseif strcmp(field{i},'beta')
        results.(field{i}) = exp(DCM.Ep.(field{i}));
        params.(field{i}) = exp(DCM.Ep.(field{i}));
    elseif strcmp(field{i},'Cr')
        results.(field{i}) = exp(DCM.Ep.(field{i}));
        params.(field{i}) = exp(DCM.Ep.(field{i}));
    elseif strcmp(field{i},'Hr')
        results.(field{i}) = rev_logit(DCM.Ep.(field{i}));
        params.(field{i}) = rev_logit(DCM.Ep.(field{i}));
    elseif strcmp(field{i},'eta_win')
        results.(field{i}) = rev_logit(DCM.Ep.(field{i}));
        params.(field{i}) = rev_logit(DCM.Ep.(field{i}));
    elseif strcmp(field{i},'eta_loss')
        results.(field{i}) = rev_logit(DCM.Ep.(field{i}));
        params.(field{i}) = rev_logit(DCM.Ep.(field{i}));
    elseif strcmp(field{i},'omega')
        results.(field{i}) = rev_logit(DCM.Ep.(field{i}));
        params.(field{i}) = rev_logit(DCM.Ep.(field{i}));
    else
        results.(field{i}) = exp(DCM.Ep.(field{i}));
        params.(field{i}) = exp(DCM.Ep.(field{i}));
    end
end


% 4 Simulate data with fitted parameters
%--------------------------------------------------------------------------
if get_LL
    mdp_out = apple_gen_model(params);
    [simMDP(1:N)]   = deal(mdp_out);
    
    for j = 1:size(simMDP,2)
        simMDP(j).o = DCM.U{j};
        simMDP(j).u = DCM.Y{j};
        simMDP(j).s = DCM.S{j};
        simMDP(j).seqllr = MDP(j).seqllr;
    end
    
    OPTIONS.plot = 0;
    simMDP  = sim_r_apple_simple(simMDP, OPTIONS);
    
    % calculate loglikelihood & action probs
    L = 0;
    for j = 1:size(simMDP,2)
        action_probs(j) = simMDP(j).P(DCM.Y{j}(2));
        L = L + log(simMDP(j).P(simMDP(j).u(2)) + eps);
    end
    
    % add simulated data and averages to results
    results.simMDP = simMDP;
    results.action_probs = action_probs;
    results.avg_action_probs = mean(action_probs);
    results.Fe = DCM.F;
    results.LL = L;
    
    % get complexity-penalized fitting metrics
    nfit = length(fields);
    ntrl = size(simMDP,2);
    results.aic = -2*L+2*nfit+2*nfit*(nfit+1)/(ntrl-nfit+1); % AIC
    results.bic = -2*L+nfit*log(ntrl); % BIC
    results.options = options;
end
    
if save_file
    fname_out = sprintf('results/%s_sub%d_task%d_cond%d.mat',outName,isubj, itask, icond);
    save(fname_out, 'results')
end

return

% auxiliary functions
%=========================================================================
function A = rev_logit(A)
A = 1/(1+exp(-A));
return