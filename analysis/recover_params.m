% Jet Lageman, 2026

% Function for parameter and model recovery
% Input:
% isubj    = participant number
% itask    = 0 (passive) or 1 (active)
% icond    = 0 (stable/low uncertainty) or 1 (volatile/high uncertainty)
% cfg      = cell array of configurations, containing the following fields:
%   .nsim               = 100 (default); number of simulations to use
%   .outName            = 'recovery' (default) or other name for output file

function output = recover_params(cfg, isubj, itask, icond)

% unpack configurations & set defaults
try nsim              = cfg.nsim;             catch, nsim             = 100;    end
try resultName        = cfg.resultName;       catch, resultName = 'full';       end
try outName           = cfg.outName;          catch, outName = 'recovery';      end

% load results file
fname_out = sprintf('results/%s_sub%d_task%d_cond%d.mat',resultName,isubj, itask, icond);
load(fname_out,'results');
output = struct();

% determine which model has which parameters
all_params = results.options.fields;
sim_params = cell(1,length(all_params));
for p = 1:length(all_params)
    sim_params{p} = strcat('sim_', all_params{p});
    output.(sim_params{p}) = zeros(nsim,1);
end

% define settings for generative model
options = [];
options.use_obs = 0;
options.sim = 1;
options.save_file = 0;
options.get_LL = 1;
options.fields = all_params;
for par = 1:length(all_params)
    try curParam = results.(all_params{par}); catch, curParam = results.simMDP(1).(all_params{par}); end
    options.(all_params{par}) = curParam;
    output.(all_params{par}) = zeros(nsim,1) + curParam;
end
output.LL = zeros(nsim,1) + results.LL;

% fit model on simulated data
for j = 1:nsim
    sim_results = apple_fit_model(isubj, itask, icond, options);

    % record fitted parameters based on simulated data
    for par = 1:length(all_params)
        output.(sim_params{par})(j) = sim_results.(all_params{par});
    end
    output.sim_LL(j) = sim_results.LL;
end
    
% save output
fname_out = sprintf('results/recovery/%s_sub%d_task%d_cond%d.mat',outName, isubj, itask, icond);
save(fname_out, 'output')

return