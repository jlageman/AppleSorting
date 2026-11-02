% Jet Lageman, 2026

% Function for parameter and model recovery
% Input:
% cfg      = cell array of configurations, containing the following fields:
%   .nsim               = 10 (default); number of simulations to use
%   .outName            = 'recovery' (default) or other name for output file
%   .all_params         = cell array of parameters to recover

function output = recover_param_values(cfg, varargin)

% unpack configurations & set defaults
try nsim              = cfg.nsim;             catch, nsim             = 10;    end
try outName           = cfg.outName;          catch, outName = 'recovery_test';      end
try all_params        = cfg.all_params;       catch, all_params = {'Hr', 'omega', 'eta_win', 'eta_loss'}; end

if nargin > 1
    outName = [outName num2str(varargin{1})];
end

% collect all parameter values to be recovered
param_vals = [];
one_bounded = {'Hr', 'eta_win', 'eta_loss', 'omega'};
unbounded = {'alpha', 'beta', 'Cr'};
for par = 1:length(all_params)
    curParam = all_params{par};
    if sum(strcmp(one_bounded, curParam))>0
        step = 1/nsim;
    elseif sum(strcmp(unbounded, curParam))>0
        step = 5/nsim;
    end
    % which range of parameter values to be tested 
    for v = 1:nsim
        param_vals{par}(v) = v*step;
    end
end

% get all combinations of parameter values
n = length(param_vals);
[param_vals{:}] = ndgrid(param_vals{end:-1:1});
param_vals = cat(n+1,param_vals{:});
param_vals = fliplr(reshape(param_vals,[],n));

% create output structure for true and simulated parameters
output = struct();
Nrows = size(param_vals,1);
sim_params = cell(1,length(all_params));
for p = 1:length(all_params)
    sim_params{p} = strcat('sim_', all_params{p});
    output.(all_params{p}) = param_vals(:,p);
    output.(sim_params{p}) = zeros(Nrows,1);
end

% define settings for generative model
options = [];
options.use_obs = 0;
options.sim = 1;
options.save_file = 0;
options.get_LL = 1;
options.fields = all_params;

for j = 1:Nrows

    for par = 1:length(all_params)
        options.(all_params{par}) = param_vals(j,par);
    end
    
    isubj = randsample(101:161, 1); itask = randsample(0:1, 1); icond = randsample(0:1, 1);
    sim_results = apple_fit_model(isubj, itask, icond, options);
    
    for par = 1:length(all_params)
        output.(sim_params{par})(j) = sim_results.(all_params{par});
    end
    
    % save output every 20 simulations
    if mod(j,20)==0
       % save output
        fname_out = sprintf('results/recovery/%s.mat',outName);
        save(fname_out, 'output') 
    end
    
end
    
% save output
fname_out = sprintf('sim/recovery/%s.mat',outName);
save(fname_out, 'output')

return