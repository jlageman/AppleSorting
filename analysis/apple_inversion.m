% Jet Lageman, 2026
% MDP inversion using Variational Bayes
% FORMAT [DCM] = apple_inversion(DCM)
%
% Expects:
%--------------------------------------------------------------------------
% DCM.MDP   % MDP structure specifying a generative model
% DCM.field % parameter (field) names to optimise
% DCM.U     % cell array of outcomes (stimuli)
% DCM.Y     % cell array of responses (action)
%
% Returns:
%--------------------------------------------------------------------------
% DCM.M     % generative model (DCM)
% DCM.Ep    % Conditional means (structure)
% DCM.Cp    % Conditional covariances
% DCM.F     % (negative) Free-energy bound on log evidence
% 
% This routine inverts (cell arrays of) trials specified in terms of the
% stimuli or outcomes and subsequent choices or responses. It first
% computes the prior expectations (and covariances) of the free parameters
% specified by DCM.field. These parameters are log scaling parameters that
% are applied to the fields of DCM.MDP. 
%
% If there is no learning implicit in multi-trial games, only unique trials
% (as specified by the stimuli), are used to generate (subjective)
% posteriors over choice or action. Otherwise, all trials are used in the
% order specified. The ensuing posterior probabilities over choices are
% used with the specified choices or actions to evaluate their log
% probability. This is used to optimise the MDP (hyper) parameters in
% DCM.field using variational Laplace (with numerical evaluation of the
% curvature).
%__________________________________________________________________________
% Copyright (C) 2005 Wellcome Trust Centre for Neuroimaging

% Karl Friston
% $Id: spm_dcm_mdp.m 7679 2019-10-24 15:54:07Z spm $

function [DCM] = apple_inversion(DCM)

% OPTIONS
%--------------------------------------------------------------------------
ALL = false;

% set defaults if parameters are not fitted
%--------------------------------------------------------------------------
if ~isfield(DCM.MDP,'Hr'), DCM.MDP.Hr = 1/9; end
if ~isfield(DCM.MDP,'alpha'), DCM.MDP.alpha = 4; end
if ~isfield(DCM.MDP,'beta' ), DCM.MDP.beta  = 1; end
if ~isfield(DCM.MDP,'eta' ), DCM.MDP.eta  = 1; end
if ~isfield(DCM.MDP,'eta_loss' ), DCM.MDP.eta_loss  = 0; end
if ~isfield(DCM.MDP,'omega' ), DCM.MDP.omega  = 0; end
if ~isfield(DCM.MDP,'Cr'), DCM.MDP.Cr = 1; end

% prior expectations and covariance
%--------------------------------------------------------------------------
prior_variance = 1;

for i = 1:length(DCM.field)
    field = DCM.field{i};
    try
        param = DCM.MDP.(field);
        param = double(~~param);
    catch
        param = 1;
    end
    if ALL
        pE.(field) = zeros(size(param));
        pC{i,i}    = diag(param);
    else
        if strcmp(field,'alpha')
            pE.(field) = log(4);          % in log-space (to keep positive)
            pC{i,i}    = prior_variance;
        elseif strcmp(field,'beta')
            pE.(field) = log(1);           % in log-space (to keep positive)
            pC{i,i}    = prior_variance;
        elseif strcmp(field,'Cr')
            pE.(field) = log(1);           % in log-space (to keep positive)
            pC{i,i}    = prior_variance;
        elseif strcmp(field,'Hr')
            pE.(field) = logit(1/9); % in logit-space - bounded between 0 and 1
            pC{i,i}    = prior_variance;
        elseif strcmp(field,'eta')
            pE.(field) = logit(0.9); % in logit-space - bounded between 0 and 1
            pC{i,i}    = prior_variance;
        elseif strcmp(field,'eta_loss')
            pE.(field) = logit(0.1); % in logit-space - bounded between 0 and 1
            pC{i,i}    = prior_variance;
        elseif strcmp(field,'omega')
            pE.(field) = logit(0.1); % in logit-space - bounded between 0 and 1
            pC{i,i}    = prior_variance;
        else
            pE.(field) = 0;                % if it can take any negative or positive value
            pC{i,i}    = prior_variance;
        end
    end
end
pC      = spm_cat(pC);

% model specification
%--------------------------------------------------------------------------
M.L     = @(P,M,U,Y)spm_mdp_L(P,M,U,Y);  % log-likelihood function
M.pE    = pE;                            % prior means (parameters)
M.pC    = pC;                            % prior variance (parameters)
M.mdp   = DCM.MDP;                       % MDP structure
% set to 0 to print & make graph
M.nograph = 1;
M.noprint = 1;

% Variational Laplace
%--------------------------------------------------------------------------
[Ep,Cp,F] = spm_nlsi_Newton(M,DCM.U,DCM.Y);

% Store posterior densities and log evidnce (free energy)
%--------------------------------------------------------------------------
DCM.M   = M;
DCM.Ep  = Ep;
DCM.Cp  = Cp;
DCM.F   = F;

return


% likelihood function
%==========================================================================

function L = spm_mdp_L(P,M,U,Y)
% log-likelihood function
% FORMAT L = spm_mdp_L(P,M,U,Y)
% P    - parameter structure
% M    - generative model
% U    - inputs
% Y    - observed repsonses
%__________________________________________________________________________

if ~isstruct(P); P = spm_unvec(P,M.pE); end

% multiply parameters in MDP
%--------------------------------------------------------------------------
mdp   = M.mdp;
field = fieldnames(M.pE);
for i = 1:length(field)
    if strcmp(field{i},'alpha')
        mdp.(field{i}) = exp(P.(field{i}));
        params.(field{i}) = exp(P.(field{i}));
    elseif strcmp(field{i},'beta')
        mdp.(field{i}) = exp(P.(field{i}));
        params.(field{i}) = exp(P.(field{i}));
    elseif strcmp(field{i},'Cr')
        mdp.(field{i}) = exp(P.(field{i}));
        params.(field{i}) = exp(P.(field{i}));
    elseif strcmp(field{i},'Hr')
        mdp.(field{i}) = rev_logit(P.(field{i}));
        params.(field{i}) = rev_logit(P.(field{i}));
    elseif strcmp(field{i},'eta')
        mdp.(field{i}) = rev_logit(P.(field{i}));
        params.(field{i}) = rev_logit(P.(field{i}));
    elseif strcmp(field{i},'eta_loss')
        mdp.(field{i}) = rev_logit(P.(field{i}));
        params.(field{i}) = rev_logit(P.(field{i}));
    elseif strcmp(field{i},'omega')
        mdp.(field{i}) = rev_logit(P.(field{i}));
        params.(field{i}) = rev_logit(P.(field{i}));
    else
        mdp.(field{i}) = exp(P.(field{i}));
        params.(field{i}) = exp(P.(field{i}));
    end
end

% create model
N = mdp.numTrials * mdp.numBlocks;
params.condition = mdp.condition;
MDP = apple_gen_model(params);
[MDP(1:N)]   = deal(MDP);

for j = 1:size(MDP,2)
    MDP(j).o = U{j};
    MDP(j).u = Y{j};
    MDP(j).s = [mdp.states(j) mdp.states(j); 0 0];
    MDP(j).seqllr = mdp.true_seqllr(j);
end

% solve MDP and accumulate log-likelihood
%--------------------------------------------------------------------------
OPTIONS.plot = 0;
MDP  = sim_r_apple_simple(MDP, OPTIONS);

L = 0;
for j = 1:N
    L = L + log(MDP(j).P(Y{j}(2)) + eps);
end

clear('MDP')

fprintf('LL: %f \n',L)

return