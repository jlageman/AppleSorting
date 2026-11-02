function [MDP] = sim_r_apple_simple(MDP,OPTIONS)
% active inference and learning for apple paradigm
% FORMAT [MDP] = spm_MDP_VB_X(MDP,OPTIONS)

% handle multiple trials, ensuring parameters (and posteriors) are updated
%==========================================================================
if size(MDP,2) > 1
    
    % check if we have one block or multiple blocks
    if size(MDP,2)==144 % AP1
        OUT1 = sim_r_apple_simple(MDP(1:72), OPTIONS);
        OUT2 = sim_r_apple_simple(MDP(73:144), OPTIONS);
        MDP = [OUT1 OUT2];
        return
    elseif size(MDP,2)==190 %AP3
        OUT1 = sim_r_apple_simple(MDP(1:95), OPTIONS);
        OUT2 = sim_r_apple_simple(MDP(96:190), OPTIONS);
        MDP = [OUT1 OUT2];
        return
    end

    % plotting options
    %----------------------------------------------------------------------
    GRAPH        = OPTIONS.plot;
    OPTIONS.plot = 0;
    
    % loop over trials
    for i = 1:size(MDP,2)                  % number of trials
        if i > 1                       % if previous inversions
            
            % update concentration parameters
            %----------------------------------------------------------
            MDP(i)  = spm_MDP_update(MDP(i),OUT(i - 1));
            
            % update initial states (post-diction)
            %----------------------------------------------------------
            for f = 1:numel(MDP(i).D)
                MDP(i).D{f} = OUT(i - 1).d{f};
            end
        end
    
        % solve this trial (for all models synchronously)
        %------------------------------------------------------------------
        OUT(i) = sim_r_apple_simple(MDP(i),OPTIONS);
        
    end
    
       % plot summary statistics - over trials
    %----------------------------------------------------------------------
    MDP = OUT;
    if GRAPH
        figure('Position', [400 200 1000 600], 'Color','w', 'ColorMap',gray(64), 'DefaultTextColor','k'); clf
        plot_resp_apple(MDP(1,:))
    end
    return
end


% set up and preliminaries
%==========================================================================

% defaults
%--------------------------------------------------------------------------
try alpha       = MDP(1).alpha;     catch, alpha    = 512;  end % action precision
try beta        = MDP(1).beta;      catch, beta     = 1;    end % policy precision
try eta_win     = MDP(1).eta_win;   catch, eta_win  = 1;    end % learning rate (consistent)
try eta_loss    = MDP(1).eta_loss;  catch, eta_loss = 0;    end % learning rate (inconsistent)
try omega       = MDP(1).omega;     catch, omega    = 0;    end % forgetting rate: discounts prior belief

% initialise model-specific variables
%==========================================================================
V = MDP.V;                    % allowable policies (T - 1,Np,Nf)
T = size(MDP.V,1) + 1;        % number of transitions

% numbers of transitions, policies and states
%----------------------------------------------------------------------
Ng = numel(MDP.A);               % number of outcome factors
Nf = numel(MDP.B);               % number of hidden state factors
Np = size(V,2);                  % number of allowable policies
for f = 1:Nf
    Ns(f) = size(MDP.B{f},1);     % number of hidden states
    Nu(f) = size(MDP.B{f},3);     % number of hidden controls
end
for g = 1:Ng
    No(g) = size(MDP.A{g},1);     % number of outcomes
end

% parameters of generative model and policies
%======================================================================

% likelihood model (for a partially observed MDP)
%----------------------------------------------------------------------
for g = 1:Ng
    
    % ensure probabilities are normalised  : A
    %------------------------------------------------------------------
    MDP.A{g} = spm_norm(MDP.A{g});
    true_A{g} = spm_norm(MDP.A{g});
    
    % parameters (concentration parameters): a
    %------------------------------------------------------------------
    if isfield(MDP,'a')
        A{g}  = spm_norm(MDP.a{g});
    else
        A{g}  = spm_norm(MDP.A{g});
    end
    
end

% transition probabilities (priors)
%----------------------------------------------------------------------
for f = 1:Nf
    for j = 1:Nu(f)
        
        % controllable transition probabilities : B
        %--------------------------------------------------------------
        MDP.B{f}(:,:,j) = spm_norm(MDP.B{f}(:,:,j));
        
        % parameters (concentration parameters): b
        %--------------------------------------------------------------
        if isfield(MDP,'b')
            sB{f}(:,:,j) = spm_norm(MDP.b{f}(:,:,j) );
            rB{f}(:,:,j) = spm_norm(MDP.b{f}(:,:,j)');
        else
            sB{f}(:,:,j) = spm_norm(MDP.B{f}(:,:,j) );
            rB{f}(:,:,j) = spm_norm(MDP.B{f}(:,:,j)');
        end
        
    end
    
end

% priors over initial hidden states - concentration parameters
%----------------------------------------------------------------------
for f = 1:Nf
    if isfield(MDP,'d')
        D{f} = spm_norm(MDP.d{f});
    elseif isfield(MDP,'D')
        D{f} = spm_norm(MDP.D{f});
    end
end

% prior preferences (log probabilities) : C
%----------------------------------------------------------------------
for g = 1:Ng
    if isfield(MDP,'C')
        C{g}  = MDP.C{g};
    else
        C{g}  = zeros(No(g),1);
    end
    
    % assume time-invariant preferences, if unspecified
    %------------------------------------------------------------------
    if size(C{g},2) == 1
        C{g} = repmat(C{g},1,T);
        if isfield(MDP,'c')
            MDP.c{g} = repmat(MDP.c{g},1,T);
            pC{g}     = repmat(pC{g},1,T);
        end
    end
    C{g} = spm_log(spm_softmax(C{g}));
end

% initialise  posterior expectations of hidden states
%----------------------------------------------------------------------
for f = 1:Nf
    x{f}  = zeros(Ns(f),T,Np)      + 1/Ns(f);
    X{f}  = repmat(D{f},1,1);
    for k = 1:Np
        x{f}(:,1,k) = D{f};
        x{f}(:,2,k) = sB{f}(:,:,V(1,f,k))*x{f}(:,1);
    end
end

% initialise posteriors over policies and action
%----------------------------------------------------------------------
P  = zeros([Nu,1]);
u  = zeros(Np,1);

% if there is only one policy
%----------------------------------------------------------------------
if Np == 1
    u = ones(Np,T);
end

% if states have not been specified set to 0
%----------------------------------------------------------------------
s  = zeros(Nf,T);
try
    i       = find(MDP.s);
    s(i) = MDP.s(i);
end
MDP.s    = s;

% if outcomes have not been specified set to 0
%----------------------------------------------------------------------
o  = zeros(Ng,T);
try
    i = find(MDP.o);
    o(i) = MDP.o(i);
end
MDP.o = o;

% (indices of) plausible (allowable) policies
%----------------------------------------------------------------------
p  = 1:Np;

% expected rate parameter (precision of posterior over policies)
%----------------------------------------------------------------------
qb = beta;                          % initialise rate parameters
w  = 1/qb;                       % posterior precision (policy)

% TIME 1: COMPUTE EFE AND SELECT ACTION
%==========================================================================
Q   = zeros(Np,1);                  % expected free energy
if Np > 1
    for k = p
        for j = 2
            
            % get expected states for this policy and time
            %--------------------------------------------------
            for f = 1:Nf
                xq{f} = x{f}(:,j,k);
            end
            
            % (negative) expected free energy
            %==================================================
            for g = 1:Ng
                
                % prior preferences about outcomes
                %----------------------------------------------
                qo   = spm_dot(A{g},xq(:));
                Q(k) = Q(k) + qo'*(C{g}(:,j));
                
            end
        end
    end
end

% posterior policy beliefs + action selection (softmax of action potential)
%----------------------------------------------------------
u = spm_softmax(w*Q)';
P = spm_softmax(alpha*log(u(:)))';

% next action - sampled from marginal posterior
%----------------------------------------------------------
try
    MDP.u(:) = MDP.u(:);
    action = MDP.u(2);
    MDP.action = action;
catch
    ind           = find(rand < cumsum(P(:)),1);
    MDP.u(:) = [1 ind];
    action = ind;
    MDP.action = action;
end

% TIME 2: GENERATE OBSERVATIONS, DO INFERENCE & LEARNING
%==========================================================================
    
% sample side of observations, if not specified
%----------------------------------------------------------
if MDP.s(2,2) == 0
    ps = MDP.B{2}(:,1,MDP.action);
    MDP.s(2,1) = 1;
    MDP.s(2,2) = find(rand < cumsum(ps),1);
end 

% sample outcome, if not specified
%--------------------------------------------------------------
for g = 1:Ng
    
    % if outcome is not specified
    %----------------------------------------------------------
    if MDP.o(g,2)==0

        % sample from likelihood given hidden state
        %--------------------------------------------------
        ind           = num2cell(MDP.s(:,2));
        po            = MDP.A{g}(:,ind{:});
        MDP.o(g,2) = find(rand < cumsum(po),1);

    end
end

% get outcome likelihood (O{m})
%------------------------------------------------------------------
for g = 1:Ng
        
    % sampled outcome
    %----------------------------------------------------------
    O{g} = sparse(MDP.o(g,2),1,1,No(g),1);
    
end
       
% compute likelihood & log-likelihood
%==================================================================
L = 1;
true_L = 1;
for g = 1:Ng
    L = L.*spm_dot(A{g},O{g});
    true_L = true_L.*spm_dot(true_A{g},O{g});
end

xq{1}(:) = [1 1];
xq{2}(:) = O{1};
% subjective likelihood
logLik = spm_dot(L,xq(:),1);
logLik = spm_log(logLik(:));
likelihood = spm_softmax(logLik);
% true likelihood
true_logLik = spm_dot(true_L,xq(:),1);
true_logLik = spm_log(true_logLik(:));
true_likelihood = spm_softmax(true_logLik);

% do inference and belief updating
%==================================================================
prior = X{1}(:,1);
logPrior = spm_log(prior);
logPosterior = logPrior + logLik;
posterior = spm_softmax(logPosterior);
X{1}(:,2) = posterior;
X{2}(:,2) = [0 0 0];
X{2}(MDP.action+1,2) = 1;

% learning - accumulate concentration parameters
%==========================================================================

% initial hidden states:
%----------------------------------------------------------------------
if isfield(MDP,'d')
    for f = 1
        MDP.d{f} = spm_softmax((1 - omega) * logPosterior);
    end
end

% mapping from hidden states to outcomes: a
%------------------------------------------------------------------
if isfield(MDP,'a')
    % consistent or inconsistent evidence?
    [~,evdir] = max(likelihood);
    if evdir==action
        cur_eta = eta_win;
    else
        cur_eta = eta_loss;
    end
    % learning
    for g = 1:Ng
        da     = O(g);
        for  f = 1:Nf
            da = spm_cross(da,X{f}(:,2));
        end
        da     = da.*(MDP.a{g} > 0);
        MDP.a{g} = MDP.a{g} + da*cur_eta;
    end
end
    
% assemble results and place in MDP structure
%----------------------------------------------------------------------
MDP.T  = T;         % number of belief updates
MDP.F  = logPosterior;         % VFE
MDP.G  = Q;         % EFE
MDP.O  = O;         % outcomes
MDP.P  = P;         % probability of action 
MDP.R  = u;         % conditional expectations over policies
MDP.X  = X;         % beliefs about hidden states
MDP.prior = prior;
MDP.likelihood = likelihood;
MDP.posterior = posterior;
MDP.LLR = spm_log(true_likelihood(1)/true_likelihood(2));

return
end

% auxillary functions
%==========================================================================

function MDP = spm_MDP_update(MDP,OUT)
% FORMAT MDP = spm_MDP_update(MDP,OUT)
% moves Dirichlet parameters from OUT to MDP
% MDP - structure array (new)
% OUT - structure array (old)
%__________________________________________________________________________

% check for concentration parameters at this level
%--------------------------------------------------------------------------
try,  MDP.a = OUT.a; end
try,  MDP.b = OUT.b; end
try,  MDP.c = OUT.c; end
try,  MDP.d = OUT.d; end
try,  MDP.e = OUT.e; end

% check for concentration parameters at nested levels
%--------------------------------------------------------------------------
try,  MDP.MDP(1).a = OUT.mdp(end).a; end
try,  MDP.MDP(1).b = OUT.mdp(end).b; end
try,  MDP.MDP(1).c = OUT.mdp(end).c; end
try,  MDP.MDP(1).d = OUT.mdp(end).d; end
try,  MDP.MDP(1).e = OUT.mdp(end).e; end

return
end
