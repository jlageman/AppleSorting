% Jet Lageman, 2026

% Function for creating MDP model for applesorting task
% Input:
% params    = structure, can contain the following:
%   .Hr        = belief about hazard rate
%   .alpha     = hyper-prior on precision of action selection  (higher = more precise)
%   .beta      = hyper-prior on precision of policy selection (higher = less precise)
%   .Cr        = reward-sensitivity parameter (>0)
%   .eta       = learning rate for prior beliefs (0-1)
%   .omega     = forgetting rate for prior beliefs (0-1)
%   .states    = cell array of true hidden states
%   .seqllr    = cell array of true LLR sequence

% Output:
% mdp model containing observation model, transition probs etc
function mdp = apple_gen_model(params)

% set defaults
%--------------------------------------------------------------------------
% parameter values
try Hr       = params.Hr;       catch, Hr       = 1/9;  end % initial Hr
try alpha    = params.alpha;    catch, alpha    = 4;    end % action precision
try beta     = params.beta;     catch, beta     = 1;    end % policy precision
try Cr       = params.Cr;       catch, Cr       = 1;    end % outcome sensitivity
try eta      = params.eta;      catch, eta      = 1;    end % learning rate
try eta_loss = params.eta_loss; catch, eta_loss = 0;    end % learning rate penalty for disagreeing evidence
try omega    = params.omega;    catch, omega    = 0;    end % forgetting rate
try condition = params.condition; catch, condition = 0; end % which condition?

% set up MDP structure
%--------------------------------------------------------------------------
%mdp.num_states = [2 3]; % FACTOR 1:  1) blueLeft, 2) orangeLeft - FACTOR 2: 1) start, 2) left, 3) right
%mdp.num_obs = [3 9 9 9 9]; % MODALITY 1: 1) start, 2) left, 3) right; MODALITY 2-5: 1) blue++++, 2) blue+++, 3) blue++, 4) blue+, 5) orange+, 6) orange++, 7) orange+++, 8) orange++++, 9) no apple
%mdp.num_controls = [1 2]; % CONTROL 1: 1) no control - CONTROL 2: 1) left, 2) right
mdp.T = 2;

% D - prior about initial states
D{1} = [0.5 0.5]';
D{2} = [1 0 0]';
% d - allow learning priors over trials
d{1} = [4 4]';
d{2} = D{2};

% A - likelihood matrix
A{1}(:,:,1) = [1 1; % start
               0 0;
               0 0];
A{1}(:,:,2) = [0 0;
               1 1; % left
               0 0];
A{1}(:,:,3) = [0 0;
               0 0;
               1 1]; % right
           
for m = 2:5
    if condition==0 % low uncertainty          
        A{m}(:,:,1) = [0 0;
                       0 0;
                       0 0;
                       0 0;
                       0 0;
                       0 0;
                       0 0;
                       0 0;
                       1 1];
        A{m}(:,:,2) = [0.18260363 0.06978214;
                       0.17578429 0.07777068;
                       0.15772787 0.09146681;
                       0.13410989 0.11075470;
                       0.1107547  0.13410989;
                       0.09146681 0.15772787;
                       0.07777068 0.17578429;
                       0.06978214 0.18260363;
                       0          0        ];
        A{m}(:,:,3) = [0.06978214 0.18260363;
                       0.07777068 0.17578429;
                       0.09146681 0.15772787;
                       0.1107547  0.13410989;
                       0.13410989 0.11075470;
                       0.15772787 0.09146681;
                       0.17578429 0.07777068;
                       0.18260363 0.06978214;
                       0          0        ];
    elseif condition==1 % high uncertainty
        A{m}(:,:,1) = [0 0;
                       0 0;
                       0 0;
                       0 0;
                       0 0;
                       0 0;
                       0 0;
                       0 0;
                       1 1];
        A{m}(:,:,2) = [0.16313631 0.09159918;
                       0.15619661 0.09575767;
                       0.14411206 0.10392340;
                       0.12966941 0.11560546;
                       0.11560546 0.12966941;
                       0.10392340 0.14411206;
                       0.09575767 0.15619661;
                       0.09159918 0.16313631;
                       0          0        ];
        A{m}(:,:,3) = [0.09159918 0.16313631;
                       0.09575767 0.15619661;
                       0.10392340 0.14411206;
                       0.11560546 0.12966941;
                       0.12966941 0.11560546;
                       0.14411206 0.10392340;
                       0.15619661 0.09575767;
                       0.16313631 0.09159918;
                       0          0        ];
    end
end


% B - transition matrix
B{1}(:,:,1) = [1 0;
               0 1];
B{2}(:,:,1) = [0 0 0; % start
               1 1 1; % left
               0 0 0]; % right
B{2}(:,:,2) = [0 0 0; % start
               0 0 0; % left
               1 1 1]; % right
% b - beliefs about transitions
b{1}(:,:,1) = [(1-Hr) Hr;
               Hr (1-Hr)];
b{2} = B{2};

% C - preference matrix
C{1} = zeros(3,mdp.T);
for m = 2:5
    C{m} = zeros(9,mdp.T);
    for t = 2:mdp.T
        C{m}(:,t) = [4*Cr 3*Cr 2*Cr Cr -Cr -2*Cr -3*Cr -4*Cr 0];
    end
end

% V - allowable policies
V(1,:,1) = [1 1];
V(1,:,2) = [1 2];


% assemble in one structure
%--------------------------------------------------------------------------
mdp.V = V;                    % policies V

mdp.A = A;                    % state-outcome mapping
mdp.B = B;                    % transition probabilities
mdp.C = C;                    % preferred states
mdp.D = D;                    % priors over initial states

mdp.b = b;                    % enable beliefs about transition probability
mdp.d = d;                    % enable learning priors over initial states

label.factor{1}   = 'mapping';   label.name{1}    = {'blueLeft','orangeLeft'};
label.factor{2}   = 'behavior state';     label.name{2}    = {'start','left','right'};
label.modality{1} = 'observed action';  label.outcome{1} = {'start','left','right'};
for m = 2:5
    label.modality{m} = 'apple';    label.outcome{m} = {'blue++++','blue+++','blue++','blue+','orange+','orange++','orange+++','orange++++','null'};
end
label.action{2} = {'left','right'};
mdp.label = label;

% add parameters and options
%--------------------------------------------------------------------------
% record parameters
mdp.Hr = Hr;
mdp.alpha = alpha;
mdp.beta = beta;
mdp.Cr = Cr;
mdp.eta = eta;
mdp.eta_loss = eta_loss;
mdp.omega = omega;

% record number of blocks and trials
mdp.numBlocks = 2;
if isfield(params,'states')
    mdp.numTrials = numel(params.states)/mdp.numBlocks;
    mdp.states = params.states;
else
    mdp.numTrials = 95;
end
mdp.condition = condition;
if isfield(params,'seqllr')
    mdp.true_seqllr = params.seqllr;
end

% final check
mdp = spm_MDP_check(mdp);

% make graph?
%spm_MDP_factor_graph(mdp)

end