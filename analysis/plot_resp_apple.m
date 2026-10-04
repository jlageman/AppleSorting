function Q = plot_resp_apple(MDP)
% auxiliary plotting routine for sim_r_apple - multiple trials
% FORMAT Q = plot_resp_apple(MDP)

% numbers of transitions, policies and states
%--------------------------------------------------------------------------
if iscell(MDP(1).X)
    Nf = numel(MDP(1).B);                 % number of hidden state factors
    Ng = numel(MDP(1).A);                 % number of outcome factors
else
    Nf = 1;
    Ng = 1;
end

% collect data for graphics
%==========================================================================
Nt    = numel(MDP);                      % number of trials
Ne    = MDP(1).T;                        % number of outcomes per trial
Np    = size(MDP(1).V,2);                % number of policies
u = zeros(2,Nt);

for i = 1:Nt

    try
        MDP(i).v;
    catch
        MDP(i).v = [];
    end
    
    s(:,i) = MDP(i).s(:,1);
    o(:,i) = MDP(i).o(:,end);
    u(1:2,i) = MDP(i).R;
    v{1,i} = MDP(i).v;

    % assemble context learning
    %----------------------------------------------------------------------
    for f = 1:Nf
        try
            try
                D = MDP(i).d{f};
            catch
                D = MDP(i).D{f};
            end
        catch
            try
                D = MDP(i).d;
            catch
                D = MDP(i).D;
            end
        end
        d{f}(:,i) = D/sum(D);
    end
end

% assemble output structure if required
%--------------------------------------------------------------------------
if nargout
    Q.prior  = d;            
    Q.policy  = u;            % final policy expectations
    Q.state  = s;            % inital hidden states
    Q.outcome  = o;            % final outcomes
    return
end

% do actual plotting
%==========================================================================
% plot: policy beliefs
col   = {'r.','g.','b.','c.','m.','k.'};
%u     = spm_cat(u);
v     = spm_cat(v);
t     = 1:Nt;
subplot(2,1,1)
if Nt < 64
    MarkerSize = 24;
else
    MarkerSize = 16;
end
image(t,1:Np,64*(1 - u)),  hold on
plot(linspace(1,Nt,numel(v)),v,'c.','MarkerSize',16)
for f = 1
    for i = 1:max(s(f,:))
        j = find(s(f,:) == i);
        plot(t(j),j - j + i - 2,col{rem(i - 1,6)+ 1},'MarkerSize',MarkerSize)
    end
end
set(gca,'YLim',[-Nf,Np] + 1/2), box off
set(gca,'YTickLabel',{});
set(gca, 'YTick', [-1,0,1,2])
set(gca, 'YTickLabel', [MDP(1).label.name{1} MDP(1).label.action{2}])
title('Initial state and policy selection')
xlabel('Trial'),ylabel('Policy'), hold off

% plot: context learning
for f = 1
    subplot(2,1,2), image(64*(1 - d{f}))
    if f < 2
        title('Context Learning')
    end
    set(gca,'XTick',1:Nt);
    if f < Nf
        set(gca,'XTickLabel',{});
    end
    set(gca,'YTick',[1,2]);
    try
        set(gca,'YTickLabel',MDP(1).label.name{f});
    end
end