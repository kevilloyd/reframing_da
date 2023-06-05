function [err, params, ppress_succ, E_tau, ptau_press_succ, ptau_press0_succ, ptau_press1_succ, ptau_press0, ptau_press1, DA0, DA1, Post1press, Post1nopress, DAsucc, DAfail, ppress0, ppress1, V, Q_tau_press, Q_other, rho, taus] =...
    roesch_func( x, w_rt, w_da, w_reg, w_prec, ppress_data, rts_means )

% find fitting error between model (parametrized by x) and data (from Roesch's mixed-valence experiment)
%
% INPUTS
% x: parameters (see below)
% w_rt: how much weight to put on RT fits relative to fits of P(press), where the latter has weight 1
% w_da:
% w_reg:
% w_prec:
% ppress_data: vector of P(press) for [f,n,s]
% rts_means: vector of mean RTs for PRESS trials [f,n,s]
%
% OUTPUTS
% err: fitting error
% params: parameter values (transformed)
% ppress_succ: model's probabilties of pressing 
% E_tau: model's average RTs
% ptau...: distributions over response times conditioned on various things
% &tc.

%% unpack parameters
ws = exp(x(1:6));           % positivity constraint on instrumental/Pavlovian weights
kap = 1./(1+exp(-x(7)));    % probability of control; Eq.8
params = [ws kap];
w_i = ws(1);    % instrumental weight for choosing appropriate (press) vs. inappropriate (other) action
w_pp = ws(2);   % Pavlovian weight choosing appropriate vs. inappropriate action for POSITIVE values/prediction errors
w_pn = ws(3);   % Pavlovian weight choosing appropriate vs. inappropriate action for NEGATIVE values/prediction errors
w_tau_i = ws(4);    % instrumental weight for choosing pressing latency
w_tau_pp = ws(5);   % Pavlovian "                 "             " for POSITIVE values/prediction errors
w_tau_pn = ws(6);   % Pavlovian "                 "             " for NEGATIVE values/prediction errors

%% (other) fixed quantities
r = 4;          % utility of food reward
t_delay = .25;  % time (seconds) that it takes for the internal decision about whether to intervene or not
xi = .7;        % time constant when convolving TD error
phi = 2.5;      % punishment sensitivity...
z = -phi*r;     % ... determines (dis)utility of shock
r_O = .0;   % reward for choosing 'other' action
c_p = -.1;  % vigour cost of pressing
alph = .8;  % alph of positive TDE goes into DA channel, and (1-alph) of negative TDE
base = z;   % baseline (i.e., utility of safety to be gained by being active on shock trials)
% dynamic programming
tol = 1e-6;
max_iters=100;
%
nS = 5; % number of states (1 = cue/lever-in/make internal choice; 2=s_uncontr., 3=s_contr., s_fail, s_succ)
nC = 3; % number of tones: 1=f(ood), 2=n(eutral), 3=s(hock)
rew=1; neu=2; shk=3; % to help with readibility below
s_cue=1; s_0=2; s_1=3; s_fail=4; s_succ=5; % states: "               "
fail = 1; success = 2;
nA = 2; % number of actions: here, we'll abstractly say that 1='task-appropriate' (press lever), and 2=task-inappropriate (other)
c_LP = 0;   % possible unit costs of actions (we'll just set to zero)
c_O = 0;
t_cue = 5; % 5s cue presentation
t_lever = 10; % max response time (i.e., max time lever available)
iti = 20; % in expt, 20s ITI
dtau = 1e-2;            % granularity of choices of latency
tau_min = .5;
tau_max = 2*t_lever; % although it does strike us as a bit unlikely that an animal would choose to press so slowly...
taus = tau_min:dtau:tau_max;
ntaus = length(taus);

%% expected immediate reward for action
idx_tau_succ = find(taus<=t_lever);
idx_tau_fail = find(taus>t_lever);
rs_press = c_LP + c_p./taus;
ttl = [taus(idx_tau_succ) ones(1,length(idx_tau_fail)).*t_lever]; % time to lever press completion or timeout
r_other = c_O + r_O;
% and at outcome
rs_outcome = zeros(nC,2);
rs_outcome(:,fail) = [0;0;z];
rs_outcome(:,success) = [r;0;0];

%% DP
V = zeros(nC,nS); % value function
Q_tau_press = zeros(nC,ntaus); % Q-values associated with pressing latencies
Q_press0 = zeros(nC,1);
Q_press1 = zeros(nC,1);
Q_other = zeros(nC,1);
rho = 0;
vals_old = [V(:); rho];
conv=0; % converged yet?
iter=0;
while conv==0 && iter<max_iters
    iter=iter+1;
    %     fprintf('\n iter %i\n    rho = %6.2f\n',iter, rho)
    
    % current prediction errors
    TDv0 = V(:,s_cue); % Eq.10
    TDv1 = TDv0;
    TDv1(shk) = V(shk,s_cue) - base; % Eq.11

    % update final state values (Eqs 17, 18)
    V(:,s_succ) = rs_outcome(:,success) - rho*iti;
    V(:,s_fail) = rs_outcome(:,fail) - rho*iti;
    
    % Q-values (Eq.11)
    Q_tau_press(:,idx_tau_succ) = rs_press(:,idx_tau_succ) + V(:,s_succ) - rho.*taus(idx_tau_succ);
    Q_tau_press(:,idx_tau_fail) = rs_press(:,idx_tau_fail) + V(:,s_fail) - rho*t_lever;
    
    % distributions over pressing latencies (Eq.14)
    vals_tau0 = w_tau_i.*Q_tau_press - ( (TDv0>=0).*w_tau_pp.*TDv0 + (TDv0<0).*w_tau_pn.*TDv0 ).*taus; % if no intervention
    vals_tau1 = w_tau_i.*Q_tau_press - ( (TDv1>=0).*w_tau_pp.*TDv1 + (TDv1<0).*w_tau_pn.*TDv1 ).*taus; % if intervention
    ptau_press0 = exp( vals_tau0 - logsumexp(vals_tau0,2) ); % distribution over chosen pressing latencies if no intervention
    ptau_press1 = exp( vals_tau1 - logsumexp(vals_tau1,2) ); % distribution over chosen pressing latencies if intervention
    
    % and therefore work out the overall Q-values with respect to those
    % distributions
    Q_press0 = sum( ptau_press0.*Q_tau_press, 2 ); % Eq.15
    Q_press1 = sum( ptau_press1.*Q_tau_press, 2 );
    Q_other = r_other + V(:,s_fail) - rho*tau_max; % Eq.13
    
    vals0 = w_i*(Q_press0-Q_other) + ( (TDv0>=0).*w_pp.*TDv0 + (TDv0<0).*w_pn.*TDv0 ); % so +ve DA facilitates appropriate action (pressing), while -ve DA does the opposite
    vals1 = w_i*(Q_press1-Q_other) + ( (TDv1>=0).*w_pp.*TDv1 + (TDv1<0).*w_pn.*TDv1 ); % so +ve DA facilitates appropriate action (pressing), while -ve DA does the opposite
    
    % Eq.16
    ppress0 = 1./(1+exp(-(vals0))); % probability of choosing to press given no intervention 
    ppress1 = 1./(1+exp(-(vals1))); % probability of choosing to press given intervention
    
    V(:,s_0) = ppress0.*Q_press0 + (1-ppress0).*Q_other;
    V(:,s_1) = ppress1.*Q_press1 + (1-ppress1).*Q_other;
    V(:,s_cue) = (1-kap).*V(:,s_0) + kap.*V(:,s_1);
    
    % average reward rate
    t_av = t_cue + (1-kap).*( ppress0.*(sum(ptau_press0.*ttl,2)) + (1-ppress0).*t_lever ) +...
        kap.*( ppress1.*(sum(ptau_press1.*ttl,2)) + (1-ppress1).*t_lever ) + ...
        iti;
    r_av = (1-kap).*( ppress0.*( sum(ptau_press0(:,idx_tau_succ).*(rs_press(idx_tau_succ)+rs_outcome(:,success)),2) + sum(ptau_press0(:,idx_tau_fail).*(rs_press(idx_tau_fail)+rs_outcome(:,fail)),2) ) +...
        (1-ppress0).*( r_other + rs_outcome(:,fail) ) ) +...
        kap.*( ppress1.*( sum(ptau_press1(:,idx_tau_succ).*(rs_press(idx_tau_succ)+rs_outcome(:,success)),2) + sum(ptau_press1(:,idx_tau_fail).*(rs_press(idx_tau_fail)+rs_outcome(:,fail)),2) ) +...
        (1-ppress1).*( r_other + rs_outcome(:,fail) ) );
    rho = sum(r_av)/sum(t_av);
    
    % update
    vals_new = [V(:); rho];
    dif = max(abs(vals_new-vals_old));
    if (dif<tol)
        conv=1;
    end
    vals_old = vals_new;
    
end

%% post-process
ppress = (1-kap).*ppress0 + kap.*ppress1; % overall probability of CHOOSING to press
psucc_press0 = sum(ptau_press0(:,idx_tau_succ),2); % conditional probability of successfully completing press, given chose to press
psucc_press1 = sum(ptau_press1(:,idx_tau_succ),2);
psucc0 = ppress0.*psucc_press0;
psucc1 = ppress1.*psucc_press1;
ppress_succ = (1-kap).*psucc0 + kap.*psucc1; % overall, unconditional probability of a SUCCESSFUL press (i.e., need not only to choose press, but the complete the press before max. response window)
%
ptau_press0_succ = ptau_press0(:,idx_tau_succ);
ptau_press0_succ = ptau_press0_succ./sum(ptau_press0_succ,2); % distribution of latencies of SUCCESSFUL presses
ptau_press1_succ = ptau_press0_succ;
ptau_press1_succ(shk,:) = ptau_press1(shk,idx_tau_succ);
ptau_press1_succ(shk,:) = ptau_press1_succ(shk,:)./sum(ptau_press1_succ(shk,:));
ptau_press_succ = (1-kap).*ptau_press0_succ + kap.*ptau_press1_succ;

E_tau = sum(ptau_press_succ.*taus(idx_tau_succ),2); % expected pressing latencies for each trial type

% Eq's 32, 33
Post1press = kap*psucc1(shk) / ( kap*psucc1(shk) + (1-kap)*psucc0(shk) ); % what is the probability of having intervened on a shock given that you succeeded in pressing?
Post1nopress = kap*(1-psucc1(shk)) / ( kap*(1-psucc1(shk)) + (1-kap)*(1-psucc0(shk)) ); % and the probability of having intervened on a shock trial given did NOT succeed in pressing?

%% DA
DA0 = alph.*(TDv0>=0).*TDv0 + (1-alph).*(TDv0<0).*TDv0; % Eq.29
DA1 = alph.*(TDv1>=0).*TDv1 + (1-alph).*(TDv1<0).*TDv1;
[DAsucc, DAfail] = TDconv( DA0, DA1,  [0;0;Post1press], [0;0;Post1nopress], xi, t_delay, tau_max, dtau  ); % Eq.31

%% error
maxDA = max(DAsucc,[],2);
DAerr = abs( maxDA(rew) - maxDA(shk) );
% Eq.34
err = sum( abs(ppress_data-ppress_succ') ) + w_rt*sum( w_prec'.*abs( rts_means - E_tau' ) ) + w_da*DAerr + w_reg*sum(abs(params));

end
