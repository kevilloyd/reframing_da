function [err, params, Psucc, P_succ0, P_succ1, Etau_leave_succ, Etau_leave_fail, Post1succ, Post1fail, DAsucc, DAfail, ptau_leave, ptau_leave_succ, ptau_leave_fail,...
    ptau_leave0, ptau_leave1, ptau_leave0_succ, ptau_leave1_succ, ptau_leave0_fail, ptau_leave1_fail, Q_tau_leave, TDv0, TDv1, DA0, DA1, taus, P_press, psucc_press, Et_press]...
    = walton_func( x, w_rt, w_da, w_reg, w_prec, succ_data, rts_corr_means, rts_err_means )

% find fitting error between model (parametrized by x) and data (from Walton's Go/NoGo experiment)
%
% INPUTS
% x: parameter vector (see below)
% w_rt: vector of weights [w_success_RTs w_fail_RTs] on RT fits relative to
% fits of P(correct), where the latter has weight 1
% w_da: weight on DA fit
% w_reg: weight for regularization
% w_prec: vector of weights to place more importance on fitting less
% variable observations
% succ_data: vector of P(correct) for [GS,GL,NGS,NGL]
% rts_corr_means: vector of mean RTs for success trials (same order as for succ_data)
% rts_err_means: vector of mean RTs for error trials (same order as for succ_data)
%
% OUTPUTS
% err: error between model and data
% params: parameter values in the transformed space
% p_succ: model P(success)
% E_tau_leave_succ: model meanRT (successful trials)
% E_tau_leave_fail: "           " (fail trials)
% Post1succ: posterior probability of intervention given success
% Post1fail: "                                   " given fail
% &tc...

%% unpack parameters
ws = exp(x(1:5));           % positivity constraint
kap = 1./(1+exp(-x(6)));    % probability of intervention (Eq.8)
r_O = 1./(1+exp(-x(7))); % reward per unit time for choosing 'other' action
c_f = -.5./(1+exp(-x(8))); % some cost per second (<=0) of being in noseport (assumed unpleasant?)
c_l = -.2./(1+exp(-x(9))); % some vigor cost for leaving noseport (i.e., can't leave infinitely fast)
c_p = -.2./(1+exp(-x(10))); % some vigor cost for pressing lever (i.e., can't press infinitely fast)
params = [ws kap r_O c_f c_l c_p];
wi_press = ws(1);        % instrumental macroscopic weight (for choosing press vs. other)
wi_tau_press = ws(2);    % instrumental weight for choosing nosepoke-exit latency and pressing latencies?
wi_tau_leave = ws(3);
wpp_tau = ws(4);   % Pavlovian "                 "             " for POSITIVE values/prediction errors
wpn_tau = ws(5);   % Pavlovian "                 "             " for NEGATIVE values/prediction errors

%% fixed quantities
alph = .8; % alph of positive TDE goes into DA channel, and (1-alph) of negative TDE
t_delay = .25; % delay between cue and next state (where either control applied or not) in seconds
xi = 0.7; % time constant when convolving TD error
base = 2; % i.e., the value of the large reward that would be lost by premature breaking of fixation
% Dynamic Programming
tol = 1e-3;
max_iters=50;
%
nS = 6; % number of states (cue, s_uncontr., s_contr., s_out, s_fail, s_succ)
nC = 4; % number of tones: 1=GS, 2=GL, 3=NGS, 4=NGL
GS=1; GL=2; NGS=3; NGL=4; % trial types: to help with readibility below
s_cue=1; s_in0=2; s_in1=3; s_out=4; s_fail=5; s_succ=6; % states: "               "
fail = 1; success = 2;
nA = 2; % number of actions: here, we'll abstractly say that 1=task-appropriate action (press or wait), and 2=task-inappropriate action (e.g. go groom)
rL = 2; % large reward
rS = rL/2; % small reward
t_go_max = 5; % on Go trial, if fail to 2xpress within 5s, you fail
t_NP = 0.5; % on average, the time to fixate until elicit cue is 0.5s (in fact, it's uniformly drawn 0.3-0.7s)
iti = 5; % 5s ITI
tto = 5; % if error or failed trial, you get additional 5s timeout before ITI
dtau = 1e-2;
tau_min = dtau;
tau_max = 5; % arbitrary max on response latency (since I think theoretically, the animal could fixate on NoGo trial for arbitrarily long)
taus = tau_min:dtau:tau_max;
ntaus = length(taus);
dt = dtau;
ts = dt:dt:t_go_max;
nts = length(ts);
tau_mat = taus(ones(nC,1),:);
tone_off1 = 1.7;
tone_off2 = 1.9;

%% immediate expected rewards
Rs_outcome = zeros(nC,2);
Rs_outcome(:,success) = [rS;rL;rS;rL];
% having left noseport at time-since-cue-onset t, we need to work out
% transitions and rewards for each choice (a,tau); this is only slightly subtle for
% lever pressing on Go trials, since must complete by 5s
p_succ_LP = zeros(ntaus,ntaus); % row index for time since cue onset
for i=1:ntaus-1
    p_succ_LP( i, taus <= (t_go_max-taus(i)) ) = 1;
end
Rs_press = zeros(ntaus,ntaus); % i.e., for each time t since cue onset
for i=1:ntaus    
    Rs_press(i,:) = c_p./taus;
end
Rs_other = (t_go_max-taus').*r_O; % on the other hand, choosing alternative accrues reward at rate r_O up to the deadline
P_succ_ng = unifcdf(taus, tone_off1, tone_off2); % i.e., probability that for each leaving time, the tone has turned off before leaving
Rs_leave = repmat( c_l./taus + c_f.*taus, [4 1] );

% DP 
V_in = zeros(nC,3); % value function for states in nosepoke: s_cue, s_in0, s_in1
V_out = zeros(ntaus,2); % outside nosepoke (just relevant for GO trials)
V_outcome = zeros(nC,2);
Q_tau_leave = zeros(nC,ntaus); % Q-values associated with NP leaving
Q_leave0 = zeros(nC,1);
Q_leave1 = Q_leave0;
Q_tau_press = zeros(ntaus,ntaus,2); % values of pressing, at each post-cue time t (row), at latency tau (column) (tone still on), for 1=GS and 2=GL
Q_press = zeros(ntaus,2); % values of choosing to press at each post-cue time t on Go trials
Q_other = zeros(ntaus,2); % values of choosing alternative at each post-cue time t on Go trials
rho = 0;
TDv0 = zeros(nC,1); TDv1 = TDv0; 
P_succ0 = zeros(nC,1);
P_succ1 = zeros(nC,1);
vals_old = [V_in(:); V_out(:); V_outcome(:); rho];
conv=0; % converged yet?
iter=0;
while conv==0 && iter<max_iters
    iter=iter+1;
    %     fprintf('\n iter %i\n    rho = %6.4f    diff = %6.4f\n',iter, rho, dif)
    
    % current best guess at prediction errors
    TDv0 = V_in(:,s_cue); % Eq.10
    TDv1 = TDv0;
    TDv1(NGL) = V_in(NGL,s_cue) - base; % Eq.11
    
    % update final state values
    V_outcome(:,success) = Rs_outcome(:,success) - rho*iti; % Eq.27
    V_outcome(:,fail) = Rs_outcome(:,fail) - rho*(tto+iti); % Eq.28
    
    % OUTSIDE NOSEPORT: for GO trials, on leaving nosepoke, we assume a choice about whether
    % to press the lever or not; not that this depends on when you left! (Eq's 21, 22)
    Q_tau_press(:,:,GS) = Rs_press + p_succ_LP.*V_outcome(GS,success) + (1-p_succ_LP).*V_outcome(GS,fail) - rho.*taus;
    Q_tau_press(:,:,GL) = Rs_press + p_succ_LP.*V_outcome(GL,success) + (1-p_succ_LP).*V_outcome(GL,fail) - rho.*taus;
    Q_other(:,GS) = Rs_other - rho.*(t_go_max-taus') + V_outcome(GS,fail);
    Q_other(:,GL) = Rs_other - rho.*(t_go_max-taus') + V_outcome(GL,fail);
    
    vals_tau_LP = wi_tau_press.*Q_tau_press;
    ptau_press = exp( vals_tau_LP - logsumexp(vals_tau_LP,2) ); % Eq.23; if choose press, what would the distribution of latencies be?
    psucc_press = squeeze( sum( ptau_press.*repmat(p_succ_LP,[1 1 2]), 2 ) );   % and what would the probability of successfully completing a press if initiated at each post-cue time?
    Q_press = squeeze( sum( ptau_press.*Q_tau_press, 2 ) ); % Eq.24
    P_press = 1./(1+exp(wi_press*(Q_other-Q_press))); % Eq.25; probability of choosing to press lever at each post-cue time, having left nosepoke
    Et_press = squeeze( sum( ptau_press.*taus, 2 ) ); % for each post-cue time, what is the expected time until outcome if you were to press (either successfull press or failure)?
    Et_other = t_go_max - taus'; % if you chose otherwise, then it's just until the cue goes off

    ttg = iti + P_press.*(Et_press + (1-psucc_press)*tto ) + (1-P_press).*(Et_other + tto); % expected time UNTIL NEXT TRIAL from point at which left nosepoke
    ppg = P_press.*psucc_press; % probabilities of success (i.e., you need to press, and you need to complete the press)
    rrg = P_press.*( squeeze(sum( ptau_press.*Rs_press, 2 )) ) + (1-P_press).*Rs_other + ppg.*[rS rL]; % expected total future reward from point at which left nosepoke
    
    V_out(:,GS:GL) = P_press.*Q_press + (1-P_press).*Q_other; % Eq.26; so these are the values of being outside the nosepoke, and haven't failed yet
    
    % INSIDE NOSEPORT
    % for both s_0 and s_1, the instrumental values for leaving q(tau) are the same -- but there will be
    % different p(tau) because of the differing Pavlovian 'warpings' of
    % those distributions (Eq.19)
    Q_tau_leave(GS:GL,:) = Rs_leave(GS:GL,:) + V_out' - rho.*taus;
    Q_tau_leave(NGS:NGL,:) = Rs_leave(NGS:NGL,:) + P_succ_ng.*V_outcome(NGS:NGL,success) + (1-P_succ_ng).*V_outcome(NGS:NGL,fail) - rho.*taus;
    % distributions over latencies, using previous values of TD errors and
    % state values (Eq.20)
    vals_tau0 = wi_tau_leave.*Q_tau_leave - ( (TDv0>=0).*wpp_tau.*TDv0 + (TDv0<0).*wpn_tau.*TDv0 ).*taus; % if no intervention
    vals_tau1 = wi_tau_leave.*Q_tau_leave - ( (TDv1>=0).*wpp_tau.*TDv1 + (TDv1<0).*wpn_tau.*TDv1 ).*taus; % if intervention
    ptau_leave0 = exp( vals_tau0 - logsumexp(vals_tau0,2) ); % distribution over latencies if NO intervention
    ptau_leave1 = exp( vals_tau1 - logsumexp(vals_tau1,2) ); % distribution over latencies if YES intervention
    vals_tau = vals_tau0;
    vals_tau(NGL,:,:) = (1-kap).*vals_tau0(NGL,:,:) + kap.*vals_tau1(NGL,:,:);
    ptau_leave = exp( vals_tau - logsumexp(vals_tau,2) );
    
    % and therefore work out the overall Q-values with respect to those
    % distributions
    Q_leave0 = sum( ptau_leave0.*Q_tau_leave, 2 );
    Q_leave1 = sum( ptau_leave1.*Q_tau_leave, 2 );
    V_in(:,s_in0) = Q_leave0;
    V_in(:,s_in1) = Q_leave1;
    V_in(:,s_cue) = (1-kap).*V_in(:,s_in0) + kap.*V_in(:,s_in1);
    
    % what are therefore the probabilities of success (as judged from the
    % experimenter's viewpoint) given s_0 vs s_1?
    % for GO trials, you need to leave the noseport, choose to press the
    % lever, and complete the press before the deadline
    P_succ0(GS:GL) = sum( ptau_leave0(GS:GL,:).*(P_press.*psucc_press)', 2 );
    P_succ1(GS:GL) = P_succ0(GS:GL);
    P_succ0(NGS:NGL) = sum( P_succ_ng.*ptau_leave0(NGS:NGL,:), 2);
    P_succ1(NGS:NGL) = sum( P_succ_ng.*ptau_leave1(NGS:NGL,:), 2);
    
    % update average reward rate
    t_av0(GS:GL) = t_NP + sum( ptau_leave0(GS:GL,:).*(taus+ttg'), 2 );
    t_av1(GS:GL) = t_av0(GS:GL);
    t_av0(NGS:NGL) = t_NP + sum( ptau_leave0(NGS:NGL,:).*taus, 2 ) + iti + (1-P_succ0(NGS:NGL))*tto;
    t_av1(NGS:NGL) = t_NP + sum( ptau_leave1(NGS:NGL,:).*taus, 2 ) + iti + (1-P_succ1(NGS:NGL))*tto;
    r_av0(GS:GL) = sum( ptau_leave0(GS:GL,:).*Rs_leave(GS:GL,:), 2 ) + sum( ptau_leave0(GS:GL,:).*rrg', 2 );
    r_av0(NGS:NGL) = sum( ptau_leave0(GS:GL,:).*Rs_leave(GS:GL,:), 2 ) + P_succ0(NGS:NGL).*[rS;rL];
    r_av1(GS:GL) = r_av0(GS:GL);
    r_av1(NGS:NGL) = sum( ptau_leave1(GS:GL,:).*Rs_leave(GS:GL,:), 2 ) + P_succ1(NGS:NGL).*[rS;rL];
    t_av = (1-kap).*t_av0 + kap.*t_av1;
    r_av = (1-kap).*r_av0 + kap.*r_av1;
    rho = sum(r_av)/sum(t_av);
    
    % update
    vals_new = [V_in(:); V_out(:); V_outcome(:); rho];
    dif = max(abs(vals_new-vals_old));
    if (dif<tol)
        conv=1;
    end
    vals_old = vals_new;
    
end

%% post-process
% P(success)
Psucc = (1-kap).*P_succ0 + kap.*P_succ1;
% Leaving times
Etau_leave0 = dot(ptau_leave0,tau_mat,2);
Etau_leave1 = dot(ptau_leave1,tau_mat,2);
Etau_leave = dot(ptau_leave,tau_mat,2);
%
ptau_leave_succ = ptau_leave;
ptau_leave_succ(GS:GL,:) = ptau_leave(GS:GL,:).*P_press'.*psucc_press'; % for GO trials, you have to leave, choose to press, and press before before the deadline!
ptau_leave_succ(NGS:NGL,:) = ptau_leave(NGS:NGL,:).*P_succ_ng;
ptau_leave_fail = ptau_leave;
ptau_leave_fail(GS:GL,:) = ptau_leave(GS:GL,:).*( P_press'.*(1-psucc_press)' + (1-P_press)' );
ptau_leave_fail(NGS:NGL,:) = ptau_leave(NGS:NGL,:).*(1-P_succ_ng);
ptau_leave_succ = ptau_leave_succ./sum(ptau_leave_succ,2);
ptau_leave_fail = ptau_leave_fail./sum(ptau_leave_fail,2);
Etau_leave_succ = sum(ptau_leave_succ.*taus,2);
Etau_leave_fail = sum(ptau_leave_fail.*taus,2);
%
ptau_leave0_succ = ptau_leave0;
ptau_leave0_succ(GS:GL,:) = ptau_leave0(GS:GL,:).*P_press'.*psucc_press'; % for GO trials, you have to leave, choose to press, and press before before the deadline!
ptau_leave0_succ(NGS:NGL,:) = ptau_leave0(NGS:NGL,:).*P_succ_ng;
ptau_leave0_succ = ptau_leave0_succ./sum(ptau_leave0_succ,2);
ptau_leave1_succ = ptau_leave0_succ;
ptau_leave1_succ(NGL,:) = ptau_leave1(NGL,:).*P_succ_ng;
ptau_leave1_succ = ptau_leave1_succ./sum(ptau_leave1_succ,2);
%
ptau_leave0_fail = ptau_leave0;
ptau_leave0_fail(GS:GL,:) = ptau_leave0(GS:GL,:).*( P_press'.*(1-psucc_press)' + (1-P_press)' );
ptau_leave0_fail(NGS:NGL,:) = ptau_leave0(NGS:NGL,:).*(1-P_succ_ng);
ptau_leave0_fail = ptau_leave0_fail./sum(ptau_leave0_fail,2);
ptau_leave1_fail = ptau_leave0_fail;
ptau_leave1_fail(NGL,:) = ptau_leave1(NGL,:).*P_succ_ng;

% pressing times (only relevant for GS and GL)
ptau_press_succ = ptau_press .* repmat(p_succ_LP,[1 1 2]);
ptau_press_succ = ptau_press_succ./sum(ptau_press_succ,2);
Etaus_press = squeeze( sum( ptau_press_succ.*taus, 2 ) ); % expected latency of press for each nosepoke exit time
Etau_press = sum( ptau_leave(GS:GL,:).*Etaus_press', 2, 'omitnan' );

% what about consideration of intervention or not for NGL trials?
% the point here is that if you have a successful NGL trial, this could
% happen if one intervened or not --- and so consider the posterior
% probability of having intervened given success (and similarly for fail)
Post1succ = P_succ1(NGL)*kap / ( P_succ1(NGL)*kap + P_succ0(NGL)*(1-kap) ); % Eq.32
Post1fail = (1-P_succ1(NGL))*kap / ( (1-P_succ1(NGL))*kap + (1-P_succ0(NGL))*(1-kap) ); % Eq.33

%% DA
DA0 = alph.*(TDv0>=0).*TDv0 + (1-alph).*(TDv0<0).*TDv0; % Eq.28
DA1 = alph.*(TDv1>=0).*TDv1 + (1-alph).*(TDv1<0).*TDv1;
[DAsucc, DAfail] = TDconv( DA0, DA1,  [0;0;0;Post1succ], [0;0;0;Post1fail], xi, t_delay, tau_max, dtau  ); % Eq.31

%% error
minDA = min(DAsucc,[],2);
maxDA = max(DAsucc,[],2);
DAerr = sum( abs( diff([minDA(GS) minDA(NGS) minDA(NGL)]) ) ) - (DA0(NGL)<0)*DA0(NGL) + (DA1(NGL)>0)*DA1(NGL);

err = sum( abs(succ_data-Psucc) ) + w_rt(1)*sum( w_prec(1:4)'.*abs( rts_corr_means - Etau_leave_succ ) ) + ...
    w_rt(2)*sum( w_prec(5:end)'.*abs( rts_err_means - Etau_leave_fail ) ) + w_da*DAerr + w_reg*sqrt(sum(x.^2)); % Eq.34

end
