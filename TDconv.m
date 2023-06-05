function [DAsucc, DAfail] = TDconv( TD0, TD1, Post1succ, Post1fail, alph, t_delay, T, dt  )

% take (dopaminergic) prediction errors, and convolve into a DA release
% time series (Eq.30)

% INPUTS
% TD0: (DAergic) TD error, no intervention
% TD1: (DAergic) TD error, intervention
% Post1succ: for each trial type, posterior probability of intervention given success
% Post1fail: for each trial type, posterior probability of intervention given failure
% alph: time constant
% t_delay: for 'intervention' that may create a different (DAergic) TD error
% T: max time
% dt: granularity

nC = length(TD0);
ts = 0:dt:T;
nts = length(ts);
drf = (ts./alph).*exp(1-(ts./alph)); % 'dopamine response function'; Eq.29
steps_delay = t_delay/dt;

DAsucc = [];
DAfail = [];
for i = 1:nC
    DAsucc = [DAsucc; conv( (1-Post1succ(i)).*[TD0(i) zeros(1,nts-1)] + Post1succ(i).*[TD0(i) zeros(1,steps_delay) TD1(i) zeros(1,nts-steps_delay-2)], drf)];
    DAfail = [DAfail; conv( (1-Post1fail(i)).*[TD0(i) zeros(1,nts-1)] + Post1fail(i).*[TD0(i) zeros(1,steps_delay) TD1(i) zeros(1,nts-steps_delay-2)], drf)];
end
