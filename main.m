clc; close all; clear

expt = 'walton'; % which experiment? 'roesch' or 'walton'?
cohort = 'good'; % 'good' or 'poor' avoiders? Only relevant for fitting 'roesch' data
num_starts_init = 1; % number of random starts for FIRST stage of optimization (find promising starting points for second stage)
num_starts_fmin = 1; % number of starting points for SECOND stage of optimization (using fminsearch)

rng(0) % set random seed (if desired)

%% meta-parameters
switch expt
    case 'roesch'
        num_free = 7;   % number of free parameters
        w_rt = .25;     % weight placed on fit to RTs (for pressing), relative to fit to accuracies
        w_da = .1;      % weight placed on fit to DA
        switch cohort
            case 'good'
                w_reg = .01; % regularization weight
            case 'poor'
                w_reg = .0;
        end
    case 'walton'
        num_free = 10;      % number of free parameters 
        w_rt = [.018 .018]; % weights placed on fit to RTs (correct, incorrect), relative to fit to accuracies
        w_da = 0;           % weight placed on fit to DA
        w_reg = 0;          % regularization weight
end

%% fit data
options = optimset('Display','iter','MaxIter',100);
switch expt
    %% Roesch
    case 'roesch'
        switch cohort
            case 'good'
                succ_data = [.95 .82 .86];
                succ_data_ses = [.03 .09 .08];
                rts_means = [1.1 1.5 1.3];
                rts_ses = [.2 .4 .6];
            case 'poor'
                succ_data = [.94 .60 .46];
                succ_data_ses = [.03 .1 .09];
                rts_means = [1.0 2.5 2.6];
                rts_ses = [.2 .5 .6];
        end
        lapse_rate = 1-succ_data(1); % presumably the proportion of food press failures reflect lapses
        succ_data_surr = succ_data+lapse_rate;
        w_prec = 1./rts_ses';
        % STAGE 1
        fprintf('initial search...\n\n')
        for i = 1:num_starts_init
            fprintf('iter %d\n',i)
            switch cohort
                case 'good'
                    %                     x0_init(i,:) = randn(1,num_free);
                    x0_init(i,:) = [0.7702  -45.9005   -9.3678    3.7337     0.3590  -27.8188   -0.5726]; 
                case 'poor'
                    %                     x0_init(i,:) = randn(1,num_free);
                    x0_init(i,:) = [-8.1868   -0.3203   -0.5237   -7.2632    0.1801   -4.9815    0.0547]; 
            end
            fval_init(i) = roesch_func( x0_init(i,:), w_rt, w_da, w_reg, w_prec, succ_data_surr, rts_means );
        end
        [vals_temp,id_temp] = sort(fval_init,'ascend');
        x0_inits = x0_init(id_temp,:);
        % STAGE 2
        fprintf('\n\n second-stage search...\n\n')
        for i = 1:num_starts_fmin
            x0 = x0_inits(i,:);
            [x(i,:),FVAL(i)] = fminsearch( @(x) roesch_func( x, w_rt, w_da, w_reg, w_prec, succ_data_surr, rts_means ), x0, options );
        end
        [fvals,bw] = sort(FVAL,'ascend');
        x_best = x(bw,:); % best
        
        [err, params, ppress, E_tau, p_tau, p_tau_succ0, p_tau_succ1, ptau_press0, ptau_press1, DA0, DA1, Post1press, Post1nopress, DAsucc, DAfail, ppress0, ppress1, V, Q_tau, Q_other, rho, taus] =...
            roesch_func( x_best(1,:), w_rt,  w_da, w_reg, w_prec, succ_data_surr, rts_means );
        
        plot_roesch_fit( succ_data, succ_data_ses, rts_means, rts_ses, (ppress-lapse_rate), E_tau, DA0, DA1, DAsucc, DAfail, taus )
        
    %% Walton
    case 'walton'
        % ordering is [GS GL NGS NGL]
        succ_data = [.83 .97 .76 .71]'; % success rates
        succ_data_ses = [.07 .01 .04 .06]';
        lapse_rate = 1-succ_data(2); % presumably the proportion of GL failures reflect lapses
        succ_data_surr = succ_data+lapse_rate; % therefore we'll actually fit to this 'surrogate' that ignores the lapse rate
        rts_corr_means = [0.9 0.4 2.9 2.7]'; % estimated mean RTs for correct trials
        rts_corr_ses = [.1 .1 .2 .2]'; % estimated SEM RTs for correct trials
        rts_err_means = [1.5 0.7 0.8 1.0]'; % mean RTs for error trials
        rts_err_ses = [.4 .3 .1 .1]'; % estimated SEM RTs for error trials
        w_prec = [1./rts_corr_ses' 1./rts_err_ses'];
        % STAGE 1
        fprintf('initial search...\n\n')
        for i = 1:num_starts_init
            fprintf('iter %d\n',i)
            %             x0_init(i,:) = randn(1, num_free);
            x0_init(i,:) = [1.0805   -0.4170    0.6616    2.4536   -5.1211    0.0001   -2.0353    1.9358    0.0001   -5.1132];
            fval_init(i) = walton_func( x0_init(i,:), w_rt, w_da, w_reg, w_prec, succ_data_surr, rts_corr_means, rts_err_means );
        end
        [vals_temp,id_temp] = sort(fval_init,'ascend');
        x0_inits = x0_init(id_temp,:);
        % STAGE 2
        fprintf('\n\n second-stage search...\n\n')
        for i = 1:num_starts_fmin
            x0 = x0_inits(i,:);
            [x(i,:), fval(i)] = fminsearch( @(x) walton_func( x, w_rt, w_da, w_reg, w_prec, succ_data_surr, rts_corr_means, rts_err_means ), x0, options );
        end
        [fvals,bw] = sort(fval,'ascend');
        x_best = x(bw,:); % best
        
        [err, params, p_succ, p_succ0, p_succ1, E_tau_succ, E_tau_fail, Post1succ, Post1fail,  DAsucc, DAfail, p_tau, p_tau_succ, p_tau_fail,...
            p_tau0, p_tau1, p_tau_succ0, p_tau_succ1, p_tau_fail0, p_tau_fail1, Q_tau, TDv0, TDv1, DA0, DA1, taus, P_press, psucc_press, Et_press] ...
            = walton_func( x_best(1,:), w_rt, w_da, w_reg, w_prec, succ_data, rts_corr_means, rts_err_means );
        
        plot_walton_fit( succ_data, succ_data_ses, rts_corr_means, rts_err_means,  rts_corr_ses, rts_err_ses,...
            (p_succ-lapse_rate), E_tau_succ, E_tau_fail, DA0, DA1, DAsucc, DAfail, taus)
        
end
