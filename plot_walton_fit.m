function [] = plot_walton_fit( succ_data, succ_data_ses, RTs_corr_means, RTs_err_means,  RTs_corr_ses, RTs_err_ses,...
                    p_succ, E_tau_succ, E_tau_fail, DA0, DA1, DAsucc, DAfail, taus )

%% plot summary measures of Walton data and fits
% succ_data: empirical probabilities correct for each trial type
% succ_data_ses: empirical standard errors
% RTs_corr_means:
% RTs_err_means:
% RTs_corr_ses:
% RTs_err_ses:
% p_succ: model P(success)
% E_tau_succ:
% E_tau_fail:
% DA0: no intervention
% DA1: intervention
% DAsucc: success
% DAfail: fail
% taus: all possible latencies

GS=1;
GL=2;
NGS=3;
NGL=4;

% colour scheme
colGS = [.8 0 0];
colGL = [1 0 0];
colNGS = [0 .5 0];
colNGL = [0 .9 0];
colMat = [colGS;colGL;colNGS;colNGL];

strVec = {'GS','GL','NGS','NGL'};
msize = 30;

T = 5; % max time
dt = 1e-2; % granularity of time, in seconds
ts = 0:dt:T;
nts = length(ts);
dtau = taus(2)-taus(1);

figure 
hold on
b=bar(succ_data);
set(gca,'ylim',[0 1])
b.FaceColor = 'flat';
b.CData = colMat;
errorbar(1:4,succ_data,succ_data_ses,'k','linestyle','none')
plot(p_succ,'k.','markerSize',msize)
ylabel('Pr(correct)')
set(gca,'xtick',1:4,'XTickLabel',strVec)

figure
hold on
ms_temp = [RTs_corr_means'; RTs_err_means'];
ms_temp = ms_temp(:);
ses_temp = [RTs_corr_ses; RTs_err_ses];
ses_temp = ses_temp(:);
b = bar(ms_temp);
set(gca,'ylim',[0 5])
b.FaceColor = 'flat';
b.CData = repelem(colMat,2,1);
errorbar(1:8,ms_temp,ses_temp,'k','linestyle','none')
ms_temp = [E_tau_succ'; E_tau_fail'];
ms_temp = ms_temp(:);
plot(ms_temp,'k.','markerSize',msize)
ylabel('RT (s)')
set(gca,'xtick',1:8,'XTickLabel',{'s','f','s','f','s','f','s','f'})

maxDA = .7;
minDA = -.4;
figure
hold on
b=bar(DA0);
b.FaceColor = 'flat';
b.CData = colMat;
ylabel('\delta_\chi')
set(gca,'xtick',1:4,'XTickLabel',strVec,'box','off','ylim',[minDA maxDA])

figure
hold on
b=bar(DA1);
b.FaceColor = 'flat';
b.CData = colMat;
ylabel('\delta_\chi')
set(gca,'xtick',1:4,'XTickLabel',strVec,'box','off','ylim',[minDA maxDA])

figure
hold on
temp = -2:dt:-dt;
h = plot([temp ts],[zeros(4,length(temp)) DAsucc([2 1 4 3],1:nts)],'linewidth',2);
set(h, {'color'}, num2cell(colMat([2 1 4 3],:), 2), {'linestyle'}, {'-';':';'-';':'});
set(gca,'xlim',[-2 5],'xtick',-2:1:5,'box','off','ylim',[minDA maxDA])
ylims = get(gca,'ylim');
plot([temp ts],zeros(1,length([temp ts])),'k','linewidth', 0.5);
plot(zeros(100,1), linspace(ylims(1),ylims(2), 100), 'k--', 'linewidth', 0.5)
plot(ones(100,1).*-.5, linspace(ylims(1),ylims(2), 100), 'k--', 'linewidth', 0.5 )
xlabel('time (s)')
ylabel('\Delta[DA]')
legend(strVec,'location','NorthWest')

figure
hold on
h = plot([temp ts],[[zeros(1,length(temp)) DAsucc(end,1:nts)];[zeros(1,length(temp)) DAfail(end,1:nts)]],'linewidth',2);
set(h, {'color'}, {colNGL;colNGL}, {'linestyle'}, {'-';'--'});
maxDA = maxDA/3;
minDA = minDA/3;
set(gca,'xlim',[-.5 3],'xtick',0:1:3,'ylim',[minDA maxDA],'ytick',-.1:.1:.2,'box','off')
ylims = get(gca,'ylim');
plot([temp ts],zeros(1,length([temp ts])),'k','linewidth', 0.5);
plot(zeros(100,1), linspace(ylims(1),ylims(2), 100), 'k--', 'linewidth', 0.5)
xlabel('time (s)')
ylabel('\Delta[DA]')
legend({'NGL success','NGL fail'},'location','NorthEast')
