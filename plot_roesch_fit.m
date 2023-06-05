function [] = plot_roesch_fit( ppress_data, ppress_data_ses, RTs_means, RTs_ses, ppress, E_tau, DA0, DA1, DAsucc, DAfail, taus )

%% plot summary measures of Roesch data and fits
% ppress_data: empirical probabilities of pressing for each trial type
% ppress_data_ses: associated empirical standard errors
% RTs_means: empirical RTs
% RTs_ses: associated empirical standard errors
% ppress: model P(press)
% E_tau: model expected latencies
% DA0: DA, no intervention
% DA1: DA, intervention
% DAsucc: DA, success
% DAfail: DA, fail
% taus: all possible latencies

%% colour scheme
col_f = [.3 .2 .7];
col_n = [.9 .8 .2];
col_s = [.6 .1 .1];
colMat = [col_f;col_n;col_s];

ttlabels = {'rew','neu','shk'};
msize = 30;

T = 10;
dt = 1e-2;
ts = 0:dt:T;
nts = length(ts);
dtau = taus(2)-taus(1);
maxDA = 8;
minDA = -2;
minDAtrace = -1;
maxDAtrace = 4;

figure 
hold on
b=bar(ppress_data);
set(gca,'ylim',[0 1])
b.FaceColor = 'flat';
b.CData = colMat;
errorbar(1:3,ppress_data,ppress_data_ses,'k','linestyle','none')
plot(ppress,'k.','markerSize',msize)
ylabel('Pr(press)')
set(gca,'xtick',1:3,'xlim',[.3 3.7],'ylim',[0 1],'ytick',0:.5:1,'XTickLabel',ttlabels)
% set(gcf,'position',[1220         526         400         324])

figure
hold on
b = bar(RTs_means);
set(gca,'ylim',[0 4],'ytick',0:1:4)
b.FaceColor = 'flat';
b.CData = colMat;
errorbar(1:3,RTs_means,RTs_ses,'k','linestyle','none')
plot(E_tau,'k.','markerSize',msize)
ylabel('RT (s)')
set(gca,'xtick',1:3,'xlim',[.3 3.7],'XTickLabel',ttlabels)
% set(gcf,'position',[1220         526         400         324])

figure;
hold on
b = bar(1:3,DA0);
b.FaceColor = 'flat';
b.CData = colMat;
b2.FaceColor = 'flat';
b2.CData = colMat(end,:);
ylabel('\delta_\chi')
set(gca,'xtick',1:3,'xlim',[.3 3.7],'XTickLabel',ttlabels,'ylim',[minDA maxDA],'ytick',minDA:2:maxDA,'box','off')
% set(gcf,'position',[1220         526         400         324])

figure;
hold on
b = bar(1:3,DA1);
b.FaceColor = 'flat';
b.CData = colMat;
ylabel('\delta_\chi')
set(gca,'xtick',1:3,'xlim',[.3 3.7],'XTickLabel',ttlabels,'ylim',[minDA maxDA],'ytick',minDA:2:maxDA,'box','off')
% set(gcf,'position',[1220         526         400         324])

figure
hold on
dt = ts(2)-ts(1);
tt = 2/dt;
ts_temp = [[-2:dt:-dt ts (ts(end)+dt):dt:12]];
h = plot(ts_temp, [zeros(3,tt) DAsucc(:,1:(nts+tt))],'linewidth',2);
xlabel('time (s)')
ylabel('\Delta[DA]')
set(h, {'color'}, num2cell([colMat], 2), {'linestyle'}, {'-';'-';'-'});
set(gca,'box','off','xlim',[-2 12],'ylim',[minDAtrace maxDAtrace],'xtick',0:5:10,'ytick',minDAtrace:1:maxDAtrace)
plot(zeros(1,100),linspace(minDAtrace,maxDAtrace),'k--')
plot(ones(1,100).*5,linspace(minDAtrace,maxDAtrace),'k--')
% set(gcf,'position',[1220         526         400         324])

figure
hold on
dt = ts(2)-ts(1);
tt = 2/dt;
ts_temp = [[-2:dt:-dt ts (ts(end)+dt):dt:12]];
DAsucc_temp = [zeros(1,tt) DAsucc(end,1:(nts+tt))];
DAfail_temp = [zeros(1,tt) DAfail(end,1:(nts+tt))];
h = plot(ts_temp,[DAsucc_temp;DAfail_temp],'linewidth',2);
xlabel('time (s)')
ylabel('\Delta[DA]')
set(h, {'color'}, {col_s;col_s}, {'linestyle'},{'-';'--'})
legend({'press','no press'}, 'AutoUpdate', 'off')
plot(zeros(1,100),linspace(minDAtrace,maxDAtrace),'k--')
plot(ones(1,100).*5,linspace(minDAtrace,maxDAtrace),'k--')
set(gca,'box','off','xlim',[-2 12],'ylim',[minDAtrace maxDAtrace],'xtick',0:5:10,'ytick',minDAtrace:1:maxDAtrace)
% set(gcf,'position',[1220         526         400         324])
