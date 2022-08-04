[num txt] = xlsread('results/statistics.xlsx');  % with treatment group code

%%
gr = num(:, 2);  % this is treatment group code
vals = num(:, 3:end);  % heights
casy = [2021 4 20 7 0 0; ...
    2021 4 27 7 0 0; ...
    2021 5 4 7 0 0; ...
    2021 5 12 7 0 0; ...
    2021 6 1 7 0 0; ...
    2021 6 8 7 0 0; ...
    2021 6 15 7 0 0; ...
    2021 6 24 7 0 0];
casy1 = datenum(casy);
days = casy1 - casy1(1);
days = round(days);
vals1 = vals; vals1(:, 2)=[];  % second measurement
vals2 = vals; vals2(:, 3)=[];  % first measurement
% [days vals1] is second measurement from two calibration measurements
% [days vals2] is first measurementn from two calibration measurements

save('honza.mat','days','vals1','vals2','gr');

%% Load prepared data

load('honza.mat') % heights, id, days

%% Fit coeffs

[mm nn] = size(vals1);
vysl = nan(mm, 3); % allocate A B C

for i=1:1:mm

    yy = vals1(i, :); % fit logist
    %beta0 = [max(yy); 10/max(days); -5]; % initial guess
    beta0 = [1; 0.1; -5];
    yy_poc = logist(beta0, days);
    beta = nlinfit(days, yy, @logist, beta0);
    vysl(i, :) = beta';

%     h = figure(1);
%     plot(days,yy,'ko','MarkerFaceColor','g')
%     hold on
%     plot(days,yy_poc,'b')
%     yy_fit = logist(beta,days);
%     plot(days,yy_fit,'r')
%     legend('data','init-guess','nln-fit','Location','NorthWest')
%     xlabel('days from start')
%     ylabel('height [m]')
%     lab = ['i = ' num2str(i) ', Pars = ' num2str(beta')];
%     title(lab)
%     jm = ['r' num2str(id(i,1),'%02g') 's' num2str(id(i,2),'%02g') '.png'];
%     saveas(h,jm,'png');
%     hold off

end

save('honza.mat', 'days', 'vals1', 'vals2', 'gr', 'vysl');

% days          8x1  time points
% gr         144x1  code of treatment
% vals1      144x8  two calibration measurements - the second one
% vals2      144x8  two calibration measurements - the first one
% vysl       144x3  [A B C] fit for vals1
