load('honza.mat')

% days          8x1  time points
% gr         144x1  code of treatment
% vals1      144x8  pozdejsi sloupec z tech dvou moznych
% vals2      144x8  drivejsi sloupec z tech dvou moznych
% vysl       144x3  pro vals1 nafitovane [A B C]

% We want to create box plot by treatment group

dot_plot(vysl(:,1), gr);
xlabel('Group code')
ylabel('Maximal height')

dot_plot(vysl(:,2), gr);
xlabel('Group code')
ylabel('Synchronicity')

dot_plot(-vysl(:,3), gr);
xlabel('Group code')
ylabel('Offset')
