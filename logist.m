function y = logist(beta, t)

% logisticka rustova fce
% t = time vector
A = beta(1);
B = beta(2);
C = beta(3);

pom = (B*t + C)';
y = A./(1 + exp(-pom));
