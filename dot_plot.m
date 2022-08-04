function dot_plot(vals, gr)

% vals is value to show
% gr is group

skup = unique(gr);
nn = length(skup);
prost = 1:1:nn;
barvy = 'rgbcmykrgbcmykrgbcmykrgbcmykrgbcmykrgbcmykrgbcmykrgbcmykrgbcmykrgbcmykrgbcmykrgbcmyk';
hold on

for i=1:1:nn
    
    where = find(gr == skup(i));  % group i
    yy = vals(where);
    xx = prost(i) + 0.1 * randn(size(yy));

    plot(xx, yy, 'ko', 'MarkerFaceColor', barvy(i))
            
end

set(gca, 'XTick', prost);
set(gca, 'XTickLabel', skup);
