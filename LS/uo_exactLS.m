% exact line search for quadratic functions
function alpha = uo_exactLS(~, d)
    Q = [4, 0; 0, 1];
    % compute alpha 
    alpha = (d' * d) / (d' * Q * d);
end