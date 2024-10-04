function [wk, niter] = uo_nn_bfgs(w, f, g, epsG, kmax, epsal, kmaxBLS, almax, c1, c2, Xtr, ytr)
    % initialize iteration counter
    k = 1;
    % initialize weights history with the initial weights
    wk = w;
    % initialize inverse Hessian approximation as identity matrix
    H = eye(length(w));
    
    % loop until the gradient norm is less than the tolerance or max iterations reached
    while norm(g(w, Xtr, ytr)) > epsG && k < kmax
        % compute the descent direction using the inverse Hessian approximation
        d = -H * g(w, Xtr, ytr);
        
        % adjust the maximum step size after the first iteration
        if k ~= 1
            almax = 2 * (f(wk(:, k), Xtr, ytr) - f(wk(:, k-1), Xtr, ytr)) / (g(wk(:, k), Xtr, ytr)' * d);
        end
        
        % perform backtracking line search to find the optimal step size
        [al, ~] = uo_BLSNW32bis(f, g, w, d, almax, c1, c2, kmaxBLS, epsal, Xtr, ytr);
        
        % update weights
        w_new = w + al * d;
        
        % compute the difference in weights and gradients
        s = w_new - w;
        y = g(w_new, Xtr, ytr) - g(w, Xtr, ytr);
        
        % update the inverse Hessian approximation using the BFGS formula
        rho = 1 / (y' * s);
        I = eye(length(w));
        H = (I - rho * s * y') * H * (I - rho * y * s') + rho * (s * s');
        
        % update weights
        w = w_new;
        
        % store the updated weights in the history
        wk = [wk w];
        
        % increment iteration counter
        k = k + 1;

        % Print iteration number and loss value
        fprintf('Iteration %d: Loss = %f\n', k, f(w, Xtr, ytr));
    end
    
    % return the number of iterations performed
    niter = k;
end