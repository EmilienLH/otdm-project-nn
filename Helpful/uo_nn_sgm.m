% Stochastic Gradient Method (SGM) function
function [wk, niter] = uo_nn_sgm(w0, L_func, grad_L_func, Xtr, ytr, Xte, yte, alpha0_sg, beta_sg, gamma_sg, e_max_sg, e_best_sg)
    % Initialize parameters
    p = size(Xtr, 2);                  % number of columns in Xtr
    m = floor(gamma_sg * p);            % mini-batch size
    k_e_sg = floor(p / m);                % iterations per epoch
    k_max_sg = e_max_sg * k_e_sg;
    k_sg = floor(abs(beta_sg * k_max_sg));
    alpha_sg = 0.01 * alpha0_sg;        % as specified in the algorithm
    
    % Initialize variables
    e = 0;                              % epoch counter
    s = 0;                              % early stopping counter
    L_te_best = inf;
    k = 0;                              % iteration counter
    w = w0;                             % current weights
    w_star = w0;                        % best weights
    wk = w0;                            % history of weights
    
    % Main loop
    while e <= e_max_sg && s < e_best_sg
        % Generate permutation for this epoch
        P = randperm(p);
        
        % Mini-batch loop
        for i = 0:floor(p / m) - 1
            % Select current mini-batch S
            start_idx = i * m + 1;
            end_idx = min((i + 1) * m, p);
            S = P(start_idx:end_idx);
            
            % Extract X_s and y_s corresponding to mini-batch S
            X_s = Xtr(:, S);
            y_s = ytr(:, S);
            
            % Compute gradient
            d_k = -grad_L_func(w, X_s, y_s);
            
            % Update learning rate
            if k <= k_sg
                alpha_k = (1 - k / k_sg) * alpha0_sg + (k / k_sg) * alpha_sg;
            else
                alpha_k = alpha_sg;
            end
            
            % Update weights and iteration counter
            w = w + alpha_k * d_k;
            k = k + 1;
            
            % Store the updated weights in the history
            wk = [wk w];
        end
        
        % Evaluate on test set after each epoch
        L_te = L_func(w, Xte, yte);
        
        % Update best model if necessary
        if L_te < L_te_best
            L_te_best = L_te;
            w_star = w;
            s = 0;
        else
            s = s + 1;
        end
        
        e = e + 1;
    end
    
    fprintf('\tDone, %d epochs.\n', e);
    % Return the number of iterations performed
    niter = k;
end