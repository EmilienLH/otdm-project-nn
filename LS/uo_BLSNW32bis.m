function [alphas, iout] = uo_BLSNW32bis(f, g, x0, d, alpham, c1, c2, maxiter, eps, Xtr, ytr)
    % Combined line search algorithm based on strong Wolfe and Armijo conditions
    % Inputs:
    % f : objective function
    % g : gradient of the objective function
    % x0 : starting point
    % d : search direction
    % alpham : maximum step size
    % c1, c2 : constants for Wolfe and Armijo conditions
    % maxiter : maximum number of iterations
    % eps : tolerance for stopping
    % Xtr, ytr : training data

    % Initialize variables
    alpha0 = 0;
    alphap = alpha0;
    alphas = alpham;
    iout = 0; % output status
    f0 = f(x0, Xtr, ytr);
    g0 = g(x0, Xtr, ytr);
    gx0 = g0' * d; % directional derivative
    i = 1;

    while i < maxiter
        % Compute the new point and function value
        x_new = x0 + alphas * d;
        fnew = f(x_new, Xtr, ytr);
        gnew = g(x_new, Xtr, ytr);
        gxnew = gnew' * d;

        % Check Armijo condition (sufficient decrease)
        if fnew > f0 + c1 * alphas * gx0 || (i > 1 && fnew >= f(x0 + alphap * d, Xtr, ytr))
            % Perform zoom to refine alpha
            [alphas, iout_zoom] = zoom(f, g, x0, d, alphap, alphas, c1, c2, eps, Xtr, ytr);
            if iout_zoom == 2
                iout = 2;
            end
            return;
        end

        % Check curvature condition (strong Wolfe)
        if abs(gxnew) <= -c2 * gx0
            return;
        end

        % Check if gradient changed direction, refine step size via zoom
        if gxnew >= 0
            [alphas, iout_zoom] = zoom(f, g, x0, d, alphas, alphap, c1, c2, eps, Xtr, ytr);
            if iout_zoom == 2
                iout = 2;
            end
            return;
        end

        % Update alpha and loop
        alphap = alphas;
        alphas = alphas + (alpham - alphas) * rand(1); % Random step size
        i = i + 1;

        % Stopping condition
        if abs(alphap - alphas) < eps
            iout = 2; % Stuck, alpha_[i] = alpha^[i-1]
            return;
        end
    end

    % If max iterations are reached
    if i == maxiter
        iout = 1; % Max iterations reached
    end
end

% Helper function for zooming
function [alphas, iout_zoom] = zoom(f, g, x0, d, alpha_low, alpha_high, c1, c2, eps, Xtr, ytr)
    % Zoom function to refine alpha when Wolfe conditions are not met
    iout_zoom = 0;
    while abs(alpha_high - alpha_low) > eps
        alphas = (alpha_low + alpha_high) / 2; % Midpoint
        x_new = x0 + alphas * d;
        fnew = f(x_new, Xtr, ytr);
        gnew = g(x_new, Xtr, ytr);
        gxnew = gnew' * d;

        % Check Armijo condition
        if fnew > f(x0, Xtr, ytr) + c1 * alphas * (g(x0, Xtr, ytr)' * d) || fnew >= f(x0 + alpha_low * d, Xtr, ytr)
            alpha_high = alphas;
        else
            % Check curvature condition
            if abs(gxnew) <= -c2 * (g(x0, Xtr, ytr)' * d)
                return;
            end
            if gxnew * d >= 0
                alpha_high = alphas;
            else
                alpha_low = alphas;
            end
        end
    end
    alphas = (alpha_low + alpha_high) / 2; % Final refined step size
end