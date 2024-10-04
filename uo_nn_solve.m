% File: uo_nn_solve.m, Project: uo_nn_v40 for OTDM course @ UPC Barcelona
% Author : Emilien L'Haridon
function [Xtr,ytr,wo,fo,tr_acc,Xte,yte,te_acc,niter,tex]=uo_nn_solve(num_target,tr_freq,tr_seed,tr_p,te_seed,te_q,la,epsG,kmax,ils,ialmax,kmaxBLS,epsal,c1,c2,isd,sg_al0,sg_be,sg_ga,sg_emax,sg_ebest,sg_seed,icg,irc,nu) %#ok<INUSD>
    % Input parameters:
    %
    % ========== DATASET GENERATION ==========
    % num_target : set of digits to be identified.
    %    tr_freq : frequency of the digits target in the data set.
    %    tr_seed : seed for the training set random generation.
    %       tr_p : size of the training set.
    %    te_seed : seed for the test set random generation.
    %       te_q : size of the test set.


    % ========== REGULARIZATION ==========
    %         la : coefficient lambda of the decay factor.


    % ========== OPTIMIZATION CRITERIA ==========
    %       epsG : optimality tolerance.
    %       kmax : maximum number of iterations.


    % ========== LINE SEARCH ==========
    %        ils : line search (1 if exact, 2 if uo_BLS, 3 if uo_BLSNW32)
    %     ialmax :  formula for the maximum step lenght (1 or 2).
    %    kmaxBLS : maximum number of iterations of the uo_BLSNW32.
    %      epsal : minimum progress in alpha, algorithm up_BLSNW32
    %      c1,c2 : (WC) parameters.


    % ========== OPTIMIZATION ALGORITHM ==========
    %        isd : optimization algorithm.


    % ========== STOCHASTIC GRADIENT ==========
    %     sg_al0 : \alpha^{SG}_0.
    %      sg_be : \beta^{SG}.
    %      sg_ga : \gamma^{SG}.
    %    sg_emax : e^{SGÇ_{max}.
    %   sg_ebest : e^{SG}_{best}.
    %    sg_seed : seed for the first random permutation of the SG.


    % ========== CONJUGATE GRADIENT ==========
    %        icg : if 1 : CGM-FR; if 2, CGM-PR+      (useless in this project).
    %        irc : re-starting condition for the CGM (useless in this project).
    %         nu : parameter of the RC2 for the CGM  (useless in this project).
    %
    % Output parameters:
    %
    %    Xtr : X^{TR}.
    %    ytr : y^{TR}.
    %     wo : w^*.
    %     fo : {\tilde L}^*.
    % tr_acc : Accuracy^{TR}.
    %    Xte : X^{TE}.
    %    yte : y^{TE}.
    % te_acc : Accuracy^{TE}.
    %  niter : total number of iterations.
    %    tex : total running time (see "tic" "toc" Matlab commands).
    tic
    fprintf('[uo_nn_solve] :::::::::::::::::::::::::::::::::::::::::::::::::::\n');
    fprintf('[uo_nn_solve] Pattern recognition with neural networks.\n');
    fprintf('[uo_nn_solve] %s\n', datetime);
    fprintf('[uo_nn_solve] :::::::::::::::::::::::::::::::::::::::::::::::::::\n');

    % Make a pretty print of the input parameters
    fprintf('[uo_nn_solve] Input parameters:\n');
    fprintf('\tTraining set parameters:\n');
    fprintf('\t \t Target number = %d\n', num_target);
    fprintf('\t \tTraining frequency = %f\n', tr_freq);
    fprintf('\t \tTraining set seed = %d\n', tr_seed);
    fprintf('\t \tTraining set size = %d\n', tr_p);
    fprintf('\t Test set parameters:\n');
    fprintf('\t \tTest set seed = %d\n', te_seed);
    fprintf('\t \tTest set size = %d\n', te_q);
    fprintf('\t Regularization parameter:\n');
    fprintf('\t \tLambda = %f\n', la);
    fprintf('\t Optimization criteria:\n');
    fprintf('\t \tOptimality tolerance = %f\n', epsG);
    fprintf('\t \tMaximum number of iterations = %d\n', kmax);
    if isd == 1
        fprintf('\t Optimization algorithm: Gradient Method\n');
    elseif isd == 2
        fprintf('\t Optimization algorithm: BFGS Method\n');
    elseif isd == 3
        fprintf('\t Optimization algorithm: Stochastic Gradient Method\n');
    end
    fprintf('\t Line search parameters:\n');
    fprintf('\t \tLine search = %d\n', ils);
    fprintf('\t \tMaximum step length formula = %d\n', ialmax);
    fprintf('\t \tMaximum number of iterations of the line search = %d\n', kmaxBLS);
    fprintf('\t \tMinimum progress in alpha = %f\n', epsal);
    fprintf('\t \tC1 = %f\n', c1);
    fprintf('\t \tC2 = %f\n', c2);

    % Generate training data set
    fprintf('[uo_nn_solve] Training data set generation.\n');
    [Xtr, ytr] = uo_nn_dataset(tr_seed, tr_p, num_target, tr_freq);

    % Generate test data set
    fprintf('[uo_nn_solve] Test data set generation.\n');
    te_freq = 0;
    [Xte, yte] = uo_nn_dataset(te_seed, te_q, num_target, te_freq);

    fprintf('[uo_nn_solve] Optimization \n');

    % clear the wk and niter variables
    clear wk niter;

    % Initialize weights
    w = randn(size(Xtr,1),1);

    % Define activation function
    sig = @(X) 1./(1+exp(-X)); 
    y = @(X,w) sig(w'*sig(X));

    % Define loss function and gradient
    L  = @(w,Xtr,ytr) (norm(y(Xtr,w)-ytr)^2)/size(ytr,2)+ (la*norm(w)^2)/2;
    gL = @(w,Xtr,ytr) (2*sig(Xtr)*((y(Xtr,w)-ytr).*y(Xtr,w).*(1-y(Xtr,w)))')/size(ytr,2)+la*w;

    % Optimization algorithm
    if isd == 1 % Gradient Method (GM)
        fprintf('[uo_nn_solve] Gradient Method \n');
        [wk, niter] = uo_nn_gm(w, L, gL, epsG, kmax, epsal, kmaxBLS, ialmax, c1, c2, Xtr, ytr);
    elseif isd == 2 % BFSG Method (BFGS)
        fprintf('[uo_nn_solve] BFGS Method \n');
        [wk, niter] = uo_nn_bfgs(w, L, gL, epsG, kmax, epsal, kmaxBLS, ialmax, c1, c2, Xtr, ytr);
    elseif isd == 3 % Stochastic Gradient Method (SGM)
        fprintf('[uo_nn_solve] Stochastic Gradient Method \n');
        
    end 

    fprintf('[uo_nn_solve] Optimization done, in %d iterations.\n', niter);
    
    % Get the final weights
    kmaxOpt = size(wk,2);
    w_gm = wk(:,kmaxOpt);

    % Calculate Training Accuracy
    fprintf('[uo_nn_solve] Training Accuracy.\n');
    tr_acc = uo_nn_accuracy(Xtr, ytr, w_gm);
    % Print tr_acc and tr_acc_nn to compare the two functions
    fprintf('[uo_nn_solve] Training Accuracy (NN): %f\n', tr_acc);

    % Calculate Test Accuracy
    fprintf('[uo_nn_solve] Test Accuracy.\n');
    te_acc = uo_nn_accuracy(Xte, yte, w_gm);
    % Print te_acc and te_acc_nn to compare the two functions
    fprintf('[uo_nn_solve] Test Accuracy (NN): %f\n', te_acc);
    % uo_nn_Xyplot(Xte, yte, w_gm);
    wo = w_gm;
    fo = L(wo, Xtr, ytr);
    tex = toc;
    

    fprintf('[uo_nn_solve] :::::::::::::::::::::::::::::::::::::::::::::::::::\n');
end