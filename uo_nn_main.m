clear;

% Parameters for dataset generation
num_target = 1;
tr_freq = 0.5;        
tr_p = 250;       
te_q = 250;       
tr_seed = 170643;    
te_seed = 170645;    

% Parameters for optimization
la = 0.1;  % L2 regularization
epsG = 1e-6; % Optimality tolerance
kmax = 10000;  % Stopping criterion

% Linesearch parameters
ils = 3; 
ialmax = 2; 
kmaxBLS = 30; 
epsal = 1e-3;
c1 = 0.01; 
c2 = 0.45;

% Search direction parameters
isd = 3; % 1: GM, 2: BFGS, 3: SGM
icg = 2; % Only for CGM (not used in this problem)
irc = 2; % Same as above
nu = 1.0; % Same as above

% SGM iteration parameters
sg_seed = 565544; 
sg_al0 = 2; 
sg_be = 0.3; 
sg_ga = 0.01;

% SGM stopping condition
sg_emax = kmax; 
sg_ebest = floor(0.01 * sg_emax);

% Run the optimization
t1=clock;
[Xtr,ytr,wo,fo,tr_acc,Xte,yte,te_acc,niter,tex]=uo_nn_solve(num_target,tr_freq,tr_seed,tr_p,te_seed,te_q,la,epsG,kmax,ils,ialmax,kmaxBLS,epsal,c1,c2,isd,sg_al0,sg_be,sg_ga,sg_emax,sg_ebest,sg_seed,icg,irc,nu);
t2=clock;
fprintf(' wall time = %6.1d s.\n', etime(t2,t1));
%


