# Pattern Recognition with Single Layer Neural Network (SLNN)

## Objective
The goal of this project is to develop an application that recognizes numbers in a sequence of blurred digits using unconstrained optimization algorithms. You will implement and test different optimization methods for training a Single Layer Neural Network (SLNN) on a digit recognition task.

## Project Structure
The project is divided into three main parts, each requiring the implementation of various optimization techniques.

### Part 1: Pattern Recognition with Gradient and Quasi-Newton Methods (GM and QNM)
- **Task:** Develop the function `uo_nn_solve.m` to perform recognition of target numbers using GM and QNM.
- **Steps:**
  1. Generate training and test datasets using `uo_nn_dataset.m`.
  2. Minimize the loss function using the optimization routines developed in the course (GM and QNM).
  3. Calculate and report training and test accuracy.
  
### Part 2: Pattern Recognition with Stochastic Gradient Method (SGM)
- **Task:** Extend `uo_nn_solve.m` to include the stochastic gradient method (SGM) for training the SLNN.
- **Steps:**
  1. Implement the SGM by selecting minibatches from the training dataset.
  2. Update weights iteratively with SGM using a learning rate schedule.
  3. Evaluate performance based on global and local convergence as well as recognition accuracy.

### Part 3: Performance Study
#### Convergence Study Instructions

##### 1. Study of the Convergence

We will study the global and local convergence of three algorithms in terms of the objective function L̃, using the information from the .csv log file.

###### a) Global Convergence

Analyze the global convergence of each algorithm and examine how this property depends on the regularization parameter λ.

**Tasks:**
- Evaluate how global convergence varies with different λ values for each algorithm
- Determine which algorithm-λ combination provides the best results
- Specifically discuss the application of global convergence conditions to the SGM algorithm

###### b) Local Convergence

**i. Speed Comparison**
Compare the convergence speed of the three algorithms based on:
- Execution time
- Number of iterations required

**ii. λ Dependency Analysis**
- Analyze how convergence speed depends on λ for each algorithm
- Identify any patterns or relationships
- Provide explanations for observed dependencies

**iii. Iteration Time Analysis**
- Calculate and compare the running time per iteration (tex/niter) for each algorithm
- Explain any differences observed between algorithms
- Consider factors that might influence iteration time

###### c) Overall Performance Evaluation

Based on the previous analyses:
1. Discuss the general performance of all three algorithms
2. Consider both local and global convergence observations
3. Justify which λ-algorithm combination is most efficient for minimizing L̃

#### Recognition Accuracy Study

##### Dataset Parameters
We will analyze the recognition accuracy (Accuracy^TE or te_acc) of the SLNN using a more realistic dataset with the following parameters:

```
tr_p = 20000      # Training samples
te_q = tr_p/10    # Test samples
tr_freq = 0.0     # Training frequency
```

##### Training Process
Run the training process for all ten digits using:
1. Three algorithms: GM, QNM, and SGM
2. For each algorithm, use the λ value that showed the best Accuracy^TE in section 1-a) with the smaller dataset (tr_p = 250)

##### Analysis Tasks

###### a) Performance Comparison
Analyze the results to determine:
- If any method clearly outperforms the others in terms of:
  1. Training speed
  2. Recognition accuracy

###### b) Consistency Analysis
Compare the best λ-algorithm combinations between:
1. Maximization of Accuracy^TE (current study)
2. Minimization of L̃ (from study 1-c)

**Analysis Requirements:**
- Determine if the best combinations coincide
- If discrepancies exist:
  - Identify the differences
  - Discuss potential reasons for these discrepancies
  - Consider how the change in dataset size might affect the results

## Datasets
- **Training dataset (`Xtr`, `ytr`)**: Created with `uo_nn_dataset.m` using various target digits.
- **Test dataset (`Xte`, `yte`)**: Generated similarly for evaluating model performance.

## Optimization Routines
- **Gradient Method (GM)**: Use first-order optimization with backtracking line search.
- **Quasi-Newton Method (QNM)**: Utilize second-order methods for faster convergence.
- **Stochastic Gradient Method (SGM)**: Train the model using minibatches of data and a learning rate schedule.

## Evaluation
- **Accuracy Calculation:** After training, report the accuracy on both training and test datasets.
- **Convergence Analysis:** Study the convergence behavior of each method by observing the change in loss and gradient.

## Final Report
- Summarize the development, implementation, and performance of the three methods.
- Include comparisons of recognition accuracy and convergence behavior.
