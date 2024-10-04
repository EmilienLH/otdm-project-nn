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
- **Task:** Perform a computational study comparing the performance of GM, QNM, and SGM.
- **Steps:**
  1. Analyze global and local convergence for each method.
  2. Compare recognition accuracy between the methods.
  3. Write a report summarizing the performance of all methods.

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
