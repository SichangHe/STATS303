# STATS 303 Homework 2

1. The centering matrix in principle component analysis is denoted as $H$,
can you find the complete form for matrix $B$ where $B=\sum_{i=1}^n H^i$
1. Given 5 points: $(-1,0),(0,0),(2,1),(0,1),(-1,-2)$.

    - Please numerically compute the result after applying principle component
    analysis to reduce these data to 1 dimension.
    - Can you code the whole process and visualize the result?

1. Defining the kernel function as $k\left(x_i,
x_j\right)$ which indicates the inner product for $x_i$ and $x_j$ after the
mapping.

    - Please write down the general form of lagrange multiplier objective
    function for kernel hard-margin SVM.
    - Assuming that we are going to apply SMO algorithm to solve it,
    by fixing $\lambda_3, . . \lambda_N$,
    please write down the detailed process and show the final form of the
    objective function in terms of $\lambda_1$

1. If we treat Gibbs sampling as a special case of MH method,
what's the formula for the acceptance rate?
Please simplify the function of acceptance rate.
1. Please implement the MH algorithm to select 10000 samples from an exponential
distribution:

    $$
    \pi(x)=e^{-x}(x \geq 0) .
    $$

    The initial $x=3$ and please plot the histogram of values of $x$ visited by
    MH algorithm.
