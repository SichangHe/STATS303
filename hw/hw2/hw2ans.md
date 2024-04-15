# STATS 303 Homework 2 Answer

1. The centering matrix in principle component analysis is denoted as $H$,
can you find the complete form for matrix $B$ where $B=\sum_{i=1}^n H^i$

    **Answer**:
    $$
    H^2=H\Rightarrow H^i=H\ \forall i\in N^+\\
    \Rightarrow B=\sum_{i=1}^n H^i=\sum_{i=1}^n H=nH\\
    H:=\mathbb I_N-\frac{1}{N}\vec1_N{\vec1_N}^T\\
    \Rightarrow B=n\mathbb I_N-\frac{n}{N}\vec1_N{\vec1_N}^T=
    \begin{bmatrix}
        n-\frac{n}{N} & -\frac{n}{N} & \cdots & -\frac{n}{N}\\
        -\frac{n}{N} & n-\frac{n}{N} & \cdots & -\frac{n}{N}\\
        \vdots & \vdots & \ddots & \vdots\\
        -\frac{n}{N} & -\frac{n}{N} & \cdots & n-\frac{n}{N}
    \end{bmatrix}.
    $$

    Assuming that $n$ and $N$ are just different notations for the same value,
    then

    $$
    B=n\mathbb I_N-\vec1_N{\vec1_N}^T=
    \begin{bmatrix}
        n-1 & -1 & \cdots & -1\\
        -1 & n-1 & \cdots & -1\\
        \vdots & \vdots & \ddots & \vdots\\
        -1 & -1 & \cdots & n-1
    \end{bmatrix}.
    $$

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
