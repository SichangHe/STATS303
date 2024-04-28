# STATS 303 Homework 3 Answer

## Q1

Given the following 8 data points:

| Data | Feature 1 | Feature 2 | Label |
|------|-----------|-----------|-------|
|   1  |     1     |     1     |   1   |
|   2  |     1     |     0     |   1   |
|   3  |     1     |     1     |   1   |
|   4  |     1     |     0     |   1   |
|   5  |     0     |     1     |   1   |
|   6  |     0     |     0     |   0   |
|   7  |     0     |     1     |   0   |
|   8  |     0     |     0     |   0   |

### Q1 (a)

Please calculate the initial entropy.

**Ans:**

$$
n=8,d=2\\
X = \begin{bmatrix}
    1 & 1 & 1 & 1 & 0 & 0 & 0 & 0 \\
    1 & 0 & 1 & 0 & 1 & 0 & 1 & 0 \\
\end{bmatrix}^T\in\R^{n\times d}\\
Y = \begin{bmatrix}
    1 & 1 & 1 & 1 & 1 & 0 & 0 & 0
\end{bmatrix}\in\R^n\\
⇒ p(y=1) = \frac{5}{8},\quad p(y=0) = \frac{3}{8}\\
⇒ H(Y) = - \sum_{i=1}^{n} p(y_i) \ln p(y_i)\\=
-\left(p(y=1) \ln p(y=1)+p(y=0) \ln p(y=0)\right)\\=
-\left(\frac{5}{8}\ln\frac{5}{8}+\frac{3}{8}\ln\frac{3}{8}\right)
\approx 0.6615
$$

### Q1 (b)

Please calculate the information gain for setting Feature 2 as a root node.

**Ans**: The information gain for splitting on Feature 2 can be calculated using the formula:

$$
⇒ H(Y|x_2)= \sum_{x} p(x_2 = x) H(Y | x_2)\\=
p(x_2=1)H(Y|x_2=1)+p(x_2=0)H(Y|x_2=0)\\=
$$

$$I(Y|x_2) = H(Y) - H(Y|x_2)$$

- When Feature 2 = 1:

$$
p(x_2 = 1) = \frac{4}{8} = \frac{1}{2},\quad
p(y=1 | x_2 = 1) = \frac{3}{4},\quad
p(y=0 | x_2 = 1) = \frac{1}{4}\\
$$

…

1. Given $\mathcal{X} \in \mathcal{R}$ and $\mathcal{F}$
    is defined as positive class=left half space.
    Please calculate the shattering coefficient $\mathcal{N}(\mathcal{F}, 4)$

1. Please summarize the process of proving generalization bound of shattering
    coefficient.
    You have to clarify how to deal with the true risk and empirical risk with
    infinite class of functions.

1. Given $\mathcal{X} \in \mathcal{R}$ and $\mathcal{F}$ is defined as
    $\{\operatorname{sign}(\operatorname{cost} x) \mid t \in R\}$.
    Please calculate the $V C(\mathcal{F})$
