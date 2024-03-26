# STATS 303

SGD > GD for online learning

SGD problem: mini batch too big

gated recurrent unit (GRU): a RNN

principal component analysis (PCA): autoencoder, dimensionality reduction

## Bayesian decision theory

loss $\lambda_{ik}$: take action $\alpha_i$ when sample belong to $C_k$

goal: minimize risk

$$
R(\alpha_i|X)=\sum_{k=1}^K\lambda_{ik}P(C_k|X)
$$

for 0-1 loss, $R(\alpha_i|X)=1-P(C_i|X)$

reject class: a $K+1$th class w/ fixed loss $\lambda\in(0,1)$

- $⇒ R(a_{K+1}|X)\equiv\lambda$
- reject when $\min_{i=1\ldots K}R(\alpha_i|X)>\lambda$

or, maximize discriminant function $g_i(x),i=1\ldots K$

maximum likelihood estimator (MLE)

parametric (distribution based on known parameters) vs non-parametric

## linear regression

$$
X\in\R^{N × P}\\
Y\in\R^{N × 1}\\
L(W)=\Vert W^TX-Y\Vert^2=W^TX^TXW-2W^TX^TY+Y^TY\\
\argmin_WL(W) ⇒ \frac{\partial L(W)}{\partial W}=0\\
⇒ 2X^TXW-2X^TY=0\\
⇒ \hat W=(X^TX)^{-1}X^TY
$$

### regularization

$$
\argmin_W\left[
    L(W) + \lambda P(W)
\right] ⇒ \hat W=(X^TX+\lambda I)^{-1}X^TY
$$

assuming $P(W)=\Vert W\Vert_2$

$X^TX+\lambda I$ invertible, proof by positive definite

## K-nearest neighbors (KNN)

need to try different $K$

## support vector machine (SVM)

### hard-margin binary SVM

$y=-1,1$

objective:

$$
\min_{W,b}\frac{1}{2}\Vert W\Vert^2
\text{ s.t. } y_i(W^TX_i+b)\ge1
$$

apply Lagrange multiplier:

$$
L(W,b,\lambda)=
    \frac{1}{2}\Vert W\Vert^2+\sum_{i=1}^N\lambda_i(1-y_i(W^TX_i+b))\\
⇒ \min_{W,b}\max_{\lambda_i\ge0}L(W,b,\lambda)\\
⇒ \max_{\lambda_i\ge0}\min_{W,b}L(W,b,\lambda)
$$

$$
\frac{\partial L}{\partial W}=\frac{\partial L}{\partial b}=0\\
⇒ \hat W=\sum_{i=1}^N\lambda_iy_iX_i,\quad\sum_{i=1}^N\lambda_iy_i=0\\
⇒ \hat L=-\frac{1}{2}\Vert\hat W\Vert^2+\sum_{i=1}^N\lambda_i=
    -\frac{1}{2}\sum_{i=1}^N\sum_{j=1}^N\lambda_i\lambda_jy_iy_jX_i^TX_j+
    \sum_{i=1}^N\lambda_i
$$

final objective

$$
\argmax_{\lambda_i\ge0}\hat L
$$

solution: sequential minimal optimization (SMO)

- fix all but 2 $\lambda_i$, and iterate
- 2 variable because $\sum_{i=1}^N\lambda_iy_i=0$
