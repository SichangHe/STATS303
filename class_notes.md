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
