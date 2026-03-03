[toc]

## B-splines

B-splines **不适合刻画突发性冲击**（如金融危机的瞬时跳变），因此本文将其与 GAS 模型进行对比，以检验不同参数演化机制下结论的稳健性。

### 原文

**==Dynamic survival models with varying coeﬃcients for credit risks.==** 
**==Viani Biatat Djeundje, Jonathan Crook ∗==**

2.2. Flexible B-splines specification

Although the family (12) is broad enough to handle some complex forms of baseline and varying coefficients, it can suffer from the global dependence of these functions on local properties of the data (De Boor, 1978). In other words, a given month can exert an unexpected influence on remote parts of the fitted $\hat{\beta}_j^{(1)}(\tau)$, and such behaviour can potentially lead to unstable predictions with poor interpolation properties, as illustrated in Djeundje (2011).

A more attractive approach is to express the baselines and varying coefficients using a basis of splines. Such bases have been used extensively in the literature to model complex variabilities; this includes radial basis, backward and forward truncated lines (Djeundje, 2016; Ruppert, Carroll, \& Wand, 2003), as well as B-splines (Brown, Ibrahim, \& DeGruttola, 2005; Eilers \& Marx, 1996). A B-spline can be described as a combination of truncated polynomials. An illustration of B-splines is shown on Fig. 1. Each B-spline has a compact support and this makes them numerically advantageous over other spline bases. For a complete description of B-splines, we refer the reader to De Boor (1978) or Eilers and Marx (1996). We use cubic B-spline basis in this paper; some motivations of this preference are discussed by Green and Silverman (1995).

In terms of B-splines, the $j$ th component $\beta_j^{(1)}$ of the coefficients vector $\boldsymbol{\beta}^{(1)}$ in Eq. (9) takes the form

$$
\beta_j^{(1)}(\tau)=\sum_r \mathcal{B}_{j, r}(\tau) \phi_{j, r}
$$

where $\mathcal{B}_{j, r}(\tau)$ are cubic B-spline functions at time point $\tau$, and $\phi_{j, r}$ are unknown splines coefficients to be estimated. The baseline can be expressed in a similar form, yet with different coefficients. ${ }^4$

In the rest of the paper, we use the term splines specification whenever the baseline and varying coefficients are expressed in terms of B-splines as in Eq. (14). Under this specification, extra pseudo covariates can be computed as in Section 2.1, but with the $\beta_j^{(1)}(\tau)$ given by (14), and all the parameters (including the splines coefficients) can then be jointly estimated by maximising the likelihood defined in (7) using standard packages for regression models.

When fitting models using B-splines however, an important point to address is the number and positions of the knots. Indeed, this can have a detrimental impact on the values and shapes of the fitted varying coefficients. For example, at one extreme, using too many splines can lead to over-fitting; at the other end, using an insufficient number of splines or poor knot locations can yield a model that does not fit the data well. This can negatively affect the predictive performance of the model. In the literature, there are two major approaches to avoid this problem.

One approach is to cover the data range with a sufficiently large number of B-splines and then penalise the roughness in adjacent spline coefficients to achieve smoothness (Eilers \& Marx, 1996; Wood, 2006). With this approach, smoothing parameters are introduced (one for each varying coefficient) and used to tune the amount of smoothing. Optimal values of these smoothing parameters must be chosen carefully because large (small) values can lead to under (over) fitting. In practice, these parameters can be selected via information criteria (or via MCMC simulations especially in the context of a large number of smoothing parameters).

An alternative approach is to carefully and parsimoniously select the number of splines and knot positions; see Friedman and Silverman (1989). In this work for example, we considered various scenarios separately for each varying coefficient (including equispaced and irregular knot spacing) and selected scenarios corresponding to lower values of the Akaike Information Criteria; see (16). ${ }^5$

All models presented in this paper were implemented using SAS software. But there are functions in other standard statistical software that can be used to estimate these models as well. For instance, varying coefficient models expressed in terms of Bsplines as in Eq. (14) can be seen as extension of GAMs (Eilers \& Marx, 2002; Hastie \& Tibshirani, 1990); thus, packages developped for GAMs such as mgcv (Wood, 2006; 2016) or R - INLA (Rue, Martino, \& Chopin, 2009) in R can be adapted to fit some of the varying coefficients models described in this paper.

### B-splines

**Flexible B-splines specification** 是一种将模型参数表示为时间（或宏观状态）的平滑函数的方法，通过 B 样条基函数线性组合刻画参数的连续变化，从而在不引入随机状态方程的情况下实现参数的时变建模。

- 避免**全局依赖问题**

**信用风险参数的变化主要来自结构性变化（周期、制度），不是高频随机冲击**

特别适合：

- 宏观周期研究

- 长样本（10–20 年）

  

  **时间变化的参数 = 一堆已知的平滑基函数 × 待估系数**

$$
\beta_t = \beta(t)\\\
\beta(t) = \sum_{k=1}^K \theta_k B_k(t)\\\
传统模型：\beta = \text{常数}
$$



#### $B_k(t)$

- B-spline 基函数
- 性质：
  - 局部支撑（local support）——某个月的数据 **只会影响附近时间的参数形状**
  - 高度平滑（连续可导）
  - 数值稳定

##### Flexible B-splines

- 结点（knots）可以很多——控制**变化速度**

- 平滑惩罚（penalty）$\lambda \sum_k (\Delta^2 \theta_k)^2
  $

  - 不要让相邻 spline 系数变化太剧烈
  - 防止过拟合

  

- 可扩展到“宏观状态驱动”

  - $\beta_t = \sum_k \theta_k B_k(\text{GDP}_t)$

  

- 多参数同时变化
  - $\text{logit}(PD_t) = \beta_0(t) + \beta_1(t) X_{it}$;
  - 每个 $\beta_j(t)$ 都有自己的 spline。



##### **结点数量和位置**

- 结点过多 → 过拟合；
- 结点过少 → 欠拟合；
- 结点位置不合理 → 参数形状失真。

###### 路径一：多 spline + 平滑惩罚（P-splines）

- 思路：
  - spline 设得**多一点**
  - 用惩罚项控制平滑度

$$
\lambda \sum (\Delta^2 \phi_r)^2
$$

- $\lambda$：平滑参数
- 大 → 更平滑
- 小 → 更灵活

- $\lambda$平滑参数可以用：

  - AIC / BIC

  - MCMC

  - GCV

###### 路径二：少 spline + 精选 knot

- 事先设计几种 knot 方案：
  - 等距
  - 不等距
- 每种都估计
- 用 **AIC 选最优**

 【作者这篇文章 **用的是这一条**(3次样条)】