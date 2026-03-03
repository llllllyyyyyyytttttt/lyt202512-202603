[toc]

## GAS

### GAS论文原文

***“Generalized Autoregressive Score Models with Time-Varying Parameters”**** 

- ***Journal of Time Series Analysis*, 2024**
- Vladimír Holý

**2.4 Dynamics of Time Varying Parameters**

In GAS models, time-varying parameters $f_t$ follow the recursion

$$
f_t=\omega+\sum_{j=1}^P \alpha_j S\left(f_{t-j}\right) \nabla\left(y_{t-j}, f_{t-j}\right)+\sum_{k=1}^Q \varphi_k f_{t-k},
$$

where $\omega$ is the intercept, $\alpha_j$ are the score parameters, $\varphi_k$ are the autoregressive parameters, and $S\left(f_t\right)$ is a scaling function for the score. In the case of a single time-varying parameter, all these quantities are scalar. In the case of multiple time-varying parameters, $\omega$ and $\nabla\left(y_{t-j}, f_{t-j}\right)$ are vectors, $\alpha_j$ and $\varphi_k$ are diagonal matrices, and $S\left(f_t\right)$ is a square matrix. In the majority of empirical studies, it is common practice to set the score order $P$ and the autoregressive order $Q$ to 1 . Furthermore, one of three scaling functions is typically chosen: the unit function, the inverse of the Fisher information, or the square root of the inverse of the Fisher information. When the latter is used, the scaled score has unit variance. However, the choice of the scaling function is not always a straightforward task and is closely tied to the underlying distribution.

The dynamics of the model can be expanded to incorporate exogenous variables as

$$
f_t=\omega+\sum_{i=1}^M \beta_i x_{t i}+\sum_{j=1}^P \alpha_j S\left(f_{t-j}\right) \nabla\left(y_{t-j}, f_{t-j}\right)+\sum_{k=1}^Q \varphi_k f_{t-k},
$$

where $\beta_i$ are the regression parameters associated with the exogenous variables $x_{t i}$. Alternatively, a different model can be obtained by defining the recursion in the fashion of regression models with dynamic errors as

$$
f_t=\omega+\sum_{i=1}^M \beta_i x_{t i}+e_t, \quad e_t=\sum_{j=1}^P \alpha_j S\left(f_{t-j}\right) \nabla\left(y_{t-j}, f_{t-j}\right)+\sum_{k=1}^Q \varphi_k e_{t-k} .
$$

**The key distinction between the two models lies in the impact of exogenous variables on $f_t$. Specifically, in the former model formulation, exogenous variables influence all future parameters through both the autoregressive term and the score term. In the latter model formulation, they affect future parameters only through the score term. When no exogenous variables are included, the two specifications are equivalent, although with differently parameterized intercept. When numerically optimizing parameter values, the latter model exhibits faster convergence, thanks to the dissociation of $\omega$ from $\varphi_k$.**这两个模型的关键区别在于外生变量对财务交易的影响。具体而言，在前一个模型公式中，外生变量通过自回归项和分数项影响所有未来参数。在后一个模型公式中，它们仅通过score项影响未来参数。<u>当不包括外生变量时，这两个规范是等价的，尽管截距参数不同</u>。<u>当数值优化参数值时，由于ω从φk分离，后一种模型表现出更快的收敛。</u>

Other model specifications can be obtained by imposing various restrictions on $\omega, \beta_i, \alpha_j$, or $\varphi_k$. In addition, it is possible to have different orders $P$ and $Q$ for individual parameters when multiple parameters are time-varying. Furthermore, the set of exogenous variables can also vary for different parameters.

The recursive nature of $f_t$ necessitates the initialization of the first few elements $f_1, \ldots, f_{\max \{P, Q\}}$. A sensible approach is to set them to the long-term value,

$$
\bar{f}= \begin{cases}\left(1-\sum_{k=1}^Q \varphi_k\right)^{-1}\left(\omega+\sum_{i=1}^M \beta_i \frac{1}{N} \sum_{t=1}^N x_{t i}\right) & \text { in model (4), } \\ \omega+\sum_{i=1}^M \beta_i \frac{1}{N} \sum_{t=1}^N x_{t i} & \text { in model (5). }\end{cases}
$$

____________

### **GAS 模型里的“时变参数”到底是怎么随时间更新的？**

核心结论是：时变参数 $f_t$是由**常数项 + 外生变量 + “过去参数的惯性” + “过去数据的反馈（score）”**共同决定的。

### GAS模型

因为 $f_t$ 是递归的，必须给初值。

Vladimír Holý(作者)建议：

- **用长期均值作为初始值**

- 系数可以直接使用极大似然估计

##### 原始模型

$f_t=\omega+\sum_{j=1}^P \alpha_j S(f_{t-j}) \nabla(y_{t-j}, f_{t-j})
+\sum_{k=1}^Q \varphi_k f_{t-k}$

- $\omega$:    长期均衡水平（类AR截距），没有新信息时，参数会往这里回归

- $\sum_{j=1}^P \alpha_j S(f_{t-j}) \nabla(y_{t-j}, f_{t-j})$:   score项，根据过去数据的拟合误差，修正参数

  | 符号                        | 含义                                                         |
  | --------------------------- | ------------------------------------------------------------ |
  | $ \nabla(y_{t-j}, f_{t-j})$ | score：对数似然的一阶导数    $\text{score}_t = \frac{\partial \log f(y_t \mid \theta_t)}{\partial \theta_t}$,  score > 0：参数该往上调 score < 0：参数该往下调 score = 0：参数刚好 |
  | $S(\cdot)$                  | 缩放函数（控制尺度）                                         |
  | $\alpha_j$                  | 参数对 score 的敏感度                                        |

- $\sum_{k=1}^Q \varphi_k f_{t-k}$:   自回归项（惯性，AR）
  - $\varphi_k$ 越大 → 参数越平滑
  - $\varphi_k = 0$ → 完全由新数据驱动

- P/Q:

| 符号 | 含义               |
| ---- | ------------------ |
| P    | score 的滞后阶数   |
| Q    | 参数自身的 AR 阶数 |

实务中,**几乎所有应用都设 (P=1, Q=1)**，稳、可解释、好收敛。

###### **缩放函数：**

score 的大小 **依赖于变量尺度和分布**，如果不处理，参数可能：

- 更新过猛（不稳定）
- 或几乎不动

| Scaling               | 含义       | 特点                                  |
| --------------------- | ---------- | ------------------------------------- |
| 单位矩阵              | 不缩放     | 最简单                                |
| Fisher 信息的逆       | 标准化     | 稳健                                  |
| Fisher 信息逆的平方根 | **最常用** | score 方差 = 1【GAS自适应、尺度无关】 |

##### 引入外生变量

###### 模型一：外生变量直接进入 $f_t$

$f_t=\omega+\sum_{i=1}^M \beta_i x_{t i}
+\text{score项}
+\text{AR项}$

即
$$
f_t=\omega+\sum_{i=1}^M \beta_i x_{t i}+\sum_{j=1}^P \alpha_j S\left(f_{t-j}\right) \nabla\left(y_{t-j}, f_{t-j}\right)+\sum_{k=1}^Q \varphi_k f_{t-k}
$$

- 宏观变量 **直接决定当前参数水平**，并且通过 AR 和 score 影响未来所有参数。

- 解释性强

- $\bar{f} = \frac{\omega + \sum \beta_i \bar{x}_i}{1-\sum \varphi_k}$，相当于 AR 模型的无条件期望

- $$
  f_t=\omega+\sum \beta_i x_{ti}
  +\underbrace{\sum \alpha_j S(\cdot)\nabla(\cdot)}_{\text{数据修正}}
  +\underbrace{\sum \varphi_k f_{t-k}}_{\text{惯性}}
  $$

  

###### 模型二：外生变量只影响“误差”

$
f_t=\omega+\sum_{i=1}^M \beta_i x_{t i}+e_t
$

$
e_t=\text{score项}+\text{AR项}
$

即：
$$
f_t=\omega+\sum_{i=1}^M \beta_i x_{t i}+e_t, \quad e_t=\sum_{j=1}^P \alpha_j S\left(f_{t-j}\right) \nabla\left(y_{t-j}, f_{t-j}\right)+\sum_{k=1}^Q \varphi_k e_{t-k}
$$

- 外生变量只影响“当前偏移”/短期偏离，**参数动态主要靠数据反馈（score）**【$x_t$只决定“锚点”，真正让参数动起来的是 score】
- 计算快

- $\bar{f} = \omega + \sum \beta_i \bar{x}_i$
  - $\bar e_t = 0$
  - 保证：
    - 初始阶段不引入人为偏差
    - 参数从“稳态”开始演化

- $$
  f_t=\underbrace{\omega+\sum \beta_i x_{ti}}_{\text{稳态/锚点}}
  
  - \underbrace{e_t}_{\text{动态偏离}}
  $$

- $$
  e_t=\alpha \cdot \text{score} + \varphi e_{t-1}
  $$

  