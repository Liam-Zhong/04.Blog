+++
author = "Liam"
title = '跟着星宇学毫升'
date = 2024-09-30T16:33:11+08:00
math = true 
draft = false
comments = true
description = "机器学习"
+++

$\begin{split}\begin{aligned} w_1 &\leftarrow w_1 - \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \frac{ \partial \ell^{(i)}(w_1, w_2, b) }{\partial w_1} = w_1 - \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}}x_1^{(i)} \left(x_1^{(i)} w_1 + x_2^{(i)} w_2 + b - y^{(i)}\right),\\ w_2 &\leftarrow w_2 - \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \frac{ \partial \ell^{(i)}(w_1, w_2, b) }{\partial w_2} = w_2 - \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}}x_2^{(i)} \left(x_1^{(i)} w_1 + x_2^{(i)} w_2 + b - y^{(i)}\right),\\ b &\leftarrow b - \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \frac{ \partial \ell^{(i)}(w_1, w_2, b) }{\partial b} = b - \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}}\left(x_1^{(i)} w_1 + x_2^{(i)} w_2 + b - y^{(i)}\right). \end{aligned}\end{split}$

超参：$|\mathcal{B}|$、$\eta$

$\begin{split}\begin{aligned} \hat{y}^{(1)} &= x_1^{(1)} w_1 + x_2^{(1)} w_2 + b,\\ \hat{y}^{(2)} &= x_1^{(2)} w_1 + x_2^{(2)} w_2 + b,\\ \hat{y}^{(3)} &= x_1^{(3)} w_1 + x_2^{(3)} w_2 + b. \end{aligned}\end{split}$

$\begin{split}\boldsymbol{\hat{y}} = \begin{bmatrix} \hat{y}^{(1)} \\ \hat{y}^{(2)} \\ \hat{y}^{(3)} \end{bmatrix},\quad \boldsymbol{X} = \begin{bmatrix} x_1^{(1)} & x_2^{(1)} \\ x_1^{(2)} & x_2^{(2)} \\ x_1^{(3)} & x_2^{(3)} \end{bmatrix},\quad \boldsymbol{w} = \begin{bmatrix} w_1 \\ w_2 \end{bmatrix}.\end{split}$

$\boldsymbol{\hat{y}} = \boldsymbol{X} \boldsymbol{w} + b$

$\boldsymbol{\hat{y}} = \boldsymbol{X} \boldsymbol{w} + b,$

$\boldsymbol{\hat{y}} \in \mathbb{R}^{n \times 1}$

$\boldsymbol{\hat{y}} = \boldsymbol{X} \boldsymbol{w} + b$,

其中模型输出$\boldsymbol{\hat{y}} \in \mathbb{R}^{n \times 1}$， 批量数据样本特征$\boldsymbol{X} \in \mathbb{R}^{n \times d}$，权重$\boldsymbol{w} \in \mathbb{R}^{d \times 1}$， 偏差$b \in \mathbb{R}$。相应地，批量数据样本标签$$\boldsymbol{y} \in \mathbb{R}^{n \times 1}$。设模型参数 $ \boldsymbol{\theta} = [w_1, w_2, b]^\top $，我们可以重写损失函数为

$\ell(\boldsymbol{\theta})=\frac{1}{2n}(\boldsymbol{\hat{y}}-\boldsymbol{y})^\top(\boldsymbol{\hat{y}}-\boldsymbol{y}).$









小批量随机梯度下降的迭代步骤为

$\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} - \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \nabla_{\boldsymbol{\theta}} \ell^{(i)}(\boldsymbol{\theta})$,







其中梯度是损失有关3个为标量的模型参数的偏导数组成的向量：

$\begin{split}\nabla_{\boldsymbol{\theta}} \ell^{(i)}(\boldsymbol{\theta})= \begin{bmatrix} \frac{ \partial \ell^{(i)}(w_1, w_2, b) }{\partial w_1} \\ \frac{ \partial \ell^{(i)}(w_1, w_2, b) }{\partial w_2} \\ \frac{ \partial \ell^{(i)}(w_1, w_2, b) }{\partial b} \end{bmatrix} = \begin{bmatrix} x_1^{(i)} (x_1^{(i)} w_1 + x_2^{(i)} w_2 + b - y^{(i)}) \\ x_2^{(i)} (x_1^{(i)} w_1 + x_2^{(i)} w_2 + b - y^{(i)}) \\ x_1^{(i)} w_1 + x_2^{(i)} w_2 + b - y^{(i)} \end{bmatrix} = \begin{bmatrix} x_1^{(i)} \\ x_2^{(i)} \\ 1 \end{bmatrix} (\hat{y}^{(i)} - y^{(i)}).\end{split}$

