# Batch-Context Feature Modulation for Domain Generalization

[Paper Link](https://doi.org/10.48550/arXiv.2504.03064)

## 核心想法

这篇文章提出的 CASA, Context-Aware Self-Adaptation, 是一个面向 domain generalization 的轻量自适应机制。它的核心不是重新设计一个复杂 backbone, 而是利用当前 mini-batch 的特征均值作为 context, 在特征层面做一个小幅、动态的调制。

更直白地说, 模型在测试时虽然不能看到目标域标签, 也不能真正 fine-tune, 但当前 batch 本身已经透露了一点分布信息。CASA 试图把这点信息用起来:

\[
\text{mini-batch feature mean}
\rightarrow
\text{context}
\rightarrow
\text{feature modulation}
\rightarrow
\text{better DG}
\]

所以它所谓的 self-adaptation 不必理解得太神秘。这里的 "self" 主要指模型不依赖额外标签或人工调参, 而是在前向过程中根据当前 batch 的统计信息自动调节中间特征。

## 方法结构

CASA 的训练是两阶段的。

第一阶段, 从已有源域中构造多个 meta-source / meta-target 组合。每个 meta-source 域训练一个普通分类模型:

\[
x \rightarrow f_i(x) \rightarrow h_i(f_i(x))
\]

其中 \(f_i\) 是特征提取器, \(h_i\) 是分类器。实验中使用的是 ResNet-50 backbone。

第二阶段, 在 \(f_i\) 和 \(h_i\) 之间插入一个 adaptation module \(g\):

\[
x \rightarrow f_i(x) \rightarrow g(f_i(x), C) \rightarrow h_i
\]

这里的 \(C\) 是 context, 具体就是当前 mini-batch 特征均值。训练 \(g\) 时, 作者希望它既能让 meta-source 模型适配 meta-target 域, 又不要破坏模型在原 meta-source 域上的预测能力。因此 loss 包含两个部分:

\[
L = L_{\text{adapt}} + \lambda L_{\text{preserve}}
\]

\(L_{\text{adapt}}\) 让模型适配 meta-target, \(L_{\text{preserve}}\) 保持 meta-source 上的性能。

在小规模数据集 PACS, VLCS 和 OfficeHome 上, 第二阶段固定第一阶段训练好的 \(f_i\) 和 \(h_i\), 只更新 \(g\)。在更复杂的 TerraIncognita 和 DomainNet 上, 作者认为 6 参数的 \(g\) 可能不够表达复杂变化, 所以训练 \(g\) 时也解冻并微调分类器 \(h_i\), 但没有解冻特征提取器 \(f_i\)。

测试时, CASA 不使用测试标签, 只使用当前 test batch 的特征均值作为 context。最后多个 adapted meta-source models 做 ensemble, 平均预测概率得到最终结果。

## CaFiLM 模块

CASA 里真正的技术核心是 CaFiLM, Context-aware Feature-wise Linear Modulation。它不是一个大的 MLP, 而是一个极小的按维度调制模块。

假设一张图片经过 \(f_i\) 后得到特征向量:

\[
z = [z_1, z_2, \ldots, z_C]
\]

一个 batch 的平均特征是:

\[
\mu = [\mu_1, \mu_2, \ldots, \mu_C]
\]

对每个特征维度 \(c\), CASA 用当前样本的 \(z_c\) 和当前 batch 的 \(\mu_c\) 生成两个调制参数:

\[
\begin{bmatrix}
\gamma_c \\
\beta_c
\end{bmatrix}
=
A
\begin{bmatrix}
z_c \\
\mu_c
\end{bmatrix}
+ b
\]

然后用它们调制原特征:

\[
z'_c = \gamma_c z_c + \beta_c
\]

最后所有 \(z'_c\) 拼回新的特征向量 \(z'\), 再送入分类器 \(h_i\)。

这里最重要的是: \(A\) 和 \(b\) 是全局共享的可学习参数, 所有特征维度共用同一套规则。论文中这个模块只有 6 个参数:

\[
A \in \mathbb{R}^{2 \times 2}, \quad b \in \mathbb{R}^{2}
\]

这个设计的直觉是: 不让 \(g\) 随意把特征映射到一个新空间, 而是在原特征空间里做保守的缩放和平移。这样既能利用 batch context, 又不至于破坏已经训练好的分类器。

## 实验和消融

论文在 DomainBed 上测试, 数据集包括 PACS, VLCS, OfficeHome, TerraIncognita 和 DomainNet。比较方法包括 ERM, I-Mixup, MLDG, SagNet, CORAL, mDSDI, ITL-Net, MIRO, SWAD, DNA 和 EoA。

ERM 是最朴素的 baseline: 把所有源域数据合在一起训练一个 ResNet-50 分类器, 然后直接在未见测试域上测试。

论文里的 Ensemble (Ours) 是作者自己的 ensemble baseline。它把多个 meta-source 模型的预测概率平均, 但不使用 CASA 的 context-aware modulation。结果显示, ensemble 本身已经能带来明显提升:

\[
\text{ERM average} = 63.3
\]

\[
\text{Ensemble (Ours) average} = 66.8
\]

完整 CASA 的平均结果是:

\[
\text{CASA average} = 68.8
\]

这说明提升来自两部分: 一部分是多个源模型 ensemble, 另一部分才是 batch-context feature modulation。

消融实验也支持这个判断。单纯把 adaptation module 加进模型结构里, 但不按 CASA 的两阶段 meta-target adaptation 方式训练, 提升很有限:

\[
\text{Ensemble}(h_i \circ f_i) = 66.8
\]

\[
\text{Ensemble}(h_i \circ g_i \circ f_i) = 67.0
\]

\[
\text{CASA} = 68.8
\]

另外, 去掉 batch context 后性能也会下降, 说明 mini-batch feature mean 确实提供了有用的 domain clue。

## 评价

这篇文章的方法很简单, 甚至有点太简单。它的主要贡献不是复杂结构, 而是一个清楚的组合:

1. 用 meta-source / meta-target 模拟 domain generalization 场景。
2. 用 mini-batch feature mean 表示当前 domain context。
3. 用极小的 CaFiLM 模块做特征调制。
4. 用多个 meta-source 模型 ensemble 提升稳定性。

我觉得这篇最值得吸收的是它对 test-time context 的使用方式。DG 通常强调测试域不可见, 但不可见不等于测试样本本身没有分布信息。CASA 把 batch 统计量当作弱 domain signal, 用非常保守的方式调制特征, 这是一个轻量但合理的设计。

不过它的 novelty 也确实主要在组合层面。所谓 context-aware self-adaptation 听起来比较大, 实际实现就是 batch mean conditioned feature modulation。真正有意思的问题是: 这种 6 参数的共享调制规则到底能学到多少 domain-specific adjustment, 以及它在更复杂、更细粒度的 domain shift 下是否仍然足够。
