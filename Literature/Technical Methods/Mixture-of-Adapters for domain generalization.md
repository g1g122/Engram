# Mixture-of-Adapters for Domain Generalization: Adaptive PEFT over Large Pretrained Models

[Paper Link](https://doi.org/10.1109/WACV61041.2025.00801)

## 目录

[1. 文章定位与核心问题](#1-文章定位与核心问题)

[2. 方法主线：从 Full Fine-tuning 到 MoA](#2-方法主线从-full-fine-tuning-到-moa)

[3. PEFT 作为正则化：为什么少改参数反而更稳](#3-peft-作为正则化为什么少改参数反而更稳)

[4. Adapter 机制：LoRA 与 KAdaptation](#4-adapter-机制lora-与-kadaptation)

[5. Mixture-of-Adapters：动态适配容量](#5-mixture-of-adapters动态适配容量)

[6. 实验结果与证据链](#6-实验结果与证据链)

[7. 我的理解与项目启发](#7-我的理解与项目启发)

[Appendix A. 数学推导：LoRA、KAdaptation 与 Kronecker Product](#appendix-a-数学推导lorakadaptation-与-kronecker-product)

[Appendix B. Loss Landscape 与 Hessian 解释](#appendix-b-loss-landscape-与-hessian-解释)

---

## 1. 文章定位与核心问题

这篇文章处理的是 **Domain Generalization, DG**。DG 的设定是：训练时有多个已知源域 source domains，测试时面对未见过的目标域 target domain。训练和测试共享类别空间，但图像风格、背景、采集方式、视觉分布可能不同。模型需要学到的不只是源域上的分类规则，而是能跨域保持稳定的表示。

用符号表示，训练域为：

\[
\mathcal{D}=\{\mathcal{D}^1,\mathcal{D}^2,\dots,\mathcal{D}^K\}
\]

目标域为：

\[
\mathcal{T}=\{\mathcal{T}^1,\mathcal{T}^2,\dots,\mathcal{T}^{K'}\}
\]

ERM 在训练域上最小化经验风险：

\[
\hat{\mathcal{E}}_{\mathcal{D}}(\theta)
=
\frac{1}{K}\sum_{i=1}^{K}
\frac{1}{n_{\mathcal{D}^i}}
\sum_{j=1}^{n_{\mathcal{D}^i}}
\ell(f_\theta(x_j^i),y_j^i)
\]

目标是希望得到的参数：

\[
\hat{\theta}
=
\arg\min_\theta \hat{\mathcal{E}}_{\mathcal{D}}(\theta)
\]

不仅在源域好，也能在未见目标域上好。

这篇文章的切入点不是再设计一个复杂的 DG loss，而是重新审视一个更基础的问题：**大预训练模型本身已经包含较强的跨域表示能力，那么 DG 的关键也许不是“重新学 invariant feature”，而是“不要破坏预训练模型已经学到的鲁棒表示”。**

因此，这篇文章可以用一句话概括：

> 这篇文章的核心不是“加了一个 adapter”，而是把 DG 问题重新理解为：在大预训练模型上，如何控制参数更新的自由度，使模型既能适配任务，又不破坏原本的 OOD 鲁棒性。

这也是它和许多传统 DG 方法的区别。传统 DG 方法常常围绕 domain-invariant objective、feature alignment、adversarial training、data augmentation、meta-learning 等机制展开；这篇文章则把重点放到 **fine-tuning protocol** 上：当 backbone 已经是 CLIP / OpenCLIP 这类大预训练视觉模型时，最重要的问题不是从零构造不变特征，而是如何调节模型被允许改变的程度。

---

## 2. 方法主线：从 Full Fine-tuning 到 MoA

整篇文章的逻辑链可以整理成六步。

第一步，指出 DG 中很多复杂算法相比 ERM 的提升并不总是显著，而大预训练模型本身已经包含较强的跨域表示能力。也就是说，如果 backbone 足够强，问题的一部分已经被 pretraining 解决了。

第二步，观察 full fine-tuning 会破坏大模型的 OOD robustness。大模型不是不能用，而是不能无约束地改。full fine-tuning 的自由度太大，容易让模型过度适应 source domains，从而损失预训练阶段学到的通用表示。

第三步，引入 PEFT，把更新限制在低维或结构化空间中。LoRA 用整体低秩更新，KAdaptation 用 Kronecker block 结构和 low-rank fast weights。它们的作用不只是省参数，更是在 DG 场景下给 fine-tuning 加正则。

第四步，发现不同数据集和不同 domain shift 需要不同适配容量。固定容量 adapter 不够灵活。Figure 1 的重要观察就是：参数越少不一定越好，参数越多也不一定越好；关键是可训练容量要和 distribution shift 的强度匹配。

第五步，提出 **Mixture-of-Adapters, MoA**：多个不同 rank 的 adapters 加 router，让 token 动态选择不同容量的适配路径。

第六步，用 benchmark、loss landscape、Hessian eigenvalues 和 routed patch visualization 共同构成证据链：受限更新和动态 adapter 容量可能使模型处在更平坦、更稳定的经验 loss 区域，从而更适合 DG。

把这条线压缩成一个概念图，就是：

\[
\text{Pretrained Robustness}
\rightarrow
\text{Avoid Destructive Fine-tuning}
\rightarrow
\text{Structured PEFT}
\rightarrow
\text{Dynamic Adapter Capacity}
\rightarrow
\text{Flatter Empirical Loss Landscape}
\]

这里最重要的思想不是某个公式本身，而是 **adaptive capacity**：模型不应该只有一个固定的微调强度，而应该能够根据数据、token 和 domain shift 的情况选择不同的适配容量。

---

## 3. PEFT 作为正则化：为什么少改参数反而更稳

PEFT, Parameter-Efficient Fine-Tuning，通常被理解为“省参数的微调”。但在这篇论文里，更重要的是它的另一个作用：**正则化**。

设某一层原始预训练权重为：

\[
W_0 \in \mathbb{R}^{k\times d}
\]

普通线性层为：

\[
h = W_0x
\]

full fine-tuning 等价于学习一个完整的新权重：

\[
W_{\text{new}} = W_0 + \Delta W
\]

于是：

\[
h = W_{\text{new}}x
=
(W_0+\Delta W)x
=
W_0x+\Delta Wx
\]

full fine-tuning 的问题在于，\(\Delta W\) 可以是任意 \(k\times d\) 矩阵。自由度很大，适应训练域很强，但也更容易破坏大预训练模型已有的通用表示。

PEFT 的思想是：冻结 \(W_0\)，只学习一个受限制的 \(\Delta W\)。限制越强，模型越不容易乱改原模型；但如果限制过强，又可能欠拟合。这就是这篇论文一直在讨论的“适配容量”和“正则强度”的平衡。

论文用 Figure 1 说明这一点：在 CLIP ViT-B/16 上，比较不同可训练参数量的方法：

- **Linear Probing**：冻结整个 backbone，只训练最后分类头。
- **Bias (MSA)**：只训练 Transformer self-attention 模块里的 bias。
- **Bias (MSA+MLP)**：训练 attention 和 MLP 里的 bias。
- **Full Fine-tuning**：所有参数都训练。

这里的核心不是“参数越少越好”，而是：**可训练容量需要和 domain shift 的程度匹配**。例如，PACS、OfficeHome、DomainNet 等数据集上，少量参数微调可能已经够用，甚至比 full fine-tuning 更稳；但 TerraIncognita 这种相机陷阱野外动物数据与常规图像差异更大，过度冻结可能欠拟合，需要更多可训练参数。

所以作者后面提出 MoA 的动机就来自这里：不同数据、不同 token、不同 shift 强度，可能需要不同容量的 adapter。单一固定容量的 PEFT 方法不一定总是最优。

---

## 4. Adapter 机制：LoRA 与 KAdaptation

这篇文章比较了多种 partial fine-tuning 和 adapter-based PEFT 方法，其中最关键的是 LoRA 与 KAdaptation。它们都冻结预训练权重，只学习一个小的更新项，但限制方式不同。

LoRA 的更新形式是：

\[
\Delta W = BA
\]

它的核心假设是：

> 下游任务需要的权重变化落在低维子空间里。

LoRA 不是直接训练完整的 \(\Delta W\)，而是把它写成两个小矩阵的乘积。这样 \(\Delta W\) 的秩不超过中间维度 \(r\)，模型只能沿少数几个独立方向修补预训练权重。它既节省参数，也限制了模型“怎么改”。

KAdaptation 的更新形式是：

\[
\Delta W
=
\sum_i A_i\otimes (u_i v_i^\top)
\]

它的核心假设是：

> 权重更新不仅是低自由度的，而且具有可复用的 Kronecker block 结构；具体内容块也可以低秩表示。

两者都可以看成对 full fine-tuning 的正则化，但正则方式不同。LoRA 是整体低秩，KAdaptation 是结构化 block 更新 + fast weight 低秩 + slow weight 共享。

从自由度角度看，可以粗略理解为：

\[
\text{Full Fine-tuning}
>
\text{LoRA}
>
\text{KAdaptation}
\]

这里的“大于”表示自由度更大、限制更弱。KAdaptation 限制更多，因此通常正则更强，更不容易破坏预训练知识；但限制太强也可能造成欠拟合，所以需要调节容量。

这也是 MoA 的入口：如果单一 adapter 的限制强度固定，那么它不可能同时适合所有 domain shift。某些 token 可能只需要很小的更新，某些 token 可能需要更强的更新。固定容量 adapter 把这些情况都压进一个通道里，灵活性不够。

---

## 5. Mixture-of-Adapters：动态适配容量

MoA, Mixture-of-Adapters，就是把一个 adapter 换成多个不同容量的 adapters，再用 router 决定每个 token 应该走哪个 adapter。

其基本形式是：

\[
f_{\mathrm{MoE}}(x)
=
\sum_{i=1}^{N}
G(x)_i A_{r_i}(x)
\]

其中：

- \(A_{r_i}\) 是第 \(i\) 个 adapter，具有不同 inner rank \(r_i\)；
- \(G(x)_i\) 是 router 给第 \(i\) 个 adapter 的权重；
- \(N\) 是 adapter / expert 数量。

论文中使用 cosine router：

\[
G(x)
=
\operatorname{TOP}_k
\left(
\operatorname{Softmax}
\left(
\frac{E^\top W x}
{\tau\|Wx\|\|E\|}
\right)
\right)
\]

其中 \(Wx\) 是 token feature 投影，\(E\) 是 learnable expert embedding，\(\tau\) 是温度参数。直观上，router 计算 token 和各个 expert embedding 的相似度，然后选择最合适的 adapter。

因此，MoA 的关键不只是“多几个 adapter”，而是：

> 对不同 token 动态选择不同容量的参数更新路径。

论文把 adapter 接到 ViT 的 attention submodules 上。ViT 处理图像时会把图片切成 patch tokens，每个 token 经过多层 self-attention 和 MLP。MoA 的 router 可以针对 token 分配 adapter，这比对整张图选择一个 expert 更细。

这意味着模型可以做到：

- 背景 token 可能走一个 adapter；
- 物体边缘 token 可能走另一个 adapter；
- 语义区域 token 可能走更高容量 adapter；
- 不同 domain 中视觉风格不同的 token 可能被路由到不同 expert。

论文在 TerraIncognita 上展示 routed patches，是为了说明 router 不只是随机分配，而是可能把语义区域或物体相关区域聚集到某些 expert 上。比如同一地点不同时间拍摄的图像共享类似背景，但动物位置发生变化时，和物体相关的 routed patches 也会随之移动。这说明 router 可能在捕捉 token-level semantic information，而不只是按全图背景做粗糙分配。

---

## 6. 实验结果与证据链

这篇文章的实验不是只靠一个 benchmark 表格，而是构成了一条比较完整的证据链。

第一类证据是 **DomainBed benchmark**。论文评估 PACS、VLCS、OfficeHome、TerraIncognita 和 DomainNet，并使用 training-domain-validation model selection。主要实验使用 OpenCLIP ViT-B/16 with LAION-2B pretraining，表格中也引用或比较了 OpenAI CLIP、SWAG、ModelPool 等结果。

需要注意的是，论文确实报告了很强的 SOTA 或接近 SOTA 结果，但在平均指标上，某些大型 ensemble / ModelPool 方法仍然有可比甚至略高的结果。因此更稳妥的理解不是“MoA 全面压倒所有 DG 方法”，而是：

> 在不依赖大规模 ModelPool 或额外文本监督的条件下，MoA 用较少可训练参数达到很有竞争力、部分数据集 SOTA 的 DG 表现。

第二类证据是 **fine-tuning capacity 的比较**。Figure 1 显示 full fine-tuning 在很多 DG 数据集上并不是最优，甚至会显著退化；linear probing 和 bias tuning 在一些数据集上表现更稳，但在 TerraIncognita 这种视觉分布差异更大的数据集上，过强正则又可能不够。这支持了“适配容量需要调节”的动机。

第三类证据是 **loss landscape**。Figure 2 比较 full fine-tuning、LoRA、KAdaptation 和 KMoA 在 PACS 上的 loss surface。视觉上，full fine-tuning 更 sharp，LoRA 与 KAdaptation 更 flat，KMoA 看起来最 flat。这说明受限更新可能让模型找到更稳定的局部解。

第四类证据是 **Hessian eigenvalue spectra**。Figure 3 计算 top Hessian eigenvalues，并显示 KMoA 的最大特征值分布更集中在 0 附近。由于 Hessian 特征值刻画 loss surface 的局部曲率，这进一步支持了“PEFT / MoA 使模型处在更平坦区域”的解释。

第五类证据是 **router 和 expert 分析**。论文可视化 routed patches，说明 router 倾向于把物体 foreground、object outline 或语义区域分配到特定 experts。这为 MoA 的 token-level routing 提供了直观解释。

这些证据合在一起，不是严格证明“flatness 必然导致 DG 泛化”，但构成了一个合理的经验解释：

\[
\text{Restricted Update}
\rightarrow
\text{Less Destructive Fine-tuning}
\rightarrow
\text{Flatter Local Loss Geometry}
\rightarrow
\text{Better OOD / DG Behavior}
\]

---

## 7. 我的理解与项目启发

这篇文章对我最重要的启发是：**Domain Generalization 不一定要从头设计 domain-invariant objective。对于 CLIP / OpenCLIP 这类大预训练模型，更关键的问题是如何保留其预训练阶段获得的通用表示，同时给下游任务足够但不过度的适配能力。**

LoRA 和 KAdaptation 通过低秩或结构化更新限制模型修改；MoA 进一步引入多个不同容量的 adapter 和 token-level routing，使模型可以动态选择更新强度。Hessian 和 loss landscape 分析则提供了一个解释：这些受限更新可能让模型处在更平坦的经验 loss 区域，从而更有利于 OOD / DG 泛化。

如果把它放到项目脉络里，这篇文章的价值不是“复现一个 adapter 方法”，而是提供了一种理解框架：

- 大预训练模型本身已经是重要的 robustness prior；
- fine-tuning 的主要风险是破坏这个 prior；
- PEFT 的作用不仅是省参数，更是限制可学习更新的自由度；
- 固定容量 PEFT 仍然不够灵活，因为不同 token / domain shift 需要不同适配强度；
- MoA 把“适配容量”变成可路由、可选择、可组合的对象。

所以这篇文章真正可以带走的思想是：

> DG 中的 adaptation 不是越强越好，也不是越弱越好，而是要让模型在保留预训练鲁棒性的前提下，对不同输入区域和不同 shift 强度使用合适的更新容量。

---

## Appendix A. 数学推导：LoRA、KAdaptation 与 Kronecker Product

### A.1 LoRA：低秩更新到底是怎么来的

LoRA 的核心是把权重更新写成：

\[
\Delta W = BA
\]

其中：

\[
A\in\mathbb{R}^{r\times d},\qquad
B\in\mathbb{R}^{k\times r}
\]

所以：

\[
BA\in\mathbb{R}^{k\times d}
\]

这和原始权重 \(W_0\) 的形状相同，因此可以作为权重更新加到 \(W_0\) 上：

\[
W_{\text{new}} = W_0 + BA
\]

于是前向传播变成：

\[
h = W_0x + BAx
\]

注意，\(BA\) 不是“变成了 \(W\)”，而是被设计成和 \(W_0\) 同形状的更新矩阵。LoRA 只训练 \(A\) 和 \(B\)，不直接训练 \(W_0\)。

如果输入为：

\[
x\in\mathbb{R}^{d}
\]

那么 LoRA 的更新路径可以理解为：

\[
x \xrightarrow{A} Ax\in\mathbb{R}^{r}
\xrightarrow{B} BAx\in\mathbb{R}^{k}
\]

也就是：

\[
d \rightarrow r \rightarrow k
\]

中间维度 \(r\) 很小，是一个信息瓶颈。

### A.2 为什么 \(BA\) 是低秩的

矩阵乘积的秩满足：

\[
\operatorname{rank}(BA)
\leq
\min(\operatorname{rank}(B),\operatorname{rank}(A))
\]

因为 \(A\) 先把输入压到一个最多 \(r\) 维的子空间中，后面的 \(B\) 再怎么变，也不可能凭空制造超过 \(r\) 个独立方向。因此：

\[
\operatorname{rank}(BA)\leq r
\]

这就是 LoRA 的本质：

> 用一个 rank 不超过 \(r\) 的低秩矩阵来近似完整的权重更新。

如果 \(k=d=768\)，完整更新 \(\Delta W\) 需要：

\[
768\times 768 = 589824
\]

个参数。而 LoRA 只需要：

\[
768r + r768 = r(768+768)
\]

如果 \(r=2\)，只有：

\[
3072
\]

个参数。

秩不是一个单纯的线性代数公式。它可以理解为：

> 一个矩阵真正能够表达的独立变化方向数量。

一个 \(768\times768\) 的矩阵看起来很大，但如果 rank 只有 2，它真正能表达的独立模式只有 2 个。LoRA 的低秩更新意味着模型不能任意修改预训练权重，而只能沿少数几个独立方向修补它。

因此，低秩更新既节省参数，也是一种正则：它限制模型“怎么改”，从而减少过拟合源域和破坏预训练表示的风险。

### A.3 KAdaptation：从整体低秩到 Kronecker 结构化更新

LoRA 的约束是整体低秩：

\[
\Delta W = BA
\]

KAdaptation 的约束不同。它把更新矩阵写成多个 Kronecker product 的和：

\[
\Delta W
=
\sum_{i=1}^{t} A_i\otimes B_i
\]

这里 \(\otimes\) 是 Kronecker product，不是普通矩阵乘法。

如果：

\[
A=
\begin{bmatrix}
a_{11} & a_{12}\\
a_{21} & a_{22}
\end{bmatrix}
\]

那么：

\[
A\otimes B
=
\begin{bmatrix}
a_{11}B & a_{12}B\\
a_{21}B & a_{22}B
\end{bmatrix}
\]

所以 Kronecker product 的作用是：用一个小矩阵 \(A\) 决定 block 的排列和缩放，用另一个矩阵 \(B\) 作为被复制、被缩放的内容块，拼出一个更大的结构化矩阵。

因此，KAdaptation 不只是“少参数”，它还强制 \(\Delta W\) 具有 block-wise structure。

### A.4 fast weight 为什么也是低秩的

KAdaptation 进一步把 \(B_i\) 写成：

\[
B_i = u_i v_i^\top
\]

于是：

\[
\Delta W
=
\sum_{i=1}^{t}
A_i\otimes (u_i v_i^\top)
\]

这一步很容易误解。**\(B_i\) 低秩不是 Kronecker product 自动导致的，而是作者额外规定的参数化方式。**

如果直接学习完整的 \(B_i\)，那么 \(B_i\) 可以是满秩矩阵。只有当我们不直接训练 \(B_i\)，而是训练两个瘦矩阵 \(u_i\) 和 \(v_i\)，并令：

\[
B_i = u_i v_i^\top
\]

时，才有：

\[
\operatorname{rank}(B_i)
=
\operatorname{rank}(u_i v_i^\top)
\leq r
\]

其中 \(r\) 是中间维度。

因此，KAdaptation 的限制有两层。

第一层是 Kronecker 结构：

\[
\Delta W = \sum_i A_i\otimes B_i
\]

第二层是 fast weight 低秩化：

\[
B_i = u_i v_i^\top
\]

这比 LoRA 的“整体低秩”更结构化。LoRA 说整个 \(\Delta W\) 只能是低秩更新；KAdaptation 说 \(\Delta W\) 必须由若干 Kronecker block 结构组成，而且其中的 fast component 也要低秩。

### A.5 slow weights / fast weights 是什么意思

KAdaptation 里，\(A_i\) 和 \(B_i\) 的角色不同。\(A_i\) 被称为 **slow weights**，\(B_i\) 被称为 **fast weights**。

\(A_i\) 更像一个共享的结构模板。它决定 Kronecker block 的布局和缩放方式，也就是：

\[
A_i\otimes B_i
=
\begin{bmatrix}
a_{11}B_i & a_{12}B_i & \cdots\\
a_{21}B_i & a_{22}B_i & \cdots\\
\vdots & \vdots & \ddots
\end{bmatrix}
\]

这里 \(A_i\) 决定哪些 block 强、哪些 block 弱、block 之间如何组合。论文中说 \(A_i\) 是跨层共享的，这意味着不同 Transformer 层不各自学习完全独立的 \(A_i\)，而是共用某些结构模板。

\(B_i\) 则更像每层具体填进去的内容。它负责具体的适配更新，因此被称为 fast weights。为了减少自由度，KAdaptation 不直接学习完整 \(B_i\)，而是让它等于 \(u_i v_i^\top\)。

可以用一个类比理解：

> \(A_i\) 像 PPT 模板，决定版式；\(B_i\) 像每页具体内容。跨层共享 \(A_i\)，意味着很多层共用同一套结构版式；每层通过自己的 fast component 填入具体适配内容。

为什么不把 \(A_i\) 也低秩分解？数学上可以，但设计上未必合适。\(A_i\) 负责共享结构模板，如果也强行低秩化，可能让结构表达能力太弱。KAdaptation 的折中是：\(A_i\) 保留模板表达力并跨层共享，\(B_i\) 负责具体更新并低秩化。

---

## Appendix B. Loss Landscape 与 Hessian 解释

### B.1 Loss landscape：Figure 2 到底在看什么

Figure 2 画的是训练后模型参数附近的 loss surface。做法是：从训练好的参数 \(\theta^\*\) 出发，选两个扰动方向 \(d_1,d_2\)，然后观察：

\[
L(\theta^\*+\alpha d_1+\beta d_2)
\]

随 \(\alpha,\beta\) 变化的曲面。

如果曲面像尖山，说明参数附近 sharp：轻微扰动参数，loss 就急剧上升。如果曲面像平地或浅碗，说明参数附近 flat：轻微扰动参数，loss 变化不大。

作者用这个图说明：

- full fine-tuning 的 loss surface 更尖；
- LoRA 更平；
- KAdaptation 更平；
- KMoA 看起来最平。

这里真正的含义是：PEFT 通过限制参数更新，可能让模型找到更稳定的局部解。这个稳定性和 OOD 泛化有关，因为如果模型只是记住 source domains，参数解可能更 sharp；如果模型保留预训练的通用表示，loss landscape 可能更 flat。

但要注意，loss landscape visualization 是高维参数空间中的二维切片。它能提供直观解释，但不是严格证明。

### B.2 Hessian：为什么二阶导会变成矩阵

Hessian 是 loss 对参数的二阶导数矩阵。这里有几个容易误解的点。

首先，不是“参数自己对自己求导”，而是 **loss 对参数求导**。模型参数虽然训练完以后固定了，但我们仍然可以问：如果从当前参数点 \(\theta^\*\) 轻微扰动，loss 会怎么变。

假设模型有 \(N\) 个标量参数，摊平成：

\[
\theta=[\theta_1,\theta_2,\dots,\theta_N]
\]

loss 是：

\[
L(\theta)
\]

一阶导数是 gradient：

\[
\nabla L
=
\begin{bmatrix}
\frac{\partial L}{\partial \theta_1}\\
\frac{\partial L}{\partial \theta_2}\\
\vdots\\
\frac{\partial L}{\partial \theta_N}
\end{bmatrix}
\]

二阶导数要描述任意两个参数之间的关系，因此形成矩阵：

\[
H_{ij}
=
\frac{\partial^2 L}
{\partial \theta_i \partial \theta_j}
\]

完整 Hessian 为：

\[
H
=
\nabla_\theta^2 L(\theta)
\in\mathbb{R}^{N\times N}
\]

如果模型有 100M 个标量参数，完整 Hessian 理论上就是：

\[
100M\times100M
\]

的矩阵。但实际不可能显式构建它，只会通过 Hessian-vector product 近似计算 top eigenvalues。

### B.3 参数之间的“相互影响”是什么意思

Hessian 的非对角项：

\[
\frac{\partial^2 L}
{\partial \theta_i\partial \theta_j}
=
\frac{\partial}{\partial \theta_j}
\left(
\frac{\partial L}{\partial \theta_i}
\right)
\]

表示：

> 当 \(\theta_j\) 变化时，\(\theta_i\) 的梯度会怎么变化。

如果这个值大，说明两个参数在 loss 中耦合强。换句话说，更新 \(\theta_i\) 的方向会受到 \(\theta_j\) 的影响。

举一个简单函数：

\[
L(\theta_1,\theta_2)
=
\theta_1^2+3\theta_1\theta_2+\theta_2^2
\]

则：

\[
\frac{\partial L}{\partial \theta_1}
=
2\theta_1+3\theta_2
\]

再对 \(\theta_2\) 求导：

\[
\frac{\partial}{\partial \theta_2}
\left(
\frac{\partial L}{\partial \theta_1}
\right)
=
3
\]

这个 3 就说明 \(\theta_2\) 会影响 \(\theta_1\) 的梯度。神经网络中，前层参数和后层参数通过计算图层层相连，因此这种耦合非常普遍。

### B.4 Hessian 特征值表示什么

Hessian 是矩阵，不能直接用一个数描述。它的特征值和特征向量可以理解为：

- 特征向量：参数空间里的某个组合扰动方向；
- 特征值：沿这个方向，loss 的曲率大小。

如果：

\[
Hv = \lambda v
\]

那么 \(v\) 是一个参数空间中的方向，\(\lambda\) 表示沿这个方向的二阶曲率。

最大 Hessian 特征值：

\[
\lambda_{\max}(H)
\]

表示所有参数扰动方向中最尖的方向有多尖。

在局部二阶近似下：

\[
L(\theta+\delta)
\approx
L(\theta)
+
\nabla L(\theta)^\top\delta
+
\frac{1}{2}\delta^\top H\delta
\]

如果训练点附近：

\[
\nabla L(\theta)\approx0
\]

那么：

\[
L(\theta+\delta)
\approx
L(\theta)
+
\frac{1}{2}\delta^\top H\delta
\]

所以 Hessian 决定了参数扰动后 loss 增长的速度。\(\lambda_{\max}\) 越大，说明存在某个方向很尖；\(\lambda_{\max}\) 越接近 0，说明连最尖方向也相对平缓。

### B.5 Hessian-vector product：为什么不用显式构建 Hessian

完整 Hessian 太大，实际不显式存储。计算 top eigenvalues 通常用 Hessian-vector product, HVP。

这里的 \(v\) 不是参数，也不是特征值，而是一个和参数向量 \(\theta\) 同维度的方向向量：

\[
v\in\mathbb{R}^N
\]

它表示：在参数空间里沿某个方向轻微移动。

HVP 是：

\[
Hv
\]

它表示：

> 如果沿方向 \(v\) 移动参数，gradient 会怎样变化。

因为：

\[
H = \nabla_\theta^2 L
\]

所以 \(Hv\) 是 gradient 在方向 \(v\) 上的方向导数。

找最大特征值可以用 power iteration。随机初始化一个方向：

\[
v_0
\]

反复做：

\[
v_{t+1}
=
\frac{Hv_t}{\|Hv_t\|}
\]

如果最大特征值方向占主导，\(v_t\) 会逐渐对齐到最大特征向量。最后用：

\[
\lambda_{\max}
\approx
\frac{v^\top Hv}{v^\top v}
\]

估计最大特征值。

所以，Hessian eigenvalue 分析不是在找“参数的特征值”，而是在找 **loss 二阶曲率矩阵的特征值**。

### B.6 Hessian 到底对哪个数据算

Hessian 不只由模型参数决定，还由数据和 loss 决定。更准确地写：

\[
H(\theta;\mathcal{D})
=
\nabla_\theta^2 L(\theta;\mathcal{D})
\]

同一个模型参数 \(\theta\)，换不同输入数据 \(x\)、不同 batch、不同 domain，loss 变了，Hessian 也会变。

对于一个 batch：

\[
B=\{(x_1,y_1),\dots,(x_m,y_m)\}
\]

batch loss 是：

\[
L_B(\theta)
=
\frac{1}{m}
\sum_{s=1}^{m}
\ell(f_\theta(x_s),y_s)
\]

其中 \(x_s,y_s\) 是固定数据，求导变量是同一套模型参数 \(\theta\)。

batch Hessian 是：

\[
H_B(\theta)
=
\nabla_\theta^2 L_B(\theta)
=
\frac{1}{m}
\sum_{s=1}^{m}
\nabla_\theta^2
\ell(f_\theta(x_s),y_s)
\]

记单样本 Hessian 为：

\[
H_s
=
\nabla_\theta^2
\ell(f_\theta(x_s),y_s)
\]

则：

\[
H_B
=
\frac{1}{m}
\sum_{s=1}^{m}H_s
\]

所以概念上，batch Hessian 等于 batch 中每个样本 Hessian 的平均。

但是 batch 的最大特征值是：

\[
\lambda_{\max}
\left(
\frac{1}{m}\sum_{s=1}^{m}H_s
\right)
\]

而不是：

\[
\frac{1}{m}
\sum_{s=1}^{m}
\lambda_{\max}(H_s)
\]

两者一般不相等，因为不同样本最尖的方向可能不同。先平均 Hessian 再取最大特征值，衡量的是这个 batch 的平均 loss 是否存在一个共同尖锐方向。

### B.7 为什么小 batch 的 Hessian 会 noisy

因为 batch Hessian 是总体 Hessian 的抽样估计。理想情况下，我们想看整个 domain 的 expected loss：

\[
L_{\mathcal{D}}(\theta)
=
\mathbb{E}_{(x,y)\sim\mathcal{D}}
[
\ell(f_\theta(x),y)
]
\]

对应 Hessian 是：

\[
H_{\mathcal{D}}(\theta)
=
\nabla_\theta^2L_{\mathcal{D}}(\theta)
\]

但实际通常只能用有限样本或 mini-batch 近似：

\[
H_B(\theta)
=
\frac{1}{m}\sum_{s=1}^{m}H_s
\]

如果 \(m\) 很小，抽到的样本类别、难度、背景、domain 风格、outlier 都可能影响 Hessian。因此不同 batch 的 top eigenvalues 可能波动很大，这种抽样波动就是 noisy。

这也说明 Hessian flatness 的含义必须说清楚：

> 它不是“模型对所有输入都平坦”，而是“在某个数据分布或经验样本集上，loss 对参数的局部曲率较小”。

所以 Figure 2 / Figure 3 的 flatness 分析应该理解为：作者在选定的 PACS test environment 上，用训练好的模型分析 empirical loss landscape 的局部曲率。
