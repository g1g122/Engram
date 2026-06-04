# A Computational Ontology of Cognitive Maps: From Space to Tasks

[Paper Link](https://doi.org/10.1038/s41593-022-01153-y)

## 目录

[1. 文章定位：认知地图模型的计算本体论](#1-文章定位认知地图模型的计算本体论)

[2. 认知地图的计算问题：为什么需要这些模型](#2-认知地图的计算问题为什么需要这些模型)

　　[2.1 空间作为状态空间](#21-空间作为状态空间)

　　[2.2 非空间状态空间与图结构](#22-非空间状态空间与图结构)

[3. 第一类模型：SR / DR —— 状态空间如何服务于强化学习和规划](#3-第一类模型sr--dr--状态空间如何服务于强化学习和规划)

　　[3.1 SR：预测性状态表征](#31-sr预测性状态表征)

　　[3.2 为什么 \(v=Sr\)](#32-为什么-vsr)

　　[3.3 SR 与 place cells / grid cells](#33-sr-与-place-cells--grid-cells)

　　[3.4 SR 的 policy dependence](#34-sr-的-policy-dependence)

　　[3.5 DR：用默认表征缓解 SR 的策略依赖](#35-dr用默认表征缓解-sr-的策略依赖)

[4. 第二类模型：CSCG —— 如何从序列中快速构建 latent-state graph](#4-第二类模型cscg--如何从序列中快速构建-latent-state-graph)

　　[4.1 aliasing 与 latent state](#41-aliasing-与-latent-state)

　　[4.2 CSCG 的核心机制：clone states](#42-cscg-的核心机制clone-states)

　　[4.3 Bayes 在 CSCG 中做什么](#43-bayes-在-cscg-中做什么)

　　[4.4 emission distribution、clone 集合与图示理解](#44-emission-distributionclone-集合与图示理解)

　　[4.5 marginalization、EM 与 planning](#45-marginalizationem-与-planning)

　　[4.6 CSCG 能解释哪些细胞](#46-cscg-能解释哪些细胞)

[5. 第三类模型：路径积分模型 CANNs / VCOs —— 如何压缩结构并实现抽象移动](#5-第三类模型路径积分模型-canns--vcos--如何压缩结构并实现抽象移动)

[6. 第四类模型：TEM / SMP —— 路径积分、记忆绑定与结构泛化的整合](#6-第四类模型tem--smp--路径积分记忆绑定与结构泛化的整合)

　　[6.1 为什么需要“抽象结构 + 记忆绑定”](#61-为什么需要抽象结构--记忆绑定)

　　[6.2 \(\mathbf M\) 到底是什么：不是身份标签，而是当前环境的变量绑定机制](#62-m-到底是什么不是身份标签而是当前环境的变量绑定机制)

　　[6.3 “抽象位置”到底是什么](#63-抽象位置到底是什么)

　　[6.4 如何用 \(\mathbf M\) 预测未来感觉观察](#64-如何用-m-预测未来感觉观察)

　　[6.5 为什么加了 \(\mathbf M\) 后模型更强大](#65-为什么加了-m-后模型更强大)

　　[6.6 MEC、LEC、HPC 的分工与图 a 的理解](#66-meclechpc-的分工与图-a-的理解)

　　[6.7 TEM 与 SMP 的实现差异](#67-tem-与-smp-的实现差异)

　　[6.8 TEM/SMP 能重现哪些神经表征](#68-temsmp-能重现哪些神经表征)

[7. 四类模型之间的真正区别](#7-四类模型之间的真正区别)

[8. 文章的整合观点：海马既能快速建图，也能帮助皮层学习抽象结构](#8-文章的整合观点海马既能快速建图也能帮助皮层学习抽象结构)

[9. Replay：离线训练、建图与 credit assignment](#9-replay离线训练建图与-credit-assignment)

[10. 行为与神经表征：为什么不同细胞类型可以被统一解释](#10-行为与神经表征为什么不同细胞类型可以被统一解释)

[11. Factorized vs entangled representations：什么时候需要可组合表征](#11-factorized-vs-entangled-representations什么时候需要可组合表征)

[12. 时间状态、time cells 与 representational drift](#12-时间状态time-cells-与-representational-drift)

[13. 海马与高阶皮层：从空间地图到任务、语言、逻辑和数学](#13-海马与高阶皮层从空间地图到任务语言逻辑和数学)

[14. 文章结论：走向整合的 computational ontology](#14-文章结论走向整合的-computational-ontology)

---

## 1. 文章定位：认知地图模型的计算本体论

这篇文章试图建立一个关于认知地图模型的**计算本体论**。文章真正关心的是：不同模型分别解决认知地图构建中的哪一类计算问题，它们如何互补，以及未来能否被整合成一个统一的海马—皮层结构学习框架。

全文的核心线索可以概括为：认知地图要解决的是如何从经验序列中学习状态空间，如何消除感觉混叠，如何压缩和泛化结构，如何将抽象结构绑定到具体经验，并最终服务于规划、强化学习、记忆和高阶认知。

文章开头提到的 **Tolman's cognitive map** 指的是一种心理学层面的“脑内地图”概念：动物不是只学习固定的刺激—反应链，而是在内部形成环境关系结构，从而支持绕路、走捷径和灵活规划。位置细胞和网格细胞的意义在于，它们为这种抽象概念提供了可能的神经实现：位置细胞更像具体地点标签，网格细胞更像可用于向量计算的空间坐标系统。

这里要注意术语范围：典型 **place cells** 更常说在 hippocampus 本体中，典型 **grid cells** 主要在 medial entorhinal cortex（MEC）中。文章很多地方说的是 **hippocampal formation** 或 hippocampal–entorhinal system，它不是狭义海马体，而是把海马、下托、内嗅皮层等作为一个功能系统讨论。因此，网格细胞不应理解成“在海马体内部的细胞”，而应理解成海马结构/海马—内嗅系统中的全局结构表征。

---

## 2. 认知地图的计算问题：为什么需要这些模型

认知地图的目标不是储存全部感觉经验，而是为行为建立一个可用的**状态空间**。所谓 state，可以理解为某一时刻与行为决策相关的“世界配置”。完整世界状态包含无数维度，其中大量维度与当前任务无关，因此系统必须抽象出合适的状态表征。状态表示得好，价值学习、路径规划和目标迁移都会更容易；状态表示不好，系统就会陷入维度灾难和低效搜索。

文章把认知地图的核心作用概括为 **organizing knowledge for generalization**：把知识组织成可复用的结构，使个体能从少量观察中快速推断更大的规律。这与心理学中的 schema 和 learning set 有关：前者强调理解新信息的心理框架，后者强调从一类任务中学会共同规则，从而更快适应新任务。

空间只是状态空间的一个特例。物理位置可以作为 state；非空间任务中的任务阶段、社会关系、证据积累、时间进程也可以作为 state。因此，真正的问题不是“空间如何表示”，而是“大脑如何学习任意结构化领域中的状态关系”。

### 2.1 空间作为状态空间

在物理空间中，状态空间很容易直观化：一个位置就是一个状态，所有位置构成状态空间。但即使状态空间已经确定，仍然存在一个关键问题：每个状态到底如何表示？

一种方式是用离散标签表示位置，例如 A、B、C、D。这类似于位置细胞：每个位置或局部区域需要有相应的局部响应模式。新地点出现时，系统需要新的标签或新的细胞组合，并重新学习它与其他状态的关系。

另一种方式是用坐标或向量结构表示位置，例如 \((x,y)\)。新地点不需要创建一个全新的离散身份，而只是同一坐标规则下的新数值实例。这类似于网格细胞：网格细胞不是给每个地点单独命名，而是提供一套可延展的周期性坐标结构。多个尺度和相位的网格细胞组合起来，可以区分大量位置，并支持“目标位置 − 当前位置”这样的向量计算。

因此，文章说 grid cells “abolish the need for computation” 并不是说大脑完全不计算，而是说：如果表征方式足够合适，原本需要逐步搜索邻接关系的问题，可以转化为较简单的向量运算。

### 2.2 非空间状态空间与图结构

非空间领域的问题更困难，因为没有显然的 \((x,y)\) 坐标。文章提出可以把空间学习理解为 graph 上的关系学习：节点表示状态，边表示状态之间可转移或相关联。这样，物理空间、家族关系、社会网络、棋子关系、植物分类等都可以被看作关系结构。

例如在家族图中，每个人可以是节点，parent、child、sibling 等关系可以是边。理解图上的结构后，系统就能进行规划：从 Bob 节点出发，沿着关系边传播，直到 Alice 节点出现，就能得到 Bob 到 Alice 的关系路径。

---

## 3. 第一类模型：SR / DR —— 状态空间如何服务于强化学习和规划

SR/DR 这一类模型主要回答：**一旦状态空间存在，它如何被用于价值计算、规划和行为控制。** 它们不重点解释状态空间最初如何从感觉序列中被发现，而是解释已有状态之间的关系如何服务强化学习。

### 3.1 SR：预测性状态表征

SR（successor representation）把认知地图理解为一种预测性状态表征。它不是只记录一步邻接关系，而是记录从当前状态出发，未来会访问其他状态的预期程度。

若 \(T\) 是一步转移矩阵，\(T^n\) 就表示 n 步后的转移结构。因为两步转移需要把所有可能中间状态加总起来：

\[
(T^2)_{ij}=\sum_k T_{ik}T_{kj}
\]

所以 \(T^n\) 可以理解为“经过 n 步之后到达各状态的概率结构”。SR 将所有步数的未来转移加权求和：

\[
S=I+\gamma T+\gamma^2T^2+\gamma^3T^3+\cdots
=\sum_{n=0}^{\infty}\gamma^nT^n
\]

其中 \(\gamma\) 是折扣因子，越远的未来权重越低。这个式子也可以写成矩阵等比级数形式：

\[
S=(I-\gamma T)^{-1}
\]

推导来自普通等比级数：

\[
1+x+x^2+\cdots=(1-x)^{-1}
\]

矩阵中令 \(A=\gamma T\)，就得到：

\[
I+A+A^2+\cdots=(I-A)^{-1}
\]

因此：

\[
S=(I-\gamma T)^{-1}
\]

SR 的元素 \(S_{ij}\) 表示从状态 \(i\) 出发，未来访问状态 \(j\) 的预期程度，也就是两个状态之间通过所有可能路径形成的 connectedness。它不是单纯的“有没有直接边”，而是包含多步可达性的强弱。

### 3.2 为什么 \(v=Sr\)

强化学习中的状态价值表示从当前状态出发，未来能获得的折扣奖励总和。如果奖励向量为 \(r\)，则：

\[
v=r+\gamma Tr+\gamma^2T^2r+\gamma^3T^3r+\cdots
\]

其中 \(Tr\) 是矩阵乘以奖励向量，表示从每个状态出发一步后预期获得的奖励；\(T^2r\) 表示两步后的预期奖励；依此类推。于是：

\[
v=(I+\gamma T+\gamma^2T^2+\cdots)r
\]

而括号中的部分就是 SR：

\[
v=Sr
\]

这里的 \(v\) 是所有状态的价值向量，不是单个状态的价值。SR 完成的是价值计算中的结构部分：它预先编码了“从每个状态未来会到哪里”；奖励向量 \(r\) 则告诉系统“哪些状态有奖励”。二者相乘就得到每个状态的长期预期价值。

需要注意，\(v=Sr\) 快速计算的是某个既定策略下的价值 \(v^\pi=S^\pi r\)，不自动等于最优价值 \(v^*\)。

### 3.3 SR 与 place cells / grid cells

Stachenfeld 等人的关键发现是，SR 矩阵的列看起来像 hippocampal place cells，而 SR 的某些特征向量看起来像 entorhinal grid cells。

原因是：固定某个状态 \(j\)，看 \(S\) 的第 \(j\) 列，就是在看“从所有位置出发，未来访问状态 \(j\) 的程度”。这个值通常在 \(j\) 附近最高，离 \(j\) 越远越低，画回空间后就像一个局部 place field。

而对 \(S\) 做特征分解，相当于提取整个状态空间连接结构中的基本空间模式。在二维环境中，这些模式中有些呈现周期性结构，经过零阈值处理后只保留正值区域，就会像网格细胞那样形成重复的 firing fields。所谓 thresholded at zero，就是把数学特征向量中的负值截为 0，因为真实神经元放电率不能为负。

文章还提到，位置细胞协方差矩阵的某些特征向量也类似网格细胞。这说明 grid-like patterns 可能不是孤立机制，而是可以从位置表征或状态连接结构的统计规律中自然产生。

### 3.4 SR 的 policy dependence

SR 的重要局限是 policy dependence。SR 学到的是某个策略 \(\pi\) 下的未来状态占用：

\[
S^\pi=I+\gamma T^\pi+\gamma^2(T^\pi)^2+\cdots
\]

其中：

\[
T^\pi(s'|s)=\sum_a \pi(a|s)P(s'|s,a)
\]

这说明 SR 把环境动力学 \(P(s'|s,a)\) 和行为策略 \(\pi(a|s)\) 合在了一起。因此它不是纯粹的环境地图，而是“在某个策略下的预测性地图”。

奖励移动时，环境结构可能没变，但最优策略会变。旧 SR 仍然描述旧策略下的未来访问结构，因此 \(S^\pi r_{new}\) 计算的是“如果还按旧策略行动，新奖励下的价值”，不一定是新最优策略下的价值。障碍物出现时更严重，因为不仅最优策略变了，状态转移结构本身也变了。

SR 不一定需要 reward 才能学出来。它可以通过状态序列学习未来占用关系；reward 主要在后续通过 \(v=Sr\) 转换成价值。reward 之所以仍会影响 SR，是因为在真实任务中 reward 会改变行为策略，而 SR 依赖产生序列的策略。

### 3.5 DR：用默认表征缓解 SR 的策略依赖

DR（default representation）试图缓解 SR 的 policy dependence。它的想法是：不要把表征绑定到某个具体目标策略上，而是先学习一个 default behavior 下的状态连接结构。default behavior 可以理解为无特定目标时的自然探索、随机移动或习惯性移动。

如果默认转移结构为 \(P_0\)，那么 DR 类似于：

\[
D\approx I+\gamma P_0+\gamma^2P_0^2+\cdots
\]

奖励出现或改变时，系统不是从零重新学习整个 SR，而是用奖励对这个默认表征进行重新加权，使默认行为偏向高价值状态。在线性强化学习中，这可以通过 desirability 一类变量实现。直观地说，下一状态越“值得去”，默认转移到它的概率就被放大：

\[
\pi^*(s'|s)\propto P_0(s'|s)z(s')
\]

所以 DR 快，不是因为学习困难消失了，而是因为困难的部分——世界的默认连接结构——已经提前学好。奖励变化时只需要更新状态的吸引力和策略偏置，而不需要重新学习整个环境结构。它适合奖励变化，但如果障碍改变了转移结构，旧 DR 仍然需要更新。

这一节最后应总结：SR/DR 的核心贡献不是解释状态空间如何从感觉中生成，而是解释一旦状态空间存在，它如何被用于价值计算、规划和行为控制。

---

## 4. 第二类模型：CSCG —— 如何从序列中快速构建 latent-state graph

CSCG 回答的是另一个问题：**状态空间本身如何从感觉序列中学出来。** SR/DR 更像是在问“已有状态空间怎样服务价值和规划”，而 CSCG 更像是在问“状态空间的节点到底从哪里来”。它的关键思想是：当前感觉输入本身不足以定义状态，模型必须从观察序列和动作序列中推断背后的 latent states。

### 4.1 aliasing 与 latent state

单个感觉观察不能直接定义状态，因为相同观察可能出现在不同位置或不同任务阶段，并导致完全不同的未来后果。这就是 aliasing：不同真实状态产生相同观察，导致系统如果只看当前感觉输入，就会把它们误认为同一个状态。

因此，状态必须是 latent state，而不只是 observation。latent state 是当前观察背后的真实状态，它需要通过上下文和序列结构来推断。当前观察可能一样，但它之前和之后连接到的状态不同，这些序列信息可以消除歧义。

例如两个白色走廊看起来一样，但一个通向奖励，另一个通向死路。如果只根据当前视觉观察，它们会被合并；如果根据前后序列，它们必须被分成两个不同状态。

更一般地说，observation 不一定只能是静态物体，也可以是声音、事件、短时间片段或已经被前级系统加工过的事件标签。关键不在于 observation 的内容是否简单，而在于：同一个 observation 仍然可能对应不同 latent states。比如同样是“门铃响”这个观察事件，在不同上下文中可能意味着朋友来了、实验信号响了，或者只是记忆片段被触发。

### 4.2 CSCG 的核心机制：clone states

CSCG 的核心做法是：为同一个 observation 建立多个 clone states。比如同样看到 frog，模型中可以有 frog clone 1、frog clone 2、frog clone 3。它们都对应同一个感觉观察，但在图中的连接关系不同。

这里要区分两个层级：

```text
frog = 表面感觉观察，也就是 observation
frog clone = 模型内部的潜在状态单元，也就是 latent state / clone state
```

所以，“青蛙”和“青蛙克隆”不是一回事。青蛙是系统当前看到的内容；青蛙克隆是模型内部用于区分“这是哪一个青蛙状态”的隐藏单元。多个 frog clones 可以生成同一个 frog observation：

```text
frog clone 1 ┐
frog clone 2 ├──→ frog
frog clone 3 ┘
```

这说明反向推断是不确定的：看到 frog，并不等于知道当前是哪一个 frog clone。模型必须结合前后序列来判断。例如：

```text
树 → frog clone 1 → 池塘
石头 → frog clone 2 → 洞口
```

两次看到的都是 frog，但因为前后连接不同，它们应被分配给不同 clone states。这解决了“同一个观察出现在不同位置或语境”的问题。模型不是根据“现在看起来像什么”来定义状态，而是根据“这个观察通常从哪里来、接下来通向哪里”来定义状态。

### 4.3 Bayes 在 CSCG 中做什么

CSCG 使用贝叶斯推断做两件事。

第一，推断当前观察对应哪个 clone。比如当前看到 frog，但有多个 frog clones 都能解释这个观察。模型会结合上一状态和已学到的转移结构，判断哪个 clone 最可能是当前真实 latent state。

第二，学习 clone 之间的转移权重。经过多次观察序列后，模型会学到哪些 clone 之间经常相继出现。这些 clone 之间的转移权重类似于图中的转移矩阵，但关键在于：普通图的节点常常是建模者预先给定的，而 CSCG 中的状态空间是模型从序列中学出来的。

CSCG 与 hidden Markov model 很接近：真正观察到的是 observation sequence 和 action sequence；真正要推断的是背后的 hidden state sequence。它可以写成：

\[
\mathbf X = \{\mathbf x_1, \mathbf x_2, \cdots, \mathbf x_T\},\quad
\mathbf A = \{\mathbf a_1, \mathbf a_2, \cdots, \mathbf a_T\},\quad
\mathbf Z = \{\mathbf z_1, \mathbf z_2, \cdots, \mathbf z_T\}
\]

其中 \(\mathbf X\) 是整条观察序列，\(\mathbf A\) 是整条动作序列，\(\mathbf Z\) 是整条 latent-state 序列；下标 \(t\) 表示第几个时间点。

论文里这些符号常写成加粗形式，是因为每个时间点的 observation、action 或 latent state 在实现中通常可表示为向量。例如 \(\mathbf x_t\) 概念上可以是 frog 这个观察类别，但实现上可以是 one-hot 向量；\(\mathbf z_t\) 概念上是当前 clone state，但实现上也可以是 one-hot 向量或概率分布向量。因此要区分：

```text
X：整条观察序列
x_t：第 t 时刻的观察表示
x：某一种 observation 类型，例如 frog 或 A

Z：整条 latent-state 序列
z_t：第 t 时刻的 latent-state 表示
z_i：latent-state 向量内部第 i 个 clone 单元
```

更严谨地说，可以把某一时刻的 latent-state 向量写成：

\[
\mathbf z_t = [z_{t,1}, z_{t,2}, \cdots, z_{t,N}]
\]

这里 \(t\) 管时间，\(i\) 管向量内部的 clone 编号。若 \(z_{t,2}=1\)，就表示第 \(t\) 个时间点第 2 个 clone 单元被激活；如果第 2 个 clone 单元是 frog clone 2，那么当前 latent state 就是 frog clone 2。

CSCG 的完整序列模型写作：

\[
p(\mathbf X,\mathbf Z,\mathbf A)
=
p(\mathbf z_0)
\prod_t
p(\mathbf x_t|\mathbf z_t)
p(\mathbf z_t,\mathbf a_t|\mathbf z_{t-1})
\]

左边 \(p(\mathbf X,\mathbf Z,\mathbf A)\) 是整条观察序列、整条 latent-state 序列和整条动作序列共同出现的联合概率。它不是单个时间点的概率，而是某一整段经验、某一条隐藏路径及其动作序列一起成立的概率。

右边可以分成三部分理解。第一项 \(p(\mathbf z_0)\) 是初始 latent state 的概率。第二项 \(p(\mathbf x_t|\mathbf z_t)\) 是 emission distribution，也就是“给定当前 latent state，会生成什么 observation”。第三项 \(p(\mathbf z_t,\mathbf a_t|\mathbf z_{t-1})\) 是转移结构，也就是“给定上一个 latent state，当前会到哪个 latent state，并伴随什么 action”。

这个连乘展开后就是：

\[
p(\mathbf z_0)
\cdot
p(\mathbf x_1|\mathbf z_1)p(\mathbf z_1,\mathbf a_1|\mathbf z_0)
\cdot
p(\mathbf x_2|\mathbf z_2)p(\mathbf z_2,\mathbf a_2|\mathbf z_1)
\cdot
p(\mathbf x_3|\mathbf z_3)p(\mathbf z_3,\mathbf a_3|\mathbf z_2)
\cdots
\]

它只依赖上一个 latent state，不是因为真实历史不重要，而是因为模型假设当前 latent state 已经压缩了与过去有关、对未来预测有用的信息。这就是马尔可夫假设：状态转移主要通过相邻 latent states 表达。

这个公式可以从一般概率链式法则理解。一般来说，联合概率可以写成：

\[
p(\mathbf X,\mathbf Z,\mathbf A)
=
p(\mathbf X|\mathbf Z,\mathbf A)p(\mathbf A|\mathbf Z)p(\mathbf Z)
\]

CSCG 进一步加入两个模型假设。第一，给定整条 latent-state 序列后，每个 observation 只由同一时刻的 latent state 生成：

\[
p(\mathbf X|\mathbf Z)
=
\prod_t p(\mathbf x_t|\mathbf z_t)
\]

这不是概率论的必然真理，而是 HMM/CSCG 的条件独立假设。它的意思是：\(\mathbf z_1\) 生成 \(\mathbf x_1\)，\(\mathbf z_2\) 生成 \(\mathbf x_2\)，依此类推；序列依赖主要放在 latent states 之间的转移里。第二，latent states 和 actions 按一阶马尔可夫方式转移：

\[
p(\mathbf Z,\mathbf A)
=
p(\mathbf z_0)\prod_t p(\mathbf z_t,\mathbf a_t|\mathbf z_{t-1})
\]

把这两部分合起来，就得到论文中的 CSCG 序列概率公式。

这里需要注意，概率链式法则本身允许按不同顺序拆联合概率，也允许把多个变量打包成一个块。例如 \(p(\mathbf z_1,\mathbf a_1|\mathbf z_0)\) 还可以继续拆成 \(p(\mathbf a_1|\mathbf z_0)p(\mathbf z_1|\mathbf z_0,\mathbf a_1)\)，也可以拆成 \(p(\mathbf z_1|\mathbf z_0)p(\mathbf a_1|\mathbf z_0,\mathbf z_1)\)。真正不能随意做的是“省略条件”。例如把 \(p(\mathbf z_t,\mathbf a_t|\mathbf z_{0:t-1},\mathbf a_{1:t-1})\) 简化成 \(p(\mathbf z_t,\mathbf a_t|\mathbf z_{t-1})\)，不是链式法则自动给出的，而是 CSCG 的一阶马尔可夫假设：过去历史中与未来预测相关的信息已经被当前 latent state 总结。

这个联合概率也可以理解为给某一条候选 hidden path 打分。假设观察到：

```text
X = frog → tree → frog
A = East → South → West
```

候选 latent-state 解释可以是：

```text
Z = frog clone 1 → tree clone 1 → frog clone 2
```

如果每个 clone 都能生成对应 observation，且这些 clone 之间的动作转移概率也高，那么连乘结果就大，说明这条 hidden path 能很好解释当前经验。反之，如果某个 clone 不应该生成当前 observation，或者某个动作转移几乎不会发生，那么连乘中会出现很小的项，整条路径概率就会很低。

### 4.4 emission distribution、clone 集合与图示理解

CSCG 中的 emission distribution \(p(x|z)\) 表示“给定 latent state \(z\)，观察到 observation \(x\) 的概率”。在这个模型中，emission 往往被设定得很硬：某个 clone 只会生成它所属的 observation。

如果 \(C(x)\) 表示 observation \(x\) 的所有 clones 的集合，那么：

\[
p(x|z_i\in C(x))=1
\]

\[
p(x|z_i\notin C(x))=0
\]

例如：

```text
C(frog) = {frog clone 1, frog clone 2, frog clone 3}
```

那么 frog clone 1、frog clone 2、frog clone 3 生成 frog 的概率都是 1；但 snail clone 生成 frog 的概率是 0。这里的重点是：一个 clone 只能生成自己的 observation，但同一个 observation 可以由多个 clones 生成。

图可以分成三层理解。左边是表面 observation 层，例如 A、B、C；中间是 clone latent-state graph，A、B、C 被复制成多个同色 clone 单元，clone 之间的连线表示学到的转移概率；右边是真实环境或真实状态图，其中同一个 observation 可以出现在多个位置。红线表示某些 clone 与右边具体状态位置的对应关系。

因此，图表达的是：表面上只有 A/B/C 三种观察，但真实环境中 A/B/C 会在多个位置重复出现。CSCG 通过把 A/B/C 各自复制成多个 clones，并学习这些 clones 之间的转移关系，恢复出一个去混叠的 hidden-state map。

### 4.5 marginalization、EM 与 planning

因为 \(\mathbf Z\) 是隐藏的，模型不知道真实 latent-state sequence 是哪一条，所以 CSCG 会对 \(\mathbf Z\) 做 marginalization。也就是说，它不会只押注某一条 clone 路径，而是把所有可能的 clone 路径都纳入计算。

例如观察序列是：

```text
A → B → C
```

可能的隐藏路径包括：

```text
A clone 1 → B clone 2 → C clone 1
A clone 2 → B clone 1 → C clone 3
A clone 3 → B clone 3 → C clone 2
```

模型训练时会综合这些可能解释，而不是预先知道哪条是真的。EM 算法正是用来处理这种有隐藏变量的训练问题。它可以粗略理解为两个步骤反复交替：先根据当前模型推断每个时间点可能是哪一个 clone，再根据这些推断更新 clone 之间的转移概率 \(p(\mathbf z_t,\mathbf a_t|\mathbf z_{t-1})\)。反复迭代后，模型就能同时得到较合理的 latent-state 推断和转移结构。

训练好以后，CSCG 可以用于 planning。给定起始 clone 和终点 clone，模型可以在已学到的 clone graph 上推断一条可能的中间路径，以及对应的动作和观察序列。这里的规划不是在表面 observation 上做，而是在去混叠后的 clone latent-state graph 上做，因为只有这个图区分了“这个 A”和“那个 A”不是同一个状态。

### 4.6 CSCG 能解释哪些细胞

CSCG 能解释很多 latent-state cells。splitter cells 和 lap cells 都可以理解为相同空间位置在不同任务上下文中的不同 latent states。

在 T-maze alternation task 中，动物在同一个中央通道上可能处于“即将左转”或“即将右转”的不同状态。splitter cells 就是在同一个空间位置上，根据未来行为或任务上下文不同而有选择性放电。

lap cells 则是在动物重复跑同一条路线时，编码“这是第几圈”。同一物理位置在第一圈、第二圈、第三圈中可能对应不同未来奖励结构，因此需要被表示为不同任务状态。

许多 hippocampal findings 都可以从 latent-state representation 的角度来理解：从基础的位置细胞，到随动物行为和任务上下文变化的复杂表征，本质上都在帮助系统区分“表面相同但未来不同”的状态。

CSCG 的优势是快速、灵活、可以直接从序列中建立去混叠的状态图，非常符合 hippocampus 快速建图的功能。局限是泛化能力弱：它能快速为当前环境建立 latent-state graph，但每个新环境大多需要 de novo 重新学习，不能很好地复用已学过的抽象结构。

这一节结尾要明确：CSCG 代表的是 hippocampus-as-map 的思路，即海马本身快速构建当前环境的关系图。它与后续 TEM/SMP 的关键区别在于：CSCG 把整个 latent space 放在海马内部推断；而 TEM/SMP 更强调皮层或内嗅输入已经携带可泛化的结构表征，海马负责把这些结构与具体经验绑定。因此，CSCG 的快速去混叠能力和 TEM/SMP 的跨环境泛化能力如果能结合，可能形成更完整的海马—皮层互补学习框架。


---

## 5. 第三类模型：路径积分模型 CANNs / VCOs —— 如何压缩结构并实现抽象移动

这一节要讲路径积分，但不要只写空间导航，而要突出其“结构压缩”意义。路径积分的基本含义是：系统根据自身运动信号持续更新当前位置，而不是每一步都依赖外界地标。换句话说，它做的是：

\[
\text{旧位置/旧状态}+\text{当前运动} \rightarrow \text{新位置/新状态}
\]

在物理空间中，这可以表现为累积 self-movement vectors，例如 head direction cells 提供的 north、south、east、west 等方向信息。这里的 north/east 不一定是地理意义上的绝对方向，而是动物内部坐标系统中的运动方向。若动物依次走：

\[
North+East+South+West
\]

理论上应该回到原点，因此可以写成：

\[
North+East+South+West=0
\]

这个例子说明，路径积分不是“看到什么就编码什么”，而是按照稳定的运动规则更新内部位置。因此文章说 path-integration maps 是 **inherently latent and abstract**：它们表示的是内部推断出来的位置或状态，而不是当前感觉观察本身；它们依赖的是规则，而不是某个具体感官画面。

这也解释了为什么路径积分与 latent state 有关。所谓“推断 latent state”，可以理解为弄清楚自己处在某个抽象空间中的哪里。在双房间任务中，全局 grid code 可以在一个统一坐标系中区分两个感觉相似的房间；在 T-maze 交替任务中，splitter cells 则把“同一个中央通道位置”区分为 left trial 中的位置和 right trial 中的位置。这里不是“左边的 splitter cell 在左边放电”，而是同一个物理位置会因为当前 trial 是左转还是右转而对应不同任务状态，因此不同 splitter cells 会选择性放电。

路径积分的意义不只是空间定位，也在于结构压缩。空间中的 east、west、north、south 有稳定含义；在家族关系中，parent、child、sibling 也可以被看作抽象动作。只要动作在不同位置具有一致含义，系统就可以通过累积动作推断未直接观察到的关系。

例如知道 Chloe 是 Bob 的 sibling，再知道 Alice 是 Bob 的 grandparent，就可以推断 Alice 也是 Chloe 的 grandparent。系统不必记住所有成对关系，只需掌握可复用的关系规则。直观地说：

\[
Sibling+Grandparent=Grandparent
\]

这就是路径积分作为认知地图机制的压缩作用。它把大量具体连接压缩成少量可组合的关系规则。

不过，并非所有图都适合路径积分。随机图中的边通常没有稳定语义，无法定义“同一个动作在所有节点上具有同样后果”。例如在随机图中，从节点 1 到节点 10 的边，以及从节点 10 到节点 19 的边，只能被描述为“走 1→10 这条边，再走 10→19 这条边”。这些边不像 East 或 Parent 那样有可迁移的共同含义，因此不能形成：

\[
\text{this action}+\text{that action}=\text{some other action}
\]

所以，路径积分不限于物理空间，但它要求图中的动作/边具有稳定、可重复、可组合的含义。物理空间和家族关系图适合路径积分；任意随机图通常不适合。

路径积分表征之所以有利于泛化，还因为它避免了普通图泛化中的一个计算难题。若用普通 graph 表示不同环境，跨环境泛化常常需要找到两个图之间的 perfect alignment，也就是判断一个图中的哪个节点对应另一个图中的哪个节点。这个问题在一般情形下是 NP-hard 的：节点越多，可能对应关系的组合数越快爆炸，实际计算会非常困难。因此，如果泛化依赖“把两个具体图完全一一匹配”，计算成本会很高。

路径积分式表征的优势在于：它不要求先把两个图的所有节点完美对齐，而是学习稳定的关系动作和组合规则。这里的 “all positions are treated equally” 不是说所有位置没有差别，而是说所有位置都服从同一套动作规则。例如在空间中，East 在任何位置都有相同含义；在家族关系中，Parent、Sibling、Niece 也可以作为关系动作。只要动作规则可复用，系统就可以在新环境中使用同一套关系结构。

因此，文章说 path-integration maps 可以 “chart the relational structure of one family just as well as for another”。这里的 chart 是动词，意思是“刻画、标出、组织出关系结构”，不是统计图表。它的意思是：同一套抽象关系地图可以刻画一个家族的关系结构，也可以刻画另一个家族的关系结构。具体成员变了，但 Parent、Sibling、Niece 这些关系动作及其组合规律仍可复用。

例如观察到：

```text
Daniel is Emily's parent
Fran is Daniel's sibling
```

对应的关系路径是：

```text
Emily --Parent--> Daniel --Sibling--> Fran
```

如果系统已经学会家族关系规则，就能进一步推断：

```text
Fran --Niece--> Emily
```

这可以写成：

\[
\mathbf a_{\text{Parent}}
+
\mathbf a_{\text{Sibling}}
+
\mathbf a_{\text{Niece}}
=
\mathbf 0
\]

这里的 \(\mathbf a_{\text{Parent}}\)、\(\mathbf a_{\text{Sibling}}\)、\(\mathbf a_{\text{Niece}}\) 是关系动作的向量或操作表征，不是人物身份标签。\(\mathbf 0\) 表示净位移为零，即经过这串关系动作后回到起点。它类似空间中的：

\[
\mathbf a_{\text{North}}
+
\mathbf a_{\text{East}}
+
\mathbf a_{\text{South}}
+
\mathbf a_{\text{West}}
=
\mathbf 0
\]

需要特别注意：Parent、Sibling、Niece 是边或动作，不是节点。Daniel 可以相对于 Emily 是 parent，也可以相对于 Fran 是 sibling，这不矛盾，因为“父母”“兄弟姐妹”“孩子”等身份都是相对于另一个节点定义的关系。节点是具体人或抽象状态，关系动作才是 Parent、Sibling、Child 等。

学习泛化通常也是 sequence-learning problem，但训练序列必须来自许多不同环境。只在一个家族里看到 Emily、Daniel、Fran，只能记住这个家族；在许多不同家族中反复看到类似的关系序列，系统才有可能抽象出 Parent、Sibling、Niece 的可迁移规则。这一点与 AI 中的 domain generalization 有相通之处：多个训练域不是为了让模型死记每个域，而是为了迫使模型抽取跨域稳定的结构规则。区别是，文章讨论的是生物/神经网络如何从序列中学习这种结构，而不是程序员手写规则。

这也解释了 hippocampal representations 和 entorhinal representations 在泛化上的差异。海马 place-cell 表征常发生 remapping：在一个环境中相邻的 place fields，换到另一个环境后不一定仍然相邻。也就是说，海马表征更容易绑定到具体情境。相反，内嗅皮层 grid cells 在同一 module 内的相对关系更稳定：换环境后整张 grid map 可以整体平移或旋转，即 realignment，但细胞之间的相对邻近关系仍保留。因此，entorhinal representations 更像可复用的结构坐标，而 hippocampal representations 更像具体环境中的情境绑定。


CANNs 和 VCOs 是两类典型路径积分机制。CANNs，即 continuous attractor neural networks，用连续吸引子网络中的活动模式移动来实现位置更新。其基本方程是：

\[
\tau \frac{d\mathbf g}{dt}
=
-\mathbf g+f(\mathbf W\mathbf g+\mathbf B\mathbf a)
\]

这里所有加粗符号都要按向量或矩阵理解。**\(\mathbf g\)** 是一组将被路径积分更新的神经元活动向量，不是单个细胞。可以写成：

\[
\mathbf g=[g_1,g_2,\cdots,g_n]
\]

每个 \(g_i\) 是第 \(i\) 个神经元的活动强度，整个 **\(\mathbf g\)** 表示神经空间中的当前状态或当前位置。所谓 “a vector of cells to be path-integrated” 不是说细胞本身被积分，而是这组细胞的活动模式会根据运动输入持续更新。

**\(\mathbf W\)** 是 recurrent weight matrix，表示神经元之间的循环连接；**\(\mathbf W\mathbf g\)** 是固定网络连接对当前活动模式的影响。**\(\mathbf a\)** 是 velocity input，表示当前运动方向和速度；**\(\mathbf B\)** 是把速度输入投射到神经元活动空间的矩阵，因此 **\(\mathbf B\mathbf a\)** 表示运动输入对神经活动模式的推动。若动物向某个方向移动，\(\mathbf B\mathbf a\) 会推动活动峰在神经空间中相应移动；若动物不动，\(\mathbf W\) 可以帮助活动峰稳定维持。

公式中的 \(-\mathbf g\) 表示当前活动自然衰减，\(f(\cdot)\) 是非线性激活函数，\(\tau\) 是时间常数。整条式子的直观意思是：神经活动一方面会衰减，另一方面又被 recurrent connections 维持和塑形，并被速度输入推动，从而实现路径积分。

文章还提到一个替代但较不符合生物实际的方程：

\[
\tau \frac{d\mathbf g}{dt}
=
-\mathbf g+f(\mathbf W_{\mathbf a}\mathbf g)
\]

这里速度信息不是通过额外输入项 \(\mathbf B\mathbf a\) 进入网络，而是通过改变 recurrent matrix **\(\mathbf W_{\mathbf a}\)** 进入模型。也就是说，不同运动速度或方向对应不同的 recurrent matrix，例如向东时用 \(\mathbf W_{east}\)，向西时用 \(\mathbf W_{west}\)。它在数学上可以表达速度调制，但生物学上不太合理，因为真实突触连接通常不会随着每一瞬间运动速度而大规模切换。相比之下，固定 **\(\mathbf W\)** 加上速度输入 **\(\mathbf B\mathbf a\)** 更像神经系统可能采用的方式。

在 CANN 中，如果设置合适的权重，网络可以执行路径积分；不同权重结构还能模拟不同细胞类型，例如 head direction cells、place cells 和 grid cells。这里所谓 “with different cell classes modeled with different weights” 的意思是，不同细胞类型可以由不同的 recurrent connectivity pattern 产生。

VCOs，即 velocity-coupled oscillators，是另一类路径积分模型。它不靠移动活动峰，而是用振荡相位差记录位移。可以粗略理解为：theta oscillation 是一个参考节律，velocity-dependent dendritic oscillation 是一个会被运动速度影响的节律。动物沿某个方向移动时，两种振荡之间的 phase difference 会逐渐积累，而这个相位差就表示沿该方向轴已经走了多远。

单个 VCO 只记录沿某一条轴的距离。若一个细胞只关心 \(x\) 轴，那么所有 \(x\) 值相同的位置对它来说相似；在二维空间中，这些位置连起来就是一条线。如果这个信号沿 \(x\) 轴周期性重复，二维活动图就会变成一组平行条纹。因此作者说这看起来像 plane wave。若有三组这样的条纹信号，偏好轴彼此相差：

\[
\frac{\pi}{3}=60^\circ
\]

三组周期性活动相加后，只有某些位置会让三组波峰同时较强，形成离散的高活动点。这些点排列成三角晶格，从每个点看周围最近邻点则呈现六边形对称性。因此，VCO 模型中 grid cell 可以被看作三个相差 60 度方向的单轴周期信号的总和。这里的“三个”和“60 度”来自六边形 grid pattern 的几何结构，而不是随意设定。

传统 CANNs/VCOs 的主要局限是，关键权重通常是研究者精心设定的，而不是从感觉经验中学习出来的。也就是说，它们解释了路径积分如何运行，但较少解释这些路径积分结构本身如何学出来。为了解决这一点，可以把路径积分设定成一个学习问题：让模型根据动作序列更新 latent state，再由 latent state 预测观察。

文章把这个学习问题写成：

\[
p(\mathbf X,\mathbf Z|\mathbf A)
=
p(\mathbf z_0)
\prod_t
p(\mathbf x_t|\mathbf z_t)
p(\mathbf z_t|\mathbf z_{t-1},\mathbf a_t)
\]

其中 **\(\mathbf A\)** 是整段动作序列，**\(\mathbf X\)** 是整段观察序列，**\(\mathbf Z\)** 是整段 latent-state 序列。左边的意思是：在整段动作序列已知的情况下，某一整段 latent-state sequence 和 observation sequence 出现的概率。

这个公式可以从链式法则和模型假设推出来。首先：

\[
p(\mathbf X,\mathbf Z|\mathbf A)
=
p(\mathbf Z|\mathbf A)p(\mathbf X|\mathbf Z,\mathbf A)
\]

接着拆第一项。完整链式法则本来会让当前状态依赖所有过去状态和整段动作序列：

\[
p(\mathbf z_t|\mathbf z_{0:t-1},\mathbf A)
\]

但路径积分模型假设，当前状态只需要上一状态和当前动作：

\[
p(\mathbf z_t|\mathbf z_{0:t-1},\mathbf A)
=
p(\mathbf z_t|\mathbf z_{t-1},\mathbf a_t)
\]

这里不是 \(\mathbf A\) 在代数上“变成” \(\mathbf a_t\)，而是模型假设整段动作序列中与第 \(t\) 步状态更新直接相关的只有当前动作 \(\mathbf a_t\)。过去动作的影响已经被 \(\mathbf z_{t-1}\) 总结，未来动作不应该影响当前状态。因此：

\[
p(\mathbf Z|\mathbf A)
=
p(\mathbf z_0)
\prod_t p(\mathbf z_t|\mathbf z_{t-1},\mathbf a_t)
\]

再拆第二项。模型假设当前观察只由当前 latent state 生成：

\[
p(\mathbf X|\mathbf Z,\mathbf A)
=
\prod_t p(\mathbf x_t|\mathbf z_t)
\]

把两部分相乘，就得到：

\[
p(\mathbf X,\mathbf Z|\mathbf A)
=
p(\mathbf z_0)
\prod_t
p(\mathbf x_t|\mathbf z_t)
p(\mathbf z_t|\mathbf z_{t-1},\mathbf a_t)
\]

这个公式和前面 CSCG 的公式相似，但动作的角色不同。CSCG 写的是：

\[
p(\mathbf X,\mathbf Z,\mathbf A)
=
p(\mathbf z_0)
\prod_t
p(\mathbf x_t|\mathbf z_t)
p(\mathbf z_t,\mathbf a_t|\mathbf z_{t-1})
\]

这里动作 **\(\mathbf A\)** 也是联合建模的变量之一，模型学习的是“从上一个 clone state 出发，会伴随什么 action 到达什么下一个 clone state”。而路径积分公式写成 \(p(\mathbf X,\mathbf Z|\mathbf A)\)，是因为动作序列被当作已知输入，模型学习的是“给定当前动作，如何从 \(\mathbf z_{t-1}\) 更新到 \(\mathbf z_t\)”。简化地说：

```text
CSCG：z_{t-1} → (z_t, a_t)
路径积分模型：z_{t-1}, a_t → z_t
```

路径积分部分：

\[
p(\mathbf z_t|\mathbf z_{t-1},\mathbf a_t)
\]

可以进一步用离散时间 RNN 实现：

\[
\mathbf z_t=f(\mathbf W\mathbf z_{t-1}+\mathbf B\mathbf a_t)+noise
\]

如果把 noise 设为 0，就得到 deterministic RNN：

\[
\mathbf z_t=f(\mathbf W\mathbf z_{t-1}+\mathbf B\mathbf a_t)
\]

所谓 deterministic RNN，就是给定同样的上一状态和同样的输入，总会得到同样的下一状态；如果有 noise，则同样输入下也可能得到略有差异的下一状态。

这些模型在被要求预测 ground truth spatial representations 时，可以学会路径积分。这里的 \(\mathbf x\) 可以是 place cell 活动，也可以是直接的 \(x,y\) 坐标。也就是说，网络输入运动序列，内部更新 latent state，再输出当前位置对应的标准答案；为了预测准确，它会学会“旧状态 + 当前运动 → 新状态”的更新规则。

训练后的神经单元常会形成 periodic representations，也就是在空间中周期性重复的活动图样。经典 grid cell 图样接近六重对称，像蜂窝或三角晶格；但一些 RNN 学出来的周期图样比较 amorphous，形状模糊、不规则，常偏向 fourfold symmetry，即更像方格或棋盘式周期结构。后续解析结果指出，从四重对称到六重对称的转变可以由一个 third-order regularization term 控制；这种性质在生物上可以通过“神经活动必须为正”这一约束实现。因为真实神经元放电率不能为负，非负性约束会改变最优图样，使六重对称的 grid-like pattern 更容易出现。

这一节结尾应说明：路径积分模型提供的是认知地图中的“可泛化结构动力学”。它使系统能在具有稳定动作规则的空间或抽象关系图中压缩状态关系，并用少量规则追踪自身位置。但单独的 CANNs/VCOs 仍不足以解释真实环境中结构如何从感觉经验中学习出来，因此需要进一步与序列学习、记忆绑定和跨环境泛化模型结合。

---

## 6. 第四类模型：TEM / SMP —— 路径积分、记忆绑定与结构泛化的整合

这一节应该是四个模型部分中的高潮，因为 TEM/SMP 是最接近整合框架的一类模型。它们试图把前面几类模型的优点合在一起：CSCG 能从感觉序列中快速学习 latent states，但泛化弱；路径积分模型能学习可复用的结构规则，但还需要知道抽象结构如何对应到当前环境中的具体感觉对象。TEM/SMP 的基本思想就是：用可复用的抽象路径积分模块表示结构，再用关系记忆把这个抽象结构同具体经验绑定起来。

### 6.1 为什么需要“抽象结构 + 记忆绑定”

TEM/SMP 要解决的问题是：如何在多个结构相似但感觉内容不同的环境之间泛化。例如多个家族的关系结构可能相似，但具体成员不同；多个二维空间都遵守相似的空间转移规则，但物体、地标和视觉输入不同。模型不能只记住某个具体环境，也不能只有空的抽象规则；它既要复用结构，又要快速适配当前环境。

文章前面写过路径积分模型：

\[
p(\mathbf X,\mathbf Z|\mathbf A)
=
p(\mathbf z_0)
\prod_t
p(\mathbf x_t|\mathbf z_t)
p(\mathbf z_t|\mathbf z_{t-1},\mathbf a_t)
\]

其中 \(p(\mathbf z_t|\mathbf z_{t-1},\mathbf a_t)\) 表示给定上一抽象状态 \(\mathbf z_{t-1}\) 和动作 \(\mathbf a_t\)，模型如何更新到新的抽象状态 \(\mathbf z_t\)。这部分是结构动力学。问题在于，若直接让每个抽象位置 \(\mathbf z\) 固定预测一个感觉观察 \(\mathbf x\)，就会把抽象结构和具体内容焊死。例如：

```text
z_A 永远预测 Emily
z_B 永远预测 Daniel
z_C 永远预测 Fran
```

这样模型只能适用于这个具体家族。换成另一个家族时，同样的抽象结构可能应该对应：

```text
z_A → Alice
z_B → Bob
z_C → Chris
```

因此，旧式固定映射的问题不是“没有规则”，而是“同一个抽象位置无法在不同环境中对应不同感觉对象”。TEM/SMP 引入关系记忆 \(\mathbf M\)，正是为了让结构和内容分离。

### 6.2 \(\mathbf M\) 到底是什么：不是身份标签，而是当前环境的变量绑定机制

\(\mathbf M\) 可以先理解成当前环境中的“地址簿”或“绑定表”。它记的不是：

```text
Daniel = parent
Fran = sibling
Emily = niece
```

这种写法是错的，因为同一个人可以相对于不同人有不同关系。Daniel 可以是 Emily 的 parent，也可以是 Fran 的 sibling，还可以是另一个人的 child。Parent、Sibling、Child 这些不是节点身份，而是关系边或动作。

更准确地说，\(\mathbf M\) 记的是：当前环境中，某些抽象节点/状态 \(\mathbf z\) 与具体感觉对象 \(\mathbf x\) 的临时对应关系。例如在一次具体经验中：

```text
抽象结构：
z_A --Parent--> z_B --Sibling--> z_C --Niece--> z_A

当前家族的 M：
z_A ↔ Emily
z_B ↔ Daniel
z_C ↔ Fran
```

换一个家族，抽象结构仍可保留，但 \(\mathbf M\) 改变：

```text
同一抽象结构：
z_A --Parent--> z_B --Sibling--> z_C --Niece--> z_A

另一个家族的 M：
z_A ↔ Alice
z_B ↔ Bob
z_C ↔ Chris
```

所以，\(\mathbf M\) 不是泛化规则本身，而是让泛化规则能落到当前具体环境上的接口。结构规则负责“怎么推”；\(\mathbf M\) 负责“当前这些具体东西对应结构里的哪些变量”。如果写成程序，它确实可以像一个字典；文章的重点是，在脑或神经网络模型里，这种字典式快速绑定不能由程序员手写，而要由模型从经验中快速形成。

这也说明，\(\mathbf M\) 不会凭空知道陌生环境里的成员是谁。新环境中，如果模型只看到一个新成员 Daf，而没有观察到他与其他成员的关系，它不能知道 Daf 在抽象结构中的位置。只有在观察到类似：

```text
Daf is ZZZ's sister
```

之后，模型才可能把 Daf 接入当前环境的关系结构。如果又观察到：

```text
Kkk is also ZZZ's sister
```

那么 “ZZZ + Sister” 就不是唯一转移，而是多个候选：

\[
\mathbf z_{\text{ZZZ}}
+
\mathbf a_{\text{Sister}}
\rightarrow
\{\mathbf z_{\text{Daf}},\mathbf z_{\text{Kkk}}\}
\]

此时模型应该输出集合或概率分布，而不是唯一答案。也就是说，\(\mathbf M\) 不负责消除所有一对多关系的歧义；这类歧义还需要任务上下文、额外线索或概率推断。\(\mathbf M\) 的作用更基本：把已经观察到的具体对象接到可复用的抽象结构上。

### 6.3 “抽象位置”到底是什么

这里的“抽象位置” \(\mathbf z\) 不是 parent、son、sibling 这些角色标签，而是关系空间中的节点状态。关系动作才是 \(\mathbf a_{\text{Parent}}\)、\(\mathbf a_{\text{Child}}\)、\(\mathbf a_{\text{Sibling}}\)。更准确的结构是：

\[
\mathbf z_{\text{Emily}}
\xrightarrow{\mathbf a_{\text{Parent}}}
\mathbf z_{\text{Daniel}}
\]

这表示从 Emily 这个节点出发，沿着 Parent 关系动作，走到 Daniel 这个节点。Daniel 同时还可以满足：

\[
\mathbf z_{\text{Daniel}}
\xrightarrow{\mathbf a_{\text{Sibling}}}
\mathbf z_{\text{Fran}}
\]

因此，Daniel 并不是固定等于 parent 或 sibling；Daniel 是节点，Parent/Sibling 是边或动作。同一个具体人在不同关系中可以扮演不同角色，因为这些角色都是相对于另一个节点定义的。

在同一个家庭里，\(\mathbf z_A\)、\(\mathbf z_B\) 的绑定也不是永久固定的。若当前路径是：

```text
Emily --Parent--> Daniel
```

可以有：

```text
z_A ↔ Emily
z_B ↔ Daniel
```

但若当前路径是：

```text
Daniel --Parent--> Emily
```

也完全可能有：

```text
z_A ↔ Daniel
z_B ↔ Emily
```

所以 \(\mathbf z_A\)、\(\mathbf z_B\) 更像当前抽象推演中的状态槽位，而不是现实世界里永久固定的个体编号，更不是“parent/child”的身份标签。

### 6.4 如何用 \(\mathbf M\) 预测未来感觉观察

原文说：预测接下来的感觉观察时，只需要先在抽象表征中想象一次转移，然后在到达的位置检索记忆。更具体地说，流程是：

\[
\mathbf z_{t-1},\mathbf a_t
\rightarrow
\mathbf z_t
\]

然后利用当前环境的记忆：

\[
(\mathbf z_t,\mathbf M)
\rightarrow
\mathbf x_t
\]

第一步是结构推演，第二步是当前环境中的记忆检索。例如当前有：

```text
抽象结构：
z_A --Parent--> z_B

M：
z_A ↔ Emily
z_B ↔ Daniel
```

模型当前在 \(\mathbf z_A\)，执行 Parent 动作，抽象结构告诉它到达 \(\mathbf z_B\)。随后查 \(\mathbf M\)，发现当前环境中 \(\mathbf z_B\) 绑定的是 Daniel，于是预测接下来会看到 Daniel。这里的关键不是“\(\mathbf M\) 神奇地知道 Daniel 是 parent”，而是：模型已经通过观察把 \(\mathbf z_B\) 与 Daniel 形成了当前环境的绑定。

如果目标只是做抽象推理，例如：

```text
A --Parent--> B
B --Sibling--> C
所以 C --Niece--> A
```

那么只要结构规则就够了。但文章中的模型任务是预测 sensory observations，也就是实际会看到什么对象、图像、声音或符号。因此，仅有结构规则只能得到“到达抽象节点 B”；要输出“Daniel”，就必须有当前环境中 \(\mathbf z_B\) 与 Daniel 的绑定。没有 \(\mathbf M\)，抽象规则只是空模板；没有抽象规则，\(\mathbf M\) 只是死记当前环境。

因此，\(\mathbf M\) 的作用可以概括为：

```text
结构规则：负责跨环境泛化
M：负责当前环境实例化
二者合起来：才能在新环境中快速预测具体感觉观察
```

### 6.5 为什么加了 \(\mathbf M\) 后模型更强大

加了 \(\mathbf M\) 并不是说模型自动解决所有问题，而是模型的预测方式发生了变化：从固定的 \(\mathbf z\rightarrow\mathbf x\)，变成了 \((\mathbf z,\mathbf M)\rightarrow\mathbf x\)。这样，\(\mathbf z\) 可以保持为抽象结构位置，\(\mathbf x\) 则由当前环境记忆决定。结构和内容分离后，模型才有机会在多个环境中学习可复用结构，同时快速适配新环境。

这也是原文说 “This is more powerful” 的含义。以前的 path-integration 模型常被要求预测整理好的空间坐标或 place-cell representation，因此容易学到空间路径积分。TEM/SMP 更进一步：直接预测真实感觉观察。如果感觉对象按二维空间排列，模型会学到空间路径积分；如果感觉世界有更复杂依赖，比如任务阶段、轨迹历史、家族关系或证据积累，模型为了预测未来感觉输入，就会学习相应的 latent-state map，并在这个潜在空间中使用路径积分。

换句话说：

```text
简单空间环境 → 学二维空间地图
家族关系环境 → 学关系空间地图
T-maze 交替任务 → 学 left/right trial 的 latent-state map
证据积累任务 → 学 position × evidence 的 latent-state map
```

这不是 \(\mathbf M\) 单独做到的，而是“抽象路径积分模块 + 关系记忆 \(\mathbf M\) + 感觉预测目标”共同造成的结果。

### 6.6 MEC、LEC、HPC 的分工与图 a 的理解

神经解释上，MEC 更偏向 structural code，即可复用的抽象结构表征；LEC 更偏向 sensory code，即具体感觉对象或内容；HPC 则形成 conjunctive code，把二者结合起来。

在 Box 4 图 a 中，下方的网格表示 MEC 的结构代码，可以理解为抽象地图或关系空间；上方的 sensory stimuli / sensory code 表示 LEC 的具体内容，例如某些图标、物体或人物；中间的连线和圆圈表示 HPC 的 conjunctive code，即“某个抽象位置与某个具体感觉对象”的组合。右侧圆点矩阵是同一思想的抽象画法：纵向是 LEC sensory code，横向或下方是 MEC structural code，每个圆点代表一种 LEC × MEC 的组合。

这张图表达的是：不同环境中，MEC 的结构成分可以复用，LEC 的感觉对象表征也可以复用，但 HPC 中的组合可以改变。Environment 1 中，一个 MEC 结构位置可能与某个 LEC 对象组合；Environment 2 中，同一个结构位置可以与另一个 LEC 对象组合。海马 remapping 在这里不是泛化的障碍，而是实现不同环境绑定的一种方式：HPC 组合变了，但 MEC/LEC 的基础成分可以复用。

因此，海马在 TEM/SMP 中不是简单“储存整张地图”，也不是简单“储存标签”，而是在当前环境中把抽象结构和具体感觉内容临时组合起来。这个 conjunctive representation 在真实 hippocampal neurons 中也很常见：同一个海马神经元或神经元组合可以同时反映抽象位置和感觉预测。

### 6.7 TEM 与 SMP 的实现差异

虽然 TEM 和 SMP 在概念上类似，但实现不同。TEM 的输入更抽象：它直接获得 allocentric actions 和 object representations。Allocentric actions 指以外部世界或地图为参考系的动作，例如 East、West、North、South；object representations 指已经加工好的对象表征。SMP 则从 egocentric input 和 pixels 出发，即从第一人称视角和像素输入中自己推断物体、朝向和世界坐标。因此，SMP 还会产生参与 egocentric-to-allocentric coordinate transformation 的细胞表征。

第二个差异是记忆实现。SMP 使用机器学习中的 memory network；TEM 使用更接近生物机制的 Hebbian learning 和 Hopfield networks。Hebbian learning 可以理解为“一起活动的神经元连接增强”；Hopfield network 则是一种联想记忆网络，给出部分线索时可以恢复完整记忆模式。由于 TEM 受这种生物约束，抽象世界和感觉世界的链接不能只是外部数据库，而必须发生在神经元活动单元中。也就是说，同一批 hippocampal neurons 必须同时包含抽象位置和感觉预测的信息。

这就是 TEM 中 conjunctive representation 的含义。例如某个海马表征不是单独表示“抽象位置 B”，也不是单独表示“Daniel”，而是表示“当前环境中抽象位置 B 与 Daniel 的组合”。换一个环境时，MEC 的抽象位置仍可复用，LEC 的感觉对象表征也可复用，但 HPC 的组合发生 remapping。

### 6.8 TEM/SMP 能重现哪些神经表征

TEM 和 SMP 都是深度人工神经网络。它们在学习结构知识泛化的过程中，会重现许多已知的 hippocampal cognitive map 表征。这里的 recapitulate 指“再现、模拟出”；a host of 指“一大批”。也就是说，这些模型不是把每一种细胞类型硬编码进去，而是在学习预测和泛化任务时自然形成类似表征。

TEM 可以学到 compositional entorhinal representations。Compositional 的意思是：表征不是为每个完整环境死记一整块，而是由可复用成分组合而成。例如二维空间结构、边界、物体、目标、关系动作等都可以作为基元；非空间任务中，Parent、Sibling、Child 等关系也可以作为可组合成分。这样，模型可以用少量结构基元组合出许多不同任务或环境。

SMP 因为从 egocentric inputs 和 pixels 工作，所以会学出与自我中心坐标到外部世界坐标转换有关的细胞，例如 head-direction cell、egocentric boundary-vector cell、border cell 和 place cell。TEM/SMP 图中的 grid cell、border cell、object-vector cell、place cell、landmark cell 等，说明模型在结构泛化过程中能产生类似真实海马—内嗅系统的表征。

TEM 还可以解决经典 relational memory tasks，例如 transitive inference。传递性推理的形式是：

```text
A > B
B > C
所以 A > C
```

它要求系统把多条关系组织成结构，并推出未直接观察到的关系。文章强调这类任务依赖 hippocampal formation，因此 TEM 能解决这类任务，是其作为海马关系记忆模型的重要证据。

这一节最后可以总结：TEM/SMP 的贡献不是单独提出一个记忆表，而是把路径积分、结构泛化、感觉预测和海马记忆绑定整合到同一框架中。MEC 提供可复用结构，LEC 提供具体感觉内容，HPC 形成二者的 conjunctive code；换环境时，结构可以保留，具体绑定可以快速更新。

---

## 7. 四类模型之间的真正区别

这一节适合做成笔记中的核心对照段，而不是简单表格。可以围绕几个问题来组织。

第一个问题是：模型是否解释状态空间如何被学习？SR/DR 通常假定已有状态空间，然后说明它如何用于价值计算；CSCG 和 TEM/SMP 更直接解释状态空间如何从序列中学习。

第二个问题是：模型是否能解决 aliasing？CSCG 明确解决，TEM/SMP 也能通过记忆和抽象结构处理；SR/DR 本身不重点解决；CANNs/VCOs 则主要处理连续空间中的位置更新。

第三个问题是：模型是否支持泛化？CSCG 快速但泛化弱；路径积分通过稳定动作规则支持结构泛化；TEM/SMP 通过抽象结构与记忆绑定实现跨环境泛化；SR/DR 的泛化更多体现在 value 和 policy 层面。

第四个问题是：海马在模型中扮演什么角色？SR/CSCG 偏向 hippocampus-as-map，TEM/SMP 偏向 hippocampus-as-memory and cortex-as-map，路径积分模型则多与内嗅皮层和 grid system 相关。

第五个问题是：模型服务于认知地图哪一层？SR/DR 偏行为和 RL 接口，CSCG 偏状态发现，路径积分偏结构动力学，TEM/SMP 偏泛化与记忆整合。

---

## 8. 文章的整合观点：海马既能快速建图，也能帮助皮层学习抽象结构

这一节是全文从模型比较走向理论综合的关键。前面几个模型看起来像是在给海马安排不同角色：SR/CSCG 更像是 **hippocampus-as-map**，认为海马神经元及其连接可以直接表示状态、位置和状态之间的关系；TEM/SMP 更像是 **hippocampus-as-memory and cortex-as-map**，认为可泛化的地图主要由内嗅皮层或其他皮层区域表示，而海马负责把抽象地图位置同具体感觉经验绑定起来。

这两种观点不是简单的概念差异，而是功能差异。hippocampus-as-map 的优势是快：一个新环境或新任务出现时，海马可以迅速构建当前可用的、去混叠的状态空间。缺点是泛化弱，因为这种状态图通常是为当前环境特化的。cortex-as-map / hippocampus-as-memory 的优势是泛化强：皮层一旦学会了二维空间、家族关系、任务规则或目标/边界/物体这类抽象结构，就能在新环境中复用；缺点是皮层学习这种抽象结构通常慢。

作者的整合方案是：真实海马可能同时具有这两类功能，但在不同学习阶段中侧重不同。新环境刚出现时，皮层还没有可用的泛化地图，海马先像 CSCG/SR 那样快速建立一个当前能用的状态图；随着经验增加，海马把这些去混叠的状态序列通过 replay 离线提供给皮层，皮层逐渐学习可复用结构；当皮层结构学好以后，海马更多地承担记忆绑定功能，把当前具体经验接到已有抽象地图上。

可以把这个动态过程写成：

```text
早期 / 新奇经验：HPC as map
    快速建立当前环境的状态图，解决 aliasing，支持即时行为

经验增加后：cortex as map + HPC as memory
    皮层学到可泛化结构，海马负责把具体感觉内容绑定到结构位置上
```

这也解释了图 3a。图中从左到右可以理解为经验增加：早期海马更多作为 map，后期海马更多作为 memory。图 3b 则说明 CSCG 和 TEM 在形式上可以整合：CSCG 给出海马内部状态之间的预测关系，TEM 给出 MEC/LEC/HPC 之间的抽象结构—感觉内容绑定。整合后的模型中，海马既能形成当前环境的状态图，也能作为记忆系统把皮层地图位置和具体经验绑定起来。

这里的“海马表征”通常不是指单个细胞，而是指海马神经元群体形成的状态代码。可以写成：

\[
\mathbf h_t=[h_{t,1},h_{t,2},\cdots,h_{t,n}]
\]

其中 \(\mathbf h_t\) 是第 \(t\) 时刻的海马群体活动向量，\(h_{t,i}\) 是第 \(i\) 个细胞的活动强度。这个群体代码可以表示位置、任务阶段、时间、感觉内容或它们的组合。整合模型中的关键变化是：海马不只是回答“这里见过什么”，还可以预测“从当前海马状态接下来会到哪个海马状态”。如果用符号帮助理解，可以写成：

\[
p(\mathbf h_{t+1}\mid \mathbf h_t,\mathbf a_t)
\]

这里 \(\mathbf h_t\) 和 \(\mathbf h_{t+1}\) 是海马状态向量，\(\mathbf a_t\) 是动作/转移输入向量。这不是文章给出的完整模型公式，而是帮助理解“海马状态预测未来海马状态”这一整合思想。

---

## 9. Replay：离线训练、建图与 credit assignment

Replay 不能只理解为“把过去经历再播放一遍”。在这篇文章的整合框架中，replay 至少有三层作用：第一，它把海马快速学到的去混叠状态序列提供给皮层，帮助皮层学习更可泛化的结构；第二，它可以离线构建状态—目标/价值关系，降低在线行为的计算负担；第三，它把传统 RL 中的 credit assignment 和状态空间构建统一起来。

如果动物在新环境中发现一个奖励位置，在线行为时它只是实际走到了奖励处。但为了之后能更快行动，系统最好让其他重要状态也“知道”自己与奖励的关系。例如起点不应只是“起点”，还应包含“目标在东北方向若干距离”这样的向量关系。这种关系可以通过 GVCs 表示。GVCs 编码的是当前状态相对于目标的距离和方向，而不是单纯在目标位置放电。

离线 replay 可以从奖励位置向外展开，通过路径积分逐步计算每个被 replay 到的状态与奖励之间的相对关系，并把对应的 GVC 绑定到相应的海马/皮层位置上。直观流程是：

```text
动物在线发现目标 G
    ↓
离线 replay 从 G 向外展开轨迹
    ↓
路径积分计算每个状态相对于 G 的向量
    ↓
把正确的 GVC 绑定到对应状态
    ↓
下次动物回到某状态时，该状态已经包含通向奖励的向量/价值信息
```

如果从某状态 \(s\) 指向目标 \(G\) 的位移写作 \(\Delta\mathbf x_{s\to G}\)，那么 replay 的作用就是在离线阶段把这种相对向量关系写入状态表征。更完整的状态向量可以理解为：

\[
\mathbf s=[\mathbf g,\mathbf p,\mathbf v_{\mathrm{goal}},\mathbf v_{\mathrm{object}},\mathbf v_{\mathrm{border}}]
\]

其中 \(\mathbf g\) 是 grid 表征向量，\(\mathbf p\) 是 place 表征向量，\(\mathbf v_{\mathrm{goal}}\) 是目标向量表征，\(\mathbf v_{\mathrm{object}}\) 和 \(\mathbf v_{\mathrm{border}}\) 分别表示物体和边界相关向量表征。这样，状态不只是“我在哪里”，而是已经包含“我相对于目标、物体、边界在哪里”。

这与传统 RL 中的 replay 观点相连。传统上，replay 可以被看作把奖励价值分配给已有状态，即 credit assignment；也可以被看作从经验中建立状态空间。作者的泛化框架把二者统一为一个过程：用预先学好的 bases（grid cells、GVCs、OVCs、BVCs 等）组合出当前任务需要的状态空间。也就是说，replay 同时在做状态空间构建和价值/策略结构的离线写入。

图 3e 正是这个意思：动物先从 Start 走到目标，随后 replay 可以离线建立从 Start 或其他关键位置指向 reward 的向量表征。下次动物回到 Start 时，不必重新探索，也不必把所有目标位置都保持在工作记忆中，因为状态表征本身已经包含与目标的关系。replay 因此是“离线构建未来可用地图”的机制，而不只是过去经验的复制。

---

## 10. 行为与神经表征：为什么不同细胞类型可以被统一解释

文章在 “New interpretations, integrations and predictions” 中从模型介绍转向整合解释。核心思想是：许多看似不同的海马细胞现象，都可以统一理解为 **latent state representations for disambiguation and generalization**。也就是说，海马不是只表示“我在物理空间哪里”，还表示“我在任务状态空间哪里”。

在 T-maze alternation task 中，动物按照左、右、左、右的顺序交替选择。物理上，同一个中央通道是同一个位置；任务上，中央通道在 left trial 和 right trial 中对应不同未来。为了预测未来，系统必须把同一物理位置拆成不同 latent states：

\[
\text{central trunk before left turn}
\neq
\text{central trunk before right turn}
\]

这就是 splitter cells 的含义。splitter cells 不是单纯编码“左边”或“右边”，而是在同一个物理位置上根据当前轨迹、trial 类型或未来行为不同而差异性放电。文章所说的 **big-loop** 是把这个交替任务展开成一个循环的任务状态空间：一半表示 going left phase，另一半表示 going right phase。place cells 负责表示物理空间位置，splitter cells 负责表示这个 big-loop 中的任务状态。

图 2a–c 展示了这一点。更复杂的空间交替任务也可以被描述成 big-loop latent state-space，真实海马和 TEM 模型中都能看到 place cells 与 trajectory-dependent / splitter-like cells。这里的重点不是每个实验细节，而是：同一个空间位置在不同任务阶段中会预测不同未来，因此需要被拆成不同 latent states。

跑圈任务中的 lap cells 也是同一原则。动物在同一环形路线中反复经过相同物理位置，但第 1 圈、第 2 圈、第 3 圈、第 4 圈对应不同奖励未来。于是状态不是单纯的物理位置，而是：

\[
\text{state}\approx(\text{position},\text{lap number})
\]

这不是严格的代数等式，而是说明状态由空间位置和圈数共同决定。lap cells 编码的就是这个非空间任务维度。

图 2e 更抽象：动物在 T-maze 中沿中央通道前进，同时积累左右 sensory cues 的证据，最后根据哪边线索更多选择左右。这里的 latent state-space 由 position 和 evidence 共同张成：

\[
\text{state}\approx(\text{position},\text{evidence})
\]

纵向 position 表示动物沿中央通道走到哪里，横向 evidence 表示当前线索差异偏向左还是右。刚开始 evidence 接近 0，越往后看到的 cue 越多，evidence 可能越偏左或偏右，所以图中的状态空间呈现展开形态。真实海马细胞和 TEM 学到的细胞都不是只编码 position，也不是只编码 evidence，而是编码二者的组合。

这一节还要理解 **composition**。泛化不一定是整张地图迁移，也可以是子组成部分的迁移。不同形状的房间可以被拆成“底层二维空间 + 可放置在任意位置的墙”；不同任务也可以拆成空间、边界、物体、目标、奖励等成分。OVCs、BVCs、GVCs、reward cells 是局部 basis representations，编码相对于物体、边界、目标或奖励的局部结构；grid cells 是全局 basis representations，因为它们在整个空间中提供全局结构骨架。

因此，不同细胞类型可以统一理解为：为了预测未来、消除混叠、支持泛化，大脑把状态空间拆成不同层级和不同成分。place cells、splitter cells、lap cells、evidence cells、GVCs、OVCs、BVCs 并不是孤立现象，而是不同任务需求下形成的状态表征或组合基元。

---

## 11. Factorized vs entangled representations：什么时候需要可组合表征

这一节解释图 3f，也解释为什么同一类神经表征有时看起来可组合，有时又会被具体任务“拉歪”。

**Factorized representation** 指不同变化因素被相对独立地表示。例如空间由 grid cells 表示，目标由 GVCs 或 reward-vector cells 表示，物体由 OVCs 表示，边界由 BVCs 表示。这样，状态向量可以理解为多个成分的组合：

\[
\mathbf s=[\mathbf g,\mathbf v_{\mathrm{goal}},\mathbf v_{\mathrm{object}},\mathbf v_{\mathrm{border}}]
\]

这里每个加粗符号都是表征向量。因子化的好处是泛化：目标换位置时，不必重写整个空间表征，只需要更新目标相关成分；边界或物体变化时，也可以更新对应成分。

**Entangled representation** 则是多个因素混在一起形成一个针对当前任务高度特化的表征。grid cells 曾经被认为是纯空间、因子化的表征，因为它们似乎忽略感觉细节和奖励信息。但实验显示，在稳定奖励位置附近，grid pattern 可以朝奖励位置扭曲，即 warped grid。这说明空间表征和奖励表征有时会纠缠在一起。

作者认为，是否 factorized 取决于泛化压力。如果动物长期重复同一个任务，奖励和空间总是以固定组合出现，那么存储一个专门的 warped representation 可能更高效。它泛化差，但适合当前固定任务。相反，如果任务经常切换，奖励位置、边界、物体和空间布局经常以不同组合出现，那么大脑更需要 factorized / compositional bases，因为这样才能快速重组。

可以把图 3f 理解为：低环境切换、长时间重复任务时，表征更可能进入 entangled phase；高环境切换、泛化压力强时，表征更可能进入 factorized phase。作者的可检验预测是：当奖励和空间经常任意组合时，应同时看到相对独立的空间表征和奖励向量表征；当奖励和空间总是固定组合时，一个定制的 warped grid 就足够。

这一节与 TEM/SMP 的 compositionality 直接相关。TEM/SMP 能泛化，依赖的不是死记每个完整环境，而是把空间、目标、物体、边界、感觉内容和任务结构分解成可复用成分，并在当前环境中快速绑定。

---

## 12. 时间状态、time cells 与 representational drift

文章进入 Open questions 后，首先讨论时间。前面的许多模型默认学到的表征会随时间稳定，但这显然不完全成立：我们可以记住同一地点、同一条件下发生在不同日期的事件；实验上也观察到，同一空间位置或同一环境中的海马细胞群体表征会随天数和经验缓慢变化，这一现象通常被称为 **representational drift**。但文章进一步提出，这种 drift 可能并不是随机漂移，而是“伪装成漂移的 remapping”：如果海马表征同时绑定空间、感觉和时间，那么即使空间和感觉保持不变，只要时间代码推进，整体海马表征也会换到新的细胞组合上。因此，所谓 drift 可能本质上是 **temporal remapping**。

这里的“海马表征漂移”通常指群体活动模式变化，而不是某个单细胞简单漂移。若同一位置 \(X\) 在 Day 5 的海马群体活动是：

\[
\mathbf h_{X,5}=[h_1,h_2,\cdots,h_n]
\]

而 Day 20 变成：

\[
\mathbf h_{X,20}=[h'_1,h'_2,\cdots,h'_n]
\]

外部位置可能相同，但海马群体代码不同。作者的问题是：如果细胞基础在变，为什么空间记忆还能稳定？

泛化模型给出的解释是：海马表征是多因素绑定的 conjunctive code，不是只表示空间。可以粗略写成：

\[
\mathbf h=F(\text{space},\text{sensory input},\text{time},\text{task context})
\]

这里 \(F\) 是组合/绑定函数，不是简单加法。如果 space 和 sensory input 不变，但 time code 改变，整个海马组合表征仍然可以改变。于是，从只看空间的人看来，这是 drift；从模型角度看，它可能是 **temporal remapping**，即变化的不是空间或感觉，而是时间维度。

所以 “representational drift is remapping in disguise” 的意思是：表面上看细胞表征在慢慢漂移，但它可能本质上是 remapping 的一种特殊形式，只不过传统 remapping 是环境、感觉或空间上下文变了，而这里是时间代码变了。若时间也是认知地图的一部分，那么 Day 5 的“同一位置”和 Day 20 的“同一位置”在完整状态空间中并不完全相同。

这带来一个预测：如果 drift 是由时间代码推进造成的，那么漂移顺序不应是随机噪声，而应沿着某种时间维度有结构地变化。

Time cells 也可以按 latent state 理解。在 delay task 中，动物物理位置可能不变，但任务进度在变化：刚进入 delay、delay 中段、delay 快结束，这些状态预测的未来不同。time cells 因此不只是“秒表细胞”，而是表示任务中的 temporal latent states。更准确地说，在 delay period 中，space 不变，但 position in task 在变；time cells 编码的是 overall task position。

图 4a–b 表达这一点：完整 latent state-space 不仅包括空间轨道，也包括 delay period 中的一串时间状态。图 4c 展示 drift 现象，图 4d 则给出解释：如果海马绑定 structural code、sensory stimuli 和 time code，那么时间代码变了，就会使海马组合表征换到新细胞上，同时空间结构仍可保持稳定。

---

## 13. 海马与高阶皮层：从空间地图到任务、语言、逻辑和数学

这一节讨论更高阶的开放问题：认知地图原则能不能从空间和简单任务推广到任务层级、语言、逻辑、数学和视觉概念。

首先是空间结构与任务结构的交互。图 2 中的 T-maze 任务虽然同时有空间和任务因素，但这些因素是固定组合的。动物在特定 T-maze 中学会 left/right alternation，并不意味着换成 W-maze 后可以直接泛化。现实中更复杂的例子是做菜：同一个菜谱可以迁移到不同厨房，烤箱的位置可以变，砧板的位置可以变，但“先切菜、再去烤箱”等任务结构仍可复用。

这要求空间表征和任务表征被 factorized。内嗅皮层可能更接近与物理环境互动有关的空间/结构表征，前额叶皮层尤其 mPFC 可能表示更抽象的任务位置，例如“现在处在菜谱中去烤箱之前的阶段”。海马则可以把二者绑定起来：mPFC 的 go-to-oven 表征必须连接到当前厨房中烤箱的空间位置，或连接到指向烤箱的 vector cells，动物才能真正导航到烤箱。

图 4e 表示一个 hierarchical TEM：在 LEC/MEC/HPC 之外加入 mPFC 模块。mPFC 提供抽象任务上下文，MEC/LEC/HPC 负责空间、感觉和记忆绑定。图 4f 给出预测：应该存在 contextually modulated vector cells，即受任务上下文调制的向量细胞。例如同一批 vector cells 可以在“去烤箱”阶段指向 oven，在“切菜”阶段指向 chopping board。这类似 splitter cells，因为同一空间环境在不同任务上下文下被不同方式表征；但它比 T-maze splitter cells 更一般，可以发生在空间中任意位置。

最后，作者把问题推广到其他认知领域。本文描述的模型把“构建地图”转化为“理解序列结构”。空间导航是序列，任务执行是序列，语言、数学和逻辑也包含序列结构。RNN、LSTM 和 Transformer 等序列学习模型能处理语言、数学和逻辑，是因为这些任务都涉及内容与结构的组合：语言中是词语与语法规则，数学中是数字与运算符，逻辑中是命题与推理规则。

数学算子可以类比为空间中的动作。加法和减法类似数轴上的 forward/backward actions：

\[
3+2=5
\]

可以理解为从状态 3 沿“+2”这个动作前进到状态 5；

\[
5-2=3
\]

可以理解为从状态 5 沿“-2”这个动作后退到状态 3。这里不是说数学真的等于空间导航，而是说某些抽象操作也可以被理解为在结构化状态空间中的变换。

对于看起来非序列的认知问题，作者也保持谨慎。理解足球和地球都是球体，并不明显需要序列转移。但仍可以把对象的生成因素，例如大小、形状、颜色，放到一个 manifold 上；在这个流形上，add-red、bigger、remove-red、smaller 这类“动作”有意义。静态图像本身不是序列，但观看图像的眼跳过程可以形成观察序列；这与图像观看时内嗅皮层出现 grid-like cells 的证据相呼应。

需要强调的是，作者不是说内嗅皮层表示所有结构，也不是说海马单独负责语言或数学。更合理的理解是：其他脑区很可能负责更抽象的结构表征，而它们与海马的交互可能遵循相似计算原则，例如状态空间、因子化、路径积分、泛化和快速记忆绑定。内嗅皮层在解剖上适合向海马提供丰富分布式表征，因此可能在许多情况下把结构信息 relay 给海马。

---

## 14. 文章结论：走向整合的 computational ontology

文章最后的核心不是说“认知地图就是空间地图”，而是反过来追问：对大脑来说，什么才算空间？如果空间的本质是一个可学习、可转移、可用于预测和行为控制的状态结构，那么许多非空间任务也可以被放进同一套计算语言里理解。

全文可以收束为一个层级化框架：SR/DR 说明已有状态空间如何服务 RL、价值计算和规划；CSCG 说明如何从序列中快速学出一个去混叠的 latent-state graph；CANNs/VCOs 说明路径积分如何用稳定动作规则压缩结构并支持抽象移动；TEM/SMP 说明可泛化结构如何通过海马记忆绑定到具体感觉经验中；replay、GVCs、OVCs、BVCs、time cells、representational drift 和高阶皮层表征则显示这些原则如何进一步连接到行为、时间和更高层认知。

一个尤其重要的整合点是：认知地图不仅服务于“我在哪里”，也服务于“我处在任务结构中的哪里”“我相对于目标在哪里”“我现在处于哪一段时间上下文”“当前具体对象如何绑定到抽象结构上”。因此，海马—内嗅—皮层系统可以被理解为一种通用的结构学习与快速绑定系统。

最后可以用四句话概括文章贡献：第一，它把不同模型放进一个 computational ontology 中，而不是把它们当作相互竞争的零散解释；第二，它把空间导航和非空间任务统一到状态空间、序列学习和泛化问题中；第三，它提出海马可能在不同阶段同时承担快速建图和记忆绑定功能；第四，它为理解语言、数学、逻辑、视觉概念等更高阶认知提供了可检验的计算思路。

---
