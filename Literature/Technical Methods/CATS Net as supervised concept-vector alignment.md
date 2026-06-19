# CATS Net as Supervised Concept-Vector Alignment

[Paper Link](https://doi.org/10.1038/s43588-026-00956-4)

## 目录

[1. 文章定位：不是纯粹的类人概念智能，而是监督概念向量对齐](#1-文章定位不是纯粹的类人概念智能而是监督概念向量对齐)

[2. 核心机制：把类别从输出头移到输入向量](#2-核心机制把类别从输出头移到输入向量)

　　[2.1 普通分类器与 CATS Net 的区别](#21-普通分类器与-cats-net-的区别)

　　[2.2 image-concept-label triplet](#22-image-concept-label-triplet)

　　[2.3 为什么没有 apple 分类头也能问 apple](#23-为什么没有-apple-分类头也能问-apple)

[3. CA/TS 架构：概念向量如何门控视觉判断](#3-cats-架构概念向量如何门控视觉判断)

　　[3.1 TS module：处理图片](#31-ts-module处理图片)

　　[3.2 CA module：把概念向量变成 gating signals](#32-ca-module把概念向量变成-gating-signals)

　　[3.3 Hadamard product 与 hierarchical gating](#33-hadamard-product-与-hierarchical-gating)

[4. 训练方式：网络参数和概念向量交替优化](#4-训练方式网络参数和概念向量交替优化)

　　[4.1 network-learning phase](#41-network-learning-phase)

　　[4.2 concept-learning phase](#42-concept-learning-phase)

　　[4.3 concept vector table 的真实含义](#43-concept-vector-table-的真实含义)

[5. 概念空间分析：哪些结果有信息，哪些偏装饰](#5-概念空间分析哪些结果有信息哪些偏装饰)

　　[5.1 basis vector probing](#51-basis-vector-probing)

　　[5.2 functional entropy](#52-functional-entropy)

　　[5.3 RDM/RSA 与人类语义模型对齐](#53-rdmrsa-与人类语义模型对齐)

[6. Concept communication：teacher-student 到底怎么迁移](#6-concept-communicationteacher-student-到底怎么迁移)

　　[6.1 leave-one-out setting](#61-leave-one-out-setting)

　　[6.2 concept vector expansion](#62-concept-vector-expansion)

　　[6.3 translation module = supervised space mapping](#63-translation-module--supervised-space-mapping)

　　[6.4 这个实验真正证明了什么](#64-这个实验真正证明了什么)

[7. Human concept spaces：Word2Vec/SPOSE 如何接入模型](#7-human-concept-spacesword2vecspose-如何接入模型)

　　[7.1 不是翻译人类概念，而是训练 CA/TS 适配固定语义空间](#71-不是翻译人类概念而是训练-cats-适配固定语义空间)

　　[7.2 和 CLIP 的相似与差别](#72-和-clip-的相似与差别)

[8. Brain alignment：RSA 证据的边界](#8-brain-alignmentrsa-证据的边界)

　　[8.1 concept layer vs VOTC](#81-concept-layer-vs-votc)

　　[8.2 CA module vs semantic-control network](#82-ca-module-vs-semantic-control-network)

　　[8.3 小相关、显著性与解释风险](#83-小相关显著性与解释风险)

[9. 我的判断：有用的技术点与不该被叙事带走的地方](#9-我的判断有用的技术点与不该被叙事带走的地方)

---

## 1. 文章定位：不是纯粹的类人概念智能，而是监督概念向量对齐

这篇文章提出 CATS Net，试图用一个双模块网络解释 concept formation、concept understanding 和 conceptual communication。作者的叙事很大：人类可以从感觉运动经验中形成概念，用概念重新激活感觉运动状态，还能通过符号和他人交流概念。CATS Net 被包装为一个可以连接视觉经验、低维概念空间和类脑语义控制机制的计算框架。

但从技术实现看，它更适合被理解为一种 **supervised concept-vector alignment** 方法。模型并不是在无监督环境中自己发现概念边界，而是依赖数据集已有类别标签，为每个类别维护一个低维 concept vector，再用监督式 Yes/No 匹配任务训练网络和概念向量。

更直白地说，文章的实际技术主线是：

1. 每个类别绑定一个低维 concept vector。
2. 模型输入是 image + concept vector，输出是 Yes/No。
3. concept vector 通过 CA module 生成 gating signal，门控 TS module 的视觉特征处理。
4. teacher-student 迁移通过 translation module 对齐两个网络的概念向量空间。
5. human concept compatibility 通过固定 Word2Vec/SPOSE 向量训练 CA/TS 去适配人类语义空间。
6. 大量 RSA/RDM、functional entropy、brain fitting 分析用于支持“概念空间像人类语义/脑表征”的叙事。

因此，这篇文章不太适合作为一个值得完整复现的 implementation 项目。它真正值得记录的是一种技术范式：**把类别知识显式压缩为低维条件向量，并用这个向量配置一个通用视觉判断器**。

---

## 2. 核心机制：把类别从输出头移到输入向量

### 2.1 普通分类器与 CATS Net 的区别

普通视觉分类器通常是：

\[
x \rightarrow f_\theta(x) \rightarrow [p_1,p_2,\ldots,p_K]
\]

其中 \(K\) 是类别数。对于 CIFAR-100，输出头有 100 个位置；对于 ImageNet-1k，输出头有 1000 个位置。类别知识部分体现在最后分类头和整个网络参数中。

CATS Net 不这样做。它的输出始终是二分类：

\[
H_\theta(x,c)\rightarrow [p_{\text{No}},p_{\text{Yes}}]
\]

其中 \(x\) 是图片，\(c\) 是概念向量。类别不在输出头中，而是在输入端的 concept vector 中。

所以问题从：

\[
\text{这张图属于哪一类？}
\]

变成：

\[
\text{这张图是否匹配这个概念向量？}
\]

这正是 CATS Net 能声称“没有 apple 分类头也能测试 apple”的原因。它不是多分类器，而是条件二分类器。

### 2.2 image-concept-label triplet

原始图像数据集是 image-label doublet：

\[
(x,y)
\]

CATS Net 把它改造成 image-concept-label triplet：

\[
(x,c,l)
\]

其中 \(c\) 是某个类别对应的概念向量，\(l\in\{\text{Yes},\text{No}\}\) 表示图片是否属于这个概念。

例如一张 apple 图片可以构造出：

\[
(x_{\text{apple}}, c_{\text{apple}}, \text{Yes})
\]

也可以构造出负样本：

\[
(x_{\text{apple}}, c_{\text{fish}}, \text{No})
\]

论文中不是枚举所有 image-category 组合，而是对一部分图片配对应概念向量作为 Yes 样本，对另一部分图片随机配非对应概念向量作为 No 样本。这样就把普通分类任务改造成了匹配判断任务。

### 2.3 为什么没有 apple 分类头也能问 apple

神经网络并不理解自然语言里的“看看这是不是苹果”。在 CATS Net 中，这句话落到数值形式就是：

\[
H_\theta(x,c_{\text{apple}})\rightarrow [p_{\text{No}},p_{\text{Yes}}]
\]

所谓 \(c_{\text{apple}}\) 一开始只是一个 20 维向量槽位。研究者用数据集标签固定这个槽位和 apple 类别的对应关系，然后通过监督信号训练这个向量，使它逐渐成为能让模型识别 apple 的控制输入。

因此，CATS Net 不是没有类别知识，而是把类别知识从固定输出 index 转移到了输入侧的 concept vector table 中。

---

## 3. CA/TS 架构：概念向量如何门控视觉判断

### 3.1 TS module：处理图片

TS module 是 sensorimotor task-solving module，负责处理图片特征并输出 Yes/No 判断。论文实现中，图像先经过预训练视觉 backbone，例如 ResNet50 或 ViT-B/16，然后输出视觉特征，再送入 TS module。

可以把 TS module 理解为：

\[
T(x)
\]

它本身处理的是图片侧信息。

### 3.2 CA module：把概念向量变成 gating signals

CA module 是 concept-abstraction module，输入低维概念向量：

\[
c\in\mathbb R^{20}
\]

它输出一组 gating signals，用于调控 TS module 的中间特征。CA module 不是直接输出类别，也不是直接产生 Yes/No，而是根据概念向量生成一组控制信号。

所以整体结构可以写成：

\[
H(x,c)=G(T(x),C(c))
\]

其中 \(C(c)\) 是 CA module 生成的控制信号，\(G\) 表示 CA 和 TS 之间的门控交互。

### 3.3 Hadamard product 与 hierarchical gating

门控操作的具体形式是 Hadamard product，也就是逐元素相乘。假设某一层 TS module 的视觉特征是：

\[
\mathbf x=[1.0,2.0,3.0,4.0]
\]

CA module 产生的 gating signal 是：

\[
\mathbf g=[-0.3,1.2,-0.1,-0.4]
\]

那么门控后的 TS representation 是：

\[
\mathbf z=\mathbf x\odot \mathbf g=[-0.3,2.4,-0.3,-1.6]
\]

这里 \(\odot\) 表示逐元素乘法。

论文称这种结构为 hierarchical gating，因为 CA module 在多个层级上调控 TS module。直观上，概念向量不是最后才拼接到分类头，而是在视觉处理过程中逐层改变哪些特征被放大、压低或传递。

这也是这篇文章相对于 CLIP 式相似度匹配的一个机制差别：CLIP 通常比较 image vector 和 text vector 的相似度，而 CATS Net 用 concept vector 去门控视觉处理过程。

---

## 4. 训练方式：网络参数和概念向量交替优化

### 4.1 network-learning phase

在 network-learning phase 中，concept vectors 固定，只更新 CA 和 TS module 的网络参数。

流程是：

\[
(x,c)\rightarrow H_\theta(x,c)\rightarrow \text{Yes/No}
\]

然后用交叉熵损失比较预测和真实标签。反向传播时只更新网络参数 \(\theta\)，不更新 concept vectors。

这一步让网络学会如何读取当前给定的 concept vector，并把它转换成有效的视觉判断配置。

### 4.2 concept-learning phase

在 concept-learning phase 中，网络参数固定，只更新 concept vectors。

同样输入：

\[
(x,c)\rightarrow H_\theta(x,c)
\]

但这次反向传播只改变 \(c\)，不改变 CA/TS 参数。这样每个类别的 concept vector 会移动到更适合当前网络读取的位置。

作者说这两个阶段按 epoch 交替进行。一个 epoch 通常指训练集完整遍历一遍，不是一次反向传播。一次反向传播通常对应一个 batch。

整体过程可以理解为：

1. 固定概念向量，训练网络读懂这些向量。
2. 固定网络，训练概念向量变得更好读。
3. 继续交替，使网络参数和概念向量互相适配。

### 4.3 concept vector table 的真实含义

对 ImageNet-1k，CATS Net 实际维护了：

\[
1000\times20
\]

的 concept vector table。对 CIFAR-100，则是：

\[
100\times20
\]

每一行对应一个类别。

这点需要说清楚：作者所谓的概念不是完全开放式、无标签地自发产生，而是在已有类别标签定义的槽位中学习出来的低维向量。标签名本身不进入网络，但标签决定哪个向量槽位对应哪个类别。

因此，CATS Net 的“概念形成”更准确地说是：

> 在监督类别框架下，学习一组可作为条件控制信号的低维类别向量。

---

## 5. 概念空间分析：哪些结果有信息，哪些偏装饰

### 5.1 basis vector probing

作者为了分析 20 维概念空间的功能特异性，输入 20 个 standard basis vectors：

\[
e_1=[1,0,\ldots,0]
\]

\[
e_2=[0,1,\ldots,0]
\]

一直到 \(e_{20}\)。

这些 one-hot 不是 ImageNet 的 1000 类 one-hot，而是 20 维概念空间中的标准基向量。它们的作用是 probe：测试单独打开某个概念坐标轴时，模型更容易对哪些类别输出 Yes。

训练前，basis vectors 的响应比较均匀。训练后，一些 basis vectors 会对特定 hyper-categories 更有选择性。这个结果说明概念空间的坐标方向被训练塑造成了某种功能偏好。

但这个分析不能过度解释。神经网络的语义方向不一定沿着单一坐标轴，也可能分布在多个维度的组合方向上。因此 basis vector probing 更像是一种可视化/探针分析，而不是严格证明某一维就等于某个语义特征。

### 5.2 functional entropy

作者进一步随机从训练后的概念空间采样 1000 个点，输入模型后统计它们对 ImageNet 1000 类的 Yes 响应分布。然后计算 functional entropy：

\[
e=-\sum_i p_i\log p_i
\]

其中：

\[
p_i=\frac{c_i}{\sum_j c_j}
\]

\(c_i\) 表示某个输入概念向量对第 \(i\) 类输出 Yes 的次数。

如果一个向量对很多类别都平均输出 Yes，那么熵高，说明没有特异性。如果一个向量主要对少数类别输出 Yes，那么熵低，说明它更像一个有选择性的概念输入。

作者发现训练后概念空间的 entropy 分布低于随机基线，于是认为整个空间变得更有类别特异性。

这个结果有一定信息量：它说明不只是训练过的类别向量有用，空间中一些随机位置也可能落入有功能意义的区域。但它仍然属于表征分析，不能单独支撑“类人概念形成”的强结论。

### 5.3 RDM/RSA 与人类语义模型对齐

作者用 RSA 比较 CATS concept space 和人类语义模型，例如 Binder65 与 SPOSE49。

RDM 是 representational dissimilarity matrix，即一组概念两两之间的距离表。RSA 的核心就是比较两张 RDM 是否相似。

如果 CATS 和 SPOSE 都认为：

\[
\text{apple 与 orange 近，apple 与 truck 远}
\]

那么它们的 RDM 就会相关。

这类分析的优点是，不要求两个空间维度相同。CATS 是 20 维，Binder65 是 65 维，SPOSE49 是 49 维，只要它们覆盖同一批概念，就可以比较概念之间的相对距离结构。

作者报告 CATS RDM 与 Binder65 和 SPOSE49 都有显著相关。这个结果支持 CATS 学出的概念空间具有某种人类语义结构相似性。不过相关系数本身不大，因此更稳妥的理解是：CATS 概念空间和人类语义空间在距离结构上有统计对应，而不是 CATS 已经学到了完整的人类概念系统。

---

## 6. Concept communication：teacher-student 到底怎么迁移

### 6.1 leave-one-out setting

teacher-student 实验使用 CIFAR-100。每一轮选择一个类别作为 held-out category，例如 apple。

设置是：

1. teacher 训练全部 100 类。
2. student 只训练 99 类，故意不训练 apple。
3. teacher 和 student 独立初始化、独立训练，因此它们的概念空间坐标系不同。

作者先检查 teacher 和 student 概念空间的结构是否相似，例如通过 RDM 相似性和聚类图。Fig. 4a/b 展示的是 teacher 与 student 的概念空间地图：每个点是一个类别概念向量，边和颜色来自层级聚类。它的含义是，不同网络虽然独立训练，但仍可能把水果、动物、车辆、家具等类别组织成相似模块。

需要注意，Fig. 4a/b 主要是可视化，不应被当作强证明。图注中也提到为了视觉对齐做过 manual adjustments。真正关键的是后续 translation 和 held-out transfer。

### 6.2 concept vector expansion

直接训练 teacher-to-student translation module 会遇到样本太少的问题。共同类别只有 99 个，也就是只有：

\[
c_T^k \rightarrow c_S^k,\quad k=1,\ldots,99
\]

这对一个 10 层、每层 500 neurons 的 MLP 来说非常少。

于是作者做 concept vector expansion。具体来说，teacher 网络训练好后固定网络参数，然后重新随机初始化某个类别的 concept vector，只优化这个 concept vector，使它也能完成该类别的 Yes/No 判断。重复多次后，同一个类别就会有多个 teacher-side concept vector variants。

例如：

\[
c_{T,\text{dog}}^{(1)},c_{T,\text{dog}}^{(2)},\ldots,c_{T,\text{dog}}^{(97)}
\]

这些都是 teacher 空间里能代表 dog 的向量变体。作者说每个类别扩展到 97 个 teacher concept vectors，也就是原始一个加上额外训练得到的 96 个变体。

这不是扩展类别，也不是扩展图片，而是扩展概念向量样本数。

### 6.3 translation module = supervised space mapping

translation module 是一个 MLP，用于学习：

\[
f(c_T)\approx c_S
\]

训练目标是把 teacher 概念空间映射到 student 概念空间。训练样本是共同 99 类：

\[
c_{T,k}^{(j)}\rightarrow c_{S,k}
\]

其中 \(k\) 是类别，\(j\) 是 teacher-side concept vector variant。

损失函数是 mean-squared error：

\[
L=\|f(c_T)-c_S\|^2
\]

测试时，student 没学过 apple，但 teacher 有 \(c_{T,\text{apple}}\)。于是：

\[
\hat c_{S,\text{apple}}=f(c_{T,\text{apple}})
\]

然后 student 用这个翻译后的概念向量进行判断：

\[
H_S(x,\hat c_{S,\text{apple}})\rightarrow \text{Yes/No}
\]

作者报告平均准确率约 0.729，显著高于 0.5 chance level。

### 6.4 这个实验真正证明了什么

作者把这个结果称为 conceptual communication 或 knowledge transfer。更保守地说，这个实验证明的是：

> 在共享类别监督下，可以训练一个 teacher concept space 到 student concept space 的映射；这个映射能对 held-out 类别产生一定泛化，使 student 在不更新高维网络参数的情况下使用新概念向量完成匹配判断。

它不是证明“概念天然可通信”，也不是证明“老师一句话学生就懂”。这个迁移依赖：

1. teacher 和 student 的类别标签体系共享。
2. 99 个共同类别用于监督训练 translation module。
3. teacher-side concept vector expansion 提供大量额外映射样本。
4. held-out 类别与共同类别处在同一个数据集语义空间中。

所以这部分技术上有价值，但作者的叙事明显比实际机制更大。

---

## 7. Human concept spaces：Word2Vec/SPOSE 如何接入模型

### 7.1 不是翻译人类概念，而是训练 CA/TS 适配固定语义空间

作者还测试 CATS Net 能否使用 human-generated concept spaces，例如 Word2Vec 和 SPOSE49。

这里没有单独训练 human-to-CATS translation module。做法是直接把人类语义向量作为 concept input，然后训练 CA/TS module 去使用这些固定向量。

以 Word2Vec 为例：

1. 类别名进入 Word2Vec，得到词向量。
2. Word2Vec 向量降到 20 维。
3. 这些 20 维向量固定不更新。
4. CATS Net 用 99 个类别训练 CA/TS 参数。
5. 留出类别的 Word2Vec 向量在测试时直接输入 CA module。

所以所谓 compatibility with human language-derived concept spaces，本质是：

> 在固定人类语义向量空间上训练 CATS 的条件判断器，并测试它能否泛化到留出类别的语义向量。

SPOSE49 也是类似，只不过向量来自人类相似性判断，而不是文本共现。

### 7.2 和 CLIP 的相似与差别

CATS Net 和 CLIP 都可以被看作 image-concept matching：

\[
\text{image}+\text{concept}\rightarrow \text{match score}
\]

但机制不同。

CLIP 通常是双塔结构：

\[
v=E_{\text{img}}(x),\quad t=E_{\text{text}}(s)
\]

\[
\text{logit}=v^\top t/\tau
\]

类别由文本 prompt 动态构造。

CATS Net 是：

\[
x\rightarrow T(x)
\]

\[
c\rightarrow C(c)\rightarrow \text{gating signals}
\]

然后通过门控调节视觉特征，最后输出 Yes/No。

因此，CATS Net 的特别之处不是“图像和概念匹配”这个任务形式，而是把概念向量设计成一种门控控制信号。它更像一个 concept-conditioned visual judge，而不是纯粹的 embedding similarity model。

但如果从批判角度看，CATS Net 的 concept vector table 也确实很像一种 supervised class embedding table。它没有摆脱类别监督，只是把类别表示从输出头搬到了输入侧。

---

## 8. Brain alignment：RSA 证据的边界

### 8.1 concept layer vs VOTC

作者用 fMRI 数据集比较 CATS concept layer 和人类 VOTC 的表征结构。被试观看 95 个物体图片，并执行口头命名任务。作者分别计算：

1. CATS concept layer 对 95 个物体的 RDM。
2. VOTC 脑活动对 95 个物体的 RDM。

然后用 RSA 比较二者。

他们还控制了低级视觉特征，即使用 ResNet sensory input layer 的 RDM 作为控制变量，计算 partial Spearman correlation：

\[
\text{corr}(\text{CATS RDM},\text{VOTC RDM}\mid \text{ResNet sensory RDM})
\]

这样做的意图是排除一种简单解释：CATS 和 VOTC 相似只是因为它们都编码颜色、纹理、边缘、形状等低级视觉特征。

结果是 CATS concept layer 与 VOTC 有小但显著的相关。更稳妥的解释是：在扣除某些低级视觉特征后，CATS concept layer 与 VOTC 仍有统计上的表征结构对应。

### 8.2 CA module vs semantic-control network

作者进一步比较 CA module 和 semantic-control brain network。逻辑是：CA module 不直接表示视觉内容，而是根据概念向量门控视觉处理；semantic-control network 也被认为负责根据任务和语境选择、访问、操作语义信息。

做法仍然是 RDM/RSA：

1. 对 95 个概念输入，提取 CA1 layer 的激活。
2. 计算 CA1 RDM。
3. 计算 semantic-control network 的脑活动 RDM。
4. 比较两张 RDM。

作者发现 CA1 与 semantic-control network 显著相关，也和 multiple-demand network 有相关，但前者更强。于是他们解释为 CA module 更特异地对应语义控制过程。

这里的证据属于表征相似性，不是机制同一性。它不能说明 CA module 就是大脑语义控制网络的真实实现，只能说明两者对这批刺激的相对差异结构有相似性。

### 8.3 小相关、显著性与解释风险

这一部分是全文章生物/认知神经科学叙事最浓的地方。很多结果都是小相关加显著性检验。由于样本结构、模型实例和 RDM 元素数量的关系，显著性不等于强解释力。

尤其需要区分：

1. 统计显著：相关不太像随机噪声。
2. 效应强：相关程度足以支持强机制解释。
3. 机制同一：模型模块真的实现了人脑对应机制。

这篇文章较好地支持了第 1 点，部分支持第 2 点，但远远没有证明第 3 点。

因此，brain alignment 部分更适合作为“模型表征与脑表征有一定结构相似性”的证据，而不是“CATS Net 已经解释人脑概念处理机制”的证据。

---

## 9. 我的判断：有用的技术点与不该被叙事带走的地方

这篇文章最有价值的地方，不是它证明了 human-like conceptual intelligence，而是它提供了一个清晰的技术构型：

\[
\text{image}+\text{low-dimensional concept vector}\rightarrow \text{Yes/No}
\]

并且让 concept vector 通过 CA module 以 gating 的形式调节视觉特征处理。

它值得记录的技术点包括：

1. 把类别从输出头移到输入侧 concept vector。
2. 用 image-concept-label triplet 训练通用匹配判断器。
3. 交替优化网络参数和 concept vectors。
4. 通过 CA/TS hierarchical gating 实现 concept-conditioned computation。
5. 用 translation module 做 teacher-student concept space alignment。
6. 用固定 Word2Vec/SPOSE 向量测试架构能否适配外部语义空间。

但也必须保留批判判断：

1. concept vectors 本质上是 supervised class embedding table。
2. 类别标签仍然定义了概念槽位，并没有真正无监督地产生开放式概念。
3. teacher-student communication 依赖共同类别监督和显式 translation module。
4. concept vector expansion 大幅增加了翻译器训练样本，使通信实验难度降低。
5. 大量 functional entropy、RDM/RSA、brain fitting 分析更像表征叙事链，而不是强机制证明。
6. 小相关的显著性结果需要谨慎解释，不能直接上升为“类脑机制”。

因此，这篇文章可以被压缩成一句话：

> CATS Net 不是一个真正证明类人概念智能的系统，而是一个受监督的概念向量对齐框架：它把类别表示成低维可学习向量，用这些向量门控视觉判断网络，并通过显式空间映射实现有限的跨模型概念迁移。

这个技术思想可以作为 concept-conditioned recognition、semantic alignment、以及可迁移类别原型设计的参考；但不适合作为强理论证据来证明人工网络已经形成了人类式概念。

