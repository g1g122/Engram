# Semantic Priors for Domain Robustness: CLIP-Based Generalization and Adaptation

[Paper Link](https://doi.org/10.1109/TPAMI.2026.3651700)

## 目录

[1. 文章定位与核心主线](#1-文章定位与核心主线)

[2. 基础概念与任务设定](#2-基础概念与任务设定)

　　[2.1 Domain 到底是什么](#21-domain-到底是什么)

　　[2.2 DG 与 DA 的根本区别](#22-dg-与-da-的根本区别)

　　[2.3 SA、SDF、SFF](#23-sasdfsff)

　　[2.4 SS/MS 与 CS/PS/OS/OPS](#24-ssms-与-cspsosops)

[3. CLIP 基础机制](#3-clip-基础机制)

　　[3.1 图文对比学习](#31-图文对比学习)

　　[3.2 CLIP 的 zero-shot 分类方式](#32-clip-的-zero-shot-分类方式)

　　[3.3 CLIP 为什么适合 DG/DA](#33-clip-为什么适合-dgda)

[4. CLIP-based DG 方法](#4-clip-based-dg-方法)

　　[4.1 Prompt Optimization](#41-prompt-optimization)

　　[4.2 CLIP as Backbone or Encoder](#42-clip-as-backbone-or-encoder)

　　[4.3 Source-Available DG](#43-source-available-dg)

　　[4.4 Source-Free DG](#44-source-free-dg)

　　[4.5 DG Discussion 与批判性理解](#45-dg-discussion-与-批判性理解)

[5. CLIP-based DA 方法](#5-clip-based-da-方法)

　　[5.1 Source-Available DA](#51-source-available-da)

　　[5.2 Source-Free DA](#52-source-free-da)

　　[5.3 DA Discussion 与批判性理解](#53-da-discussion-与-批判性理解)

[6. 常用数据集与评价指标](#6-常用数据集与评价指标)

　　[6.1 Multi-domain datasets](#61-multi-domain-datasets)

　　[6.2 Single-domain datasets](#62-single-domain-datasets)

　　[6.3 Accuracy 与 HOS](#63-accuracy-与-hos)

[7. Challenges and Future Directions](#7-challenges-and-future-directions)

[8. 全文总结与个人评价](#8-全文总结与个人评价)

---

## 1. 文章定位与核心主线

这篇文章是一篇关于 **CLIP-powered Domain Generalization（DG）与 Domain Adaptation（DA）** 的综述。它的价值主要不在于某一个模型解释得特别深，而在于把 CLIP 时代的跨域泛化与跨域自适应方法放进了一个相对统一的 taxonomy 中。文章先定义 DG/DA、SA/SF、SS/MS、CS/PS/OS/OPS 等任务设定，再介绍 CLIP 的基础图文对齐机制，然后分别综述 CLIP-based DG 和 CLIP-based DA 方法，最后整理常用数据集、评价指标和未来方向。

整篇文章可以用一句话概括：

> CLIP-based DG/DA 的核心变化，是从传统“纯视觉特征对齐”转向“利用 CLIP 的图文语义先验、prompt、adapter、蒸馏、伪标签与多模态知识来处理域偏移”。

传统 DG/DA 更多围绕图像特征展开，例如学习 domain-invariant representation、做 adversarial alignment、数据增强、元学习等。CLIP 出现后，类别文本、prompt、图文相似度、zero-shot 分类能力变成新的核心资源。也就是说，模型不再只是学“这个图像属于第几个 index”，而是把图像和自然语言类别语义对齐。

但是这篇文章也有明显问题：方法介绍较罗列，很多模型只有一句机制概括；最后的 future directions 比较模板化，像是常见 AI survey 的收尾。真正有价值的是它的分类体系、数据集整理、评价指标整理，以及它间接暴露出的一个事实：很多 CLIP-based DG/DA 的性能提升可能主要来自 CLIP 预训练语义先验，而不一定说明真实复杂场景中的域泛化问题已经被解决。

---

## 2. 基础概念与任务设定

### 2.1 Domain 到底是什么

在 DG/DA 语境下，**domain** 不是任意数据集合，而是指同一任务语义框架下，由不同数据生成机制产生的数据分布。比如“照片狗 → 素描狗 → 油画狗 → 像素狗”是典型 domain shift，因为任务仍然是识别狗、猫、车等类别，只是图像风格、纹理、边缘、背景、成像方式发生变化。

更正式地说，一个 domain 可以由图像分布、采集设备、场景环境、风格、噪声、背景、地理位置、医院设备等因素决定。比如医学中“医院 A 的肺部 CT → 医院 B 的肺部 CT”可以是 domain shift，因为任务仍然是肺部疾病判断，只是设备、参数、人群、标注习惯不同。

但“照片狗 → 医学肿瘤良恶性判断”通常不是普通 domain shift，而是跨任务迁移。因为类别语义、视觉模式、任务目标都变了。人类也需要先知道肿瘤长什么样，不能凭借“看过狗”直接识别肿瘤。因此 DG/DA 通常隐含一个前提：源域和目标域在任务语义上有关系，闭集设定中类别一致，开放设定中至少有共享类或可定义 unknown。

### 2.2 DG 与 DA 的根本区别

**Domain Generalization（DG）** 的核心是：训练时只能访问一个或多个有标签源域，不能访问目标域样本，训练完成后直接在未见目标域上测试。

\[
\mathcal{D}_s=\{D_1,D_2,\ldots,D_S\}
\]

表示源域集合。DG 的目标是学习一个模型 \(f\)，使它在训练时未见过的目标域 \(\mathcal{D}_t\) 上也能表现稳定。比如用 Photo、Cartoon、Art 训练，直接测试 Sketch。

**Domain Adaptation（DA）** 的核心是：训练或适应时可以访问目标域样本，但目标域通常没有标签。它不是泛化到任意未知域，而是适应某一个已经给定的目标域。

\[
D_t=\{x_j^t\}_{j=1}^{N_t}
\]

表示目标域无标签数据。比如用 Photo 有标签数据和 Sketch 无标签数据共同训练，使模型适应 Sketch。

DG 与 DA 的核心差别不是“哪个更高级”，而是信息条件不同。DG 训练时完全看不到目标域，因此追求的是对未知域的鲁棒性；DA 能看到目标域无标签样本，因此追求的是对特定目标域的适应。DA 看起来信息更多，但也会带来伪标签错误、目标域过拟合、灾难性遗忘等问题。

| 项目 | DG | DA |
|---|---|---|
| 目标域训练时是否可见 | 不可见 | 可见，通常无标签 |
| 目标 | 未知域直接泛化 | 适应特定目标域 |
| 典型流程 | 源域训练 → 目标域直接测试 | 源域训练 + 目标域无标签适应 → 目标域测试 |
| 主要风险 | 学到源域特有偏差 | 伪标签错误、适应错、遗忘旧知识 |

### 2.3 SA、SDF、SFF

**Source-Available（SA）** 表示源域有标签数据在训练/适应时仍然可用。标准 UDA 通常就是 SA：源域有标签，目标域无标签，二者共同参与适应。

**Source-Data-Free（SDF）** 表示源域原始数据不可用，但有一个在源域上训练好的模型。比如医院 A 的数据不能给你，但给你一个医院 A 训练好的模型，你要用它和医院 B 的无标签数据做适应。

**Source-Fully-Free（SFF）** 更严格，表示源域数据、源域模型、源域统计信息都没有，只能依赖 CLIP 这类通用预训练基础模型、目标域无标签样本、类别名或 prompt。SFF 在 DA 中通常仍然可以访问目标域无标签数据；在 DG 中则更极端，通常没有源域真实数据，也没有目标域训练数据，只靠 CLIP、类别名和风格/语义 prompt 机制。

这里必须强调：SFF 并不是真的“没有源”。它只是当前下游任务的源域数据不可用，但它强烈依赖 CLIP 的预训练数据和预训练语义。换句话说，source-free 不是 data-free，而是 downstream-source-free。CLIP 的预训练数据承担了很大一部分隐性先验来源。

### 2.4 SS/MS 与 CS/PS/OS/OPS

**SS/MS** 是源域数量维度。SS 表示 Single-Source，只有一个源域；MS 表示 Multi-Source，有多个源域。DG 和 DA 都可以有 SS/MS。

**CS/PS/OS/OPS** 是标签空间关系维度，描述源域标签集合 \(Y_s\) 和目标域标签集合 \(Y_t\) 的关系。

| 场景 | 数学关系 | 含义 |
|---|---|---|
| CS | \(Y_s=Y_t\) | 源域和目标域类别完全一样 |
| PS | \(Y_t\subset Y_s\) | 目标域类别是源域类别子集 |
| OS | \(Y_s\subset Y_t\) | 目标域包含源域类别，并有新类别 |
| OPS | \(Y_s\cap Y_t\neq\varnothing,\ Y_s\nsubseteq Y_t,\ Y_t\nsubseteq Y_s\) | 两边有共享类，也各自有私有类 |

CS 最简单，只需要处理分布差异。PS 的难点是目标域没有源域私有类，模型不能把目标样本误分到这些源私有类。OS 的难点是目标域出现源域没见过的新类，传统分类器通常不能输出具体新类别，只能把它们识别为 unknown。OPS 最复杂，因为它同时有源域私有类和目标域私有类。模型既要对齐共享类，又要抑制源域私有类，还要识别目标未知类。

关于 OPS 需要特别注意：传统固定分类器的输出 index 是固定的，比如 dog/cat/car 三类。如果目标域来了 horse，普通分类器不会凭空多出 horse 输出。因此传统 OPS/OS 通常是把所有源域没见过的目标类合并成 unknown。CLIP 的优势在于如果给定候选类别名，可以通过文本 prompt 动态构造类别原型，从而扩展输出空间。但如果未知类名称也不知道，即使用 CLIP 通常也只能做 unknown detection。

---

## 3. CLIP 基础机制

### 3.1 图文对比学习

CLIP 的核心是把图像和文本编码到同一个向量空间。给定一批图文对：

\[
I=\{I_1,I_2,\ldots,I_N\}
\]

\[
T=\{T_1,T_2,\ldots,T_N\}
\]

图像 encoder 得到图像向量：

\[
v_i=E_{\text{img}}(I_i)
\]

文本 encoder 得到文本向量：

\[
t_i=E_{\text{text}}(T_i)
\]

CLIP 希望正确图文对 \((v_i,t_i)\) 靠近，错误图文对 \((v_i,t_j), i\neq j\) 远离。它使用对称对比损失：

\[
L_{\text{image}\rightarrow \text{text}}
=
-\frac{1}{N}
\sum_{i=1}^{N}
\log
\frac{\exp(v_i^\top t_i/\tau)}
{\sum_{j=1}^{N}\exp(v_i^\top t_j/\tau)}
\]

这个 loss 的含义是：给定一张图像 \(v_i\)，从一批文本中找出正确文本 \(t_i\)。

\[
L_{\text{text}\rightarrow \text{image}}
=
-\frac{1}{N}
\sum_{i=1}^{N}
\log
\frac{\exp(t_i^\top v_i/\tau)}
{\sum_{j=1}^{N}\exp(t_i^\top v_j/\tau)}
\]

这个 loss 的含义是：给定一个文本 \(t_i\)，从一批图像中找出正确图像 \(v_i\)。

最终损失是：

\[
L_c=
\frac{1}{2}
\left(
L_{\text{image}\rightarrow \text{text}}
+
L_{\text{text}\rightarrow \text{image}}
\right)
\]

其中 \(\tau\) 是 temperature 参数，用于控制 softmax 分布的尖锐程度。这个对称损失使 CLIP 同时具备 image-to-text retrieval 和 text-to-image retrieval 的能力，也形成了后续 zero-shot 分类和 prompt tuning 的基础。

### 3.2 CLIP 的 zero-shot 分类方式

传统分类器通常是：

\[
\text{image feature}\rightarrow \text{linear classifier}\rightarrow \text{class index}
\]

CLIP 分类方式不同。它先为每个类别构造文本 prompt，例如：

\[
\text{"a photo of a dog"}
\]

\[
\text{"a photo of a cat"}
\]

文本 encoder 将这些 prompt 编码成类别文本原型 \(t_c\)。图像 encoder 将输入图像编码为 \(v\)。分类 logit 是：

\[
\text{logit}_c=\frac{v^\top t_c}{\tau}
\]

最终预测来自图像特征和各类别文本特征的相似度。也就是说，CLIP 的类别不是固定分类头中的 index，而是由类别名称和 prompt 构造出的文本原型。这解释了为什么 prompt、类别语义、文本描述、LLM 生成属性会在 CLIP-based DG/DA 中变得重要。

### 3.3 CLIP 为什么适合 DG/DA

DG/DA 的核心困难是源域和目标域分布不同。传统视觉模型容易学习源域特有纹理、背景、颜色或风格。CLIP 的优势在于它把图像表示锚定到语言语义空间，比如 dog 的文本语义在照片、素描、卡通、油画中相对稳定。因此 CLIP-based 方法可以用文本类别原型、prompt、多模态对齐和 zero-shot 能力减少对源域视觉风格的依赖。

但这一点不能神化。CLIP 的泛化能力来自大规模图文预训练，因此 benchmark 结果必须和预训练分布、数据集难度、强 baseline 一起看。PACS、VLCS、DomainNet 等数据集之间的差异会在第 8 节集中讨论。

---

## 4. CLIP-based DG 方法

DG 部分主要分两条线：一条是 **Prompt Optimization**，研究如何优化 CLIP 的 prompt；另一条是 **CLIP as Backbone or Encoder**，研究如何把 CLIP 作为特征提取器、可训练 backbone、adapter 框架或蒸馏教师用于 DG。

### 4.1 Prompt Optimization

Prompt Optimization 的核心问题是：手写 prompt 如 “a photo of a [CLASS]” 不一定适合下游任务，也可能偏向 photo 域，因此能否把 prompt 变成可学习、条件化、多模态或语义正则化的表示。

| 方法 | 目的 | 原理 |
|---|---|---|
| CoOp | 用可学习 prompt 替代手写模板 | 将上下文词换成连续可训练向量 \([V_1]\ldots[V_M][CLASS]\)，冻结 CLIP image/text encoder，只训练 prompt tokens。它可以使用所有类别共享的 unified context，也可以为每类学习 class-specific context。 |
| CoCoOp | 提升 CoOp 对未见类/未见域的泛化 | CoOp 的 prompt 是静态的；CoCoOp 用轻量网络根据输入图像生成 input-conditional tokens，使每张图的 prompt 可以根据图像特征动态调整。 |
| KgCoOp | 防止 learnable prompt 偏离 CLIP 原始语义 | 学习 prompt 时加入知识约束和对比约束，使 learnable prompt 的文本表示不要离 hand-crafted template 太远，从而减少语义漂移。 |
| ProGrad | 避免 prompt tuning 过拟合源域 | 不是所有梯度都更新 prompt，而是只保留与 VLM general knowledge 方向一致的梯度，减少破坏 CLIP 原有语义的更新。 |
| MaPLe | 单侧 text prompt 不足以适配 CLIP 双模态结构 | 同时引入 text prompt 与 visual prompt，并通过 conditioning function 耦合图像侧和文本侧 prompt，让两种模态联合优化。 |
| LAMM | 类别 token 本身也可能需要优化 | 不只训练上下文 token，还优化 〈CLASS〉 类别 embedding，并用 hierarchical loss 保持多个空间中的语义一致性。 |

prompt learning 不是改英文句子，而是在 text encoder 输入端插入可学习 embedding。以 CoOp 为例，原始 prompt “a photo of a dog” 会被 tokenizer 转成 token embedding，而 CoOp 用可训练向量替代 “a photo of a” 这部分上下文：

\[
[V_1,V_2,\ldots,V_M,\text{dog}]
\]

图像特征 \(v\) 与每个类别文本特征 \(t_c\) 计算相似度，再用源域标签做交叉熵。反向传播时 CLIP 主体通常冻结，只更新 \(V_1,\ldots,V_M\)。

MaPLe 中可以把视觉 prompt 和文本 prompt 的关系抽象为：

\[
P^v=h(P^t)
\]

这里 \(P^t\) 是文本侧 prompt，\(P^v\) 是视觉侧 prompt，\(h\) 是可训练的映射或 conditioning function。它表达的不是严格固定公式，而是图文 prompt 不是各学各的，而是通过某种函数产生联系。通常 \(h\) 和 \(P^t\) 会训练，\(P^v\) 要么直接训练，要么由 \(P^t\) 和 \(h\) 间接生成。dog/cat 这些类别名 token 通常保持为 CLIP 原始词表 embedding，不随便训练；LAMM 这类方法才会尝试训练 category token，并额外约束语义不漂移。

### 4.2 CLIP as Backbone or Encoder

这一部分讨论 CLIP 在 DG 中的两种使用方式。第一种是把 CLIP image/text encoder 当作可训练 backbone，结合任务特定结构一起训练；第二种是冻结 CLIP encoder，把它作为静态特征提取器。前者适应能力强，但可能过拟合源域并破坏 CLIP 原有知识；后者更稳定、更省计算，但下游适配能力有限。

这一部分的方法按照 source availability 分为 Source-Available 和 Source-Free。Source-Available 进一步按 single-source / multi-source、closed-set / open-set 组织。

### 4.3 Source-Available DG

#### 4.3.1 SS-CSDG：单源闭集 DG

SS-CSDG 中只有一个有标签源域，目标域不可见且类别相同。它的难点在于源域信息太少，模型容易学到源域特有风格。

| 方法类别 | 方法 | 原理 |
|---|---|---|
| Prompt-driven | DAPT | 同时优化文本 prompt 和视觉 prompt，使二者匹配源域数据分布。它使用 inter-dispersion 与 intra-dispersion loss：类间文本/特征要分散，同类图像特征要紧凑，从而提升泛化。 |
| Prompt-driven | PromptSRC | 自正则化 prompt learning。它用 frozen CLIP 作为语义锚点，使 learnable prompt 与原模型保持一致，并约束 prompt 内部一致性，减少源域过拟合。 |
| Prompt-driven | SPG | 从生成视角理解 prompt learning。训练一个 prompt generator，测试时根据输入图像生成 instance-specific prompt。它不直接输出类别，而是为 CLIP 文本原型生成当前图像条件化上下文。 |
| Prompt-driven | LDFS | 用文本引导生成多样化 domain features，通过 text-guided instance-conditional augmentation 模拟潜在域变化，并用 pairwise regularizer 保持特征语义一致。 |
| Prompt-driven | GalLoP | 结合 global 与 local visual features 生成 prompts。局部 prompt 与稀疏图像区域对齐，使文本-图像匹配更细粒度，并通过 prompt dropout 和多尺度设计增加多样性。 |
| Prompt-driven | FrogDogNet | 面向遥感等场景，引入 Fourier filter block 保留低频域不变成分，抑制背景噪声，同时用 remote sensing prompt alignment loss 增强跨域稳定性。 |
| Architecture-enhanced | BorLan | 用预训练语言模型对齐视觉与语言特征，使视觉模型学习 semantic-aware、domain-agnostic 表示。 |
| Architecture-enhanced | MMA | 引入 multimodal adapter，在图像分支和文本分支之间聚合特征并允许梯度交流，高层使用 adapter 以平衡判别性和泛化性。 |
| Architecture-enhanced | StyLIP | 冻结 CLIP vision encoder，用轻量 projector 提取多尺度 style features，将 style/content 信息显式建模以增强泛化。 |

SPG 这类 instance-specific prompt 方法容易让人困惑：测试时的确会把测试图像先输入 image encoder，再由训练好的 generator 生成 prompt，然后和类别名一起送入 text encoder 做分类。只要测试时不反向传播更新参数，这仍然是 DG inference，不是 DA。但如果 generator 太强，它可能退化成分类器；因此这类方法通常限制 generator 只生成 prompt 条件，而不直接输出类别概率。

#### 4.3.2 MS-CSDG：多源闭集 DG

MS-CSDG 中有多个源域，目标域不可见且类别相同。它比单源 DG 更有机会学习域不变规律，因为多个源域之间的差异可以帮助模型识别哪些特征稳定、哪些只是域特有偏差。

| 方法类别 | 方法 | 原理 |
|---|---|---|
| Distillation | RISE | 用 VLM teacher 的文本语义特征指导学生图像特征。absolute distance 让学生特征靠近正确类别文本原型，relative distance 保持类别间语义关系。 |
| Distillation | VL2V-SD | Vision-Language to Vision 的 self-distillation，用文本语义指导视觉模型在不同增强或视角下保持一致，提升 OOD 泛化。 |
| Distillation | VL2V-ADiP | 黑盒特征融合，不一定访问 VLM 内部，只利用 VLM 输出特征与视觉模型特征融合，借助 VLM 语义先验提升泛化。 |
| Distillation | CAL | 训练时用 CAFT 增强鲁棒性，推理时用 ETTA 进行轻量校准，结合训练期稳健学习与测试期适应。 |
| Prompt-driven | DPL | 用三层 MLP 生成 prompt，在参数量紧凑的前提下提升多源 DG 性能。 |
| Prompt-driven | Any-Shift Prompting | 通过 hierarchical prompts 与 pseudo-shift mechanism 模拟潜在分布偏移，提升对未知 shift 的适应性。 |
| Prompt-driven | DPR | 解耦 text prompt 与 visual prompt，用 domain-specific prototypes 和 invariant prediction fusion 保留域不变预测。 |
| Prompt-driven | Wen et al. | 使用多样文本 prompt 模拟多个 domain contexts，通过 suppression、consistency、diversification 减弱 CLIP 对源域特有特征的偏置。 |
| Prompt-driven | PADG | DPR 的扩展版，用 controllable language prompts 解耦 invariant features，并用 worst explicit representation alignment 强化图文一致性。 |
| Architecture | HAM | 先为每个源域训练单独 CLIP encoder，再通过 conflict-aware model merging 合并，避免不同源域参数更新冲突。 |
| Architecture | ClipMix | 用 CLIP 作为外部知识指导图像混合和 label shift，在像素级与特征级增强中保持语义完整。 |
| Architecture | Mixup-CLIPood | 结合 mix-up loss 与大 VLM backbone 做多模态 DG，增强对象识别的 OOD 泛化。 |
| Architecture | MoA | 在 CLIP/ViT 中插入多个不同容量 adapter，并用 learnable router 为每个 token 选择合适 adapter，实现参数高效微调。 |
| Architecture | StyLIP | 多源场景中利用多源风格差异更好地分离 style 与 content，并生成 domain-aware prompt。 |

**MoA** 是这一类中很有代表性的结构方法。它不是改 prompt，也不是全量 fine-tune CLIP，而是在 ViT/CLIP 的 transformer block 中插入多个 adapter experts。一个 ViT 图像会被切成 patch tokens：

\[
x_1,x_2,\ldots,x_n
\]

每个 token 都可以由 router 计算 expert 分数：

\[
r_i=[r_{i1},r_{i2},\ldots,r_{iK}]
\]

然后选择 top-k adapters，并加权混合：

\[
A_{\text{mix}}(x_i)=\sum_{k\in \text{Top-}k} r_{ik}A_k(x_i)
\]

最后加回原始 CLIP 层输出：

\[
y_i=W_0(x_i)+A_{\text{mix}}(x_i)
\]

其中 \(W_0\) 是冻结的预训练权重，\(A_k\) 是不同容量的 adapters。小 adapter 改动小、正则化强，大 adapter 改动大、适配能力强。router 学会根据 token 类型、位置、语义或背景选择不同适配强度。MoA 的核心价值是平衡“适配源域任务”和“保留 CLIP 原始泛化能力”。

#### 4.3.3 MS-OSDG：多源开集 DG

MS-OSDG 中目标域类别包含源域没见过的新类别：

\[
Y_s\subset Y_t
\]

目标域不可见且可能有 unknown 类，因此模型既要识别已知类，也要把未知类检测为 unknown。文章指出该方向探索较少，现有方法主要利用扩散模型、prompt learning、蒸馏与 CLIP 的 open-vocabulary 能力。

| 方法 | 原理 |
|---|---|
| CLIPood | 面向 OOD 的 CLIP fine-tuning。它使用 margin metric softmax 与 class-adaptive margins，使类别边界受 CLIP 文本语义关系约束；同时用 beta-weighted ensemble 融合 zero-shot CLIP 和 fine-tuned CLIP，避免微调损害 OOD 能力。 |
| ODG-CLIP | 用 Stable Diffusion 生成 unknown proxy images，为未知类提供训练代理样本，并用 style-aware prompt 增强语义对齐与域区分。 |
| SCI-PD | 将 VLM 的鲁棒性蒸馏给轻量视觉模型。它在扰动下从 score、class、instance 三个层面蒸馏，使学生模仿 VLM 的稳定输出和关系结构。 |
| OSLoPrompt | 结合 GPT-4o 生成语义属性与 Stable Diffusion 生成 pseudo-open samples，用 domain-agnostic prompt learning 区分 known 和 unknown。 |
| MetaPrompt | 用 meta-learning、semantic attention、unsupervised contrastive loss 训练 prompt，使模型在多源、多域、开放类场景下更好地区分 known/unknown。 |

CLIPood 的关键是处理 fine-tuning 与 OOD 泛化之间的冲突。zero-shot CLIP 泛化强但下游精度可能不足；fine-tuned CLIP 源域更准但容易损害开放类和未知域能力。CLIPood 通过语义 margin 约束类别边界，并通过时间/权重集成保留 zero-shot 能力，从而避免模型完全变成 source-specific classifier。

### 4.4 Source-Free DG

Source-Free DG 尤其是 SFF-DG 没有真实源域数据，也没有目标域训练数据。模型只能依赖 CLIP、类别名、prompt、domain bank、风格词或 LLM 生成语义。这类方法的核心是：在 CLIP 的文本空间或图文联合空间中模拟潜在域变化。

| 方法 | 原理 |
|---|---|
| DUPRG | 使用 domain bank 聚合潜在域 embeddings，通过 domain-unified prompt generator 学习 domain-invariant text representation。 |
| PromptStyler | 在 CLIP 图文联合空间中合成多种 style prompts，模拟 sketch、painting、cartoon 等潜在目标域风格。 |
| DPStyler | 设计 style generation 和 style removal 两个模块，前者不断生成多样风格，后者减少风格变化对 encoder 输出特征的影响。 |
| PromptTA | 引入 text adapter 和 style feature resampling，用重采样覆盖风格特征分布，使文本表示对未知域更稳。 |
| BatStyler | 基于 PromptStyler，引入 GPT-4 生成粗语义，并利用 neural collapse 思想生成均匀分布的 style templates。 |

SFF-DG 的 prompt 不是全靠手写。类别名和基础模板可能是手写起点，但风格 prompt、domain prompt、style embedding、prompt generator、text adapter、uniform style templates 等往往是自动生成、采样或优化出来的。它们的训练信号不来自真实源域图像，而来自 CLIP 语义一致性、类别名称、风格多样性约束、LLM 语义扩展和 prompt 空间正则。

但这一类方法必须被批判性理解：SFF-DG 强烈依赖 CLIP 预训练。如果 CLIP 已经见过大量类似风格和类别，方法会表现很好；如果目标域非常复杂、长尾、细粒度或远离互联网图文分布，source-free prompt 方法可能会明显下降。

### 4.5 DG Discussion 与批判性理解

文章对 DG 的 discussion 大致有四个判断。第一，prompt 从手写模板发展到可学习、多层、多模态、风格化和生成式机制。第二，方法越来越利用 CLIP 的图文语义做 semantic-level generalization，而不是传统显式 domain alignment。第三，在固定 ViT-B/16 backbone 时，性能差异更多来自 multimodal semantics 和 prompt design，而不是结构复杂度。第四，source-free 方法实用但通常在困难 benchmark 上落后于 source-available 方法。

这里有一个需要保留的补充判断：PACS、VLCS 上的高分不应直接等同于真实未知域泛化能力。它们类别少、域风格常见，CLIP 预训练分布可能已经覆盖了大量相似图文模式。这个问题会在第 8 节集中讨论。

---

## 5. CLIP-based DA 方法

DA 与 DG 的不同在于目标域样本可见。CLIP-based DA 的核心是：如何利用源域知识、目标域无标签数据和 CLIP 的图文语义完成适应。文章按 Source-Available 和 Source-Free 分类。

### 5.1 Source-Available DA

#### 5.1.1 SS-CSUDA：单源闭集 UDA

这是最标准的 DA 场景：一个有标签源域，一个无标签目标域，类别相同。

| 方法类别 | 方法 | 原理 |
|---|---|---|
| Domain-aware prompt | PADCLIP | 将 domain name 纳入 prompt，使 CLIP 感知源/目标域差异，并用 adaptive debiasing 防止目标域适应时遗忘 CLIP 原始知识。 |
| Domain-aware prompt | AD-CLIP | 学习 domain token、image token、class token，使 prompt 同时编码域、图像和类别信息。 |
| Domain-aware prompt | DAPrompt | 为不同域学习 domain-specific prompts，并动态调整分类器以适应目标域。 |
| Dual-branch prompt | PDA | 双分支 prompt tuning，一个分支保留基础识别能力，另一个分支负责跨域对齐。 |
| Prompt + feature | PTT-VFR | 结合 prompt task-dependent tuning、visual feature refinement、domain-aware pseudo-labeling 和 zero-shot prediction。 |
| LLM-enhanced | MAwLLM | 用 LLM 生成细粒度类别知识或属性描述，增强 prompt 语义和图文对齐。 |
| Dual-branch | PDbDa | foundation branch 保留 CLIP 基础知识，adaptation branch 适应目标域，并用 prompt knowledge constraint 防止跑偏。 |
| Adversarial prompt | ADAPT | 同时使用 text prompt 与 visual prompt，通过对抗式交互学习源域与目标域共享表示。 |

#### 5.1.2 Cross-modal Alignment

这一类方法强调图像特征和文本语义之间的桥接。目标域图像由于分布偏移，可能与正确文本类别原型错位，因此需要更细致的跨模态对齐。

| 方法 | 原理 |
|---|---|
| DAMP | Domain-Agnostic Mutual Prompting。用图像上下文反向提示语言分支，让文本 prompt 根据目标图像实例调整。 |
| UniMoS | 将 CLIP 视觉特征拆成 vision component 和 language component，再通过 modality discriminator 对齐两种成分。 |
| PADA-Net | 结合 prompt learning、interactive bridge、dual Meta-nets、optimal transport 和博弈策略，在图像与文本之间建立交互对齐。 |
| PIMA | 用 invertible mapping 保持域内拓扑结构，避免对齐时破坏样本关系；同时用 cross-modal implicit contrastive learning 减少无关特征干扰。 |
| UTISA | 构造 domain-specific、layer-specific image prompts，并在多层级上与 text prompts 对齐。 |

#### 5.1.3 Distribution Alignment / Regularization

这类方法更接近传统 DA，但将 CLIP 语义引入伪标签、分布对齐和正则化中。

| 方法 | 原理 |
|---|---|
| CLIP-Div | 用 CLIP 的 domain-agnostic distribution 做源目标分布对齐，并通过 language-guided pseudo-labeling 校准目标域伪标签。 |
| DACR | 结合 prompt optimization、image adapter 和 consistency regularization，对目标域增强视图保持预测一致。 |
| CMKD | Cross-modal knowledge distillation，用 CLIP 图文知识蒸馏目标模型，同时通过稀疏/残差训练减少参数成本。 |
| SWG | Strong-Weak Guidance。高置信目标样本用强监督，低置信样本用软预测蒸馏，减少硬伪标签错误。 |
| CRPL | Cluster-preserving regularization。目标域样本往往有簇结构，方法在 prompt learning 中保持这种聚类结构，避免伪标签训练破坏目标分布。 |

#### 5.1.4 Fuzzy System-enhanced Adaptation

模糊系统方法的动机是：目标域无标签，样本类别归属可能不确定，不应该把伪标签都当成硬标签。

| 方法 | 原理 |
|---|---|
| VLMTSK-DA | 用 TSK fuzzy system 作为 image adapter，处理目标域不确定性，并结合 prompt learning 同步视觉与文本更新。 |
| FUZZLE | 引入 fuzzy prompt learning，使用 fuzzy C-means 和 instance-level fuzzy vectors 表示样本对类别的软隶属度，并用 KL divergence 约束学习。 |

#### 5.1.5 SS-OSUDA：单源开集 UDA

这里目标域包含源域未见类别：

\[
Y_s\subset Y_t
\]

模型要识别共享 known 类，同时把目标域私有类判为 unknown。

| 方法 | 原理 |
|---|---|
| ODA with CLIP | 利用 CLIP 和 ODA 模型输出熵判断样本是否属于 known 类。高不确定性样本更可能是 unknown。 |
| PromptDIV | 解耦 domain invariance 和 domain variance，用 one-vs-all text-based clustering 生成伪标签，并用 domain-specific prompt 表达不同域变化。 |
| COSMo | 使用 source-guided prompt learning 和 domain-specific bias network，并为 known 与 unknown 分别构造 prompt。 |

#### 5.1.6 MS-CSUDA 与 MS-OPSUDA

多源 DA 中，源域之间本身也有分布差异，因此不仅要适应目标域，还要融合多源知识。

| 方法 | 原理 |
|---|---|
| MPA | Multi-Prompt Alignment。为每个 source-target pair 学 prompt，再用 autoencoder 对齐多 prompt 表示。 |
| LanDA | Language-guided multi-source DA。用语言描述引导多个源域知识迁移，保留任务相关语义。 |
| MSDPL | 强调 domain prompt 的作用，联合学习 domain prompt 和 class prompt，并用 domain-aware mixup 增强泛化。 |
| VAMP | 使用 vision-aware prompts 生成 domain-guided text prompts，在多源场景中增强图文语义对齐。 |
| SAP-CLIP | 面向多源开放部分集，用 semantic-aware adaptive prompts 和 dynamic margin 同时处理 domain shift 与 class shift。 |

MS-OPSUDA 或 UniMDA 是更复杂的场景：源域和目标域部分类别重叠，同时各自有私有类。模型不仅要对齐共享类，还要抑制源域私有类，并检测目标域 unknown 类。

### 5.2 Source-Free DA

Source-Free DA 分为 SFF 和 SDF。SFF 没有源域数据、源模型或源统计，只依赖 CLIP、目标域无标签数据和类别名。SDF 没有源域原始数据，但有源域训练好的模型或源知识。

#### 5.2.1 SFF-CSUDA / CS-UFT

闭集 SFF-DA 假设目标域类别集合与预设类别一致，但没有源数据和源模型。

| 方法 | 原理 |
|---|---|
| UPL | Unsupervised Prompt Learning。用 CLIP 初始预测生成 top-K pseudo labels，并通过 pseudo-label ensemble 和 prompt representation ensemble 学 prompt。 |
| POUF | 使用 transport-based distribution alignment 和 mutual information maximization，使目标样本与类别原型匹配，同时避免所有样本塌缩到少数类。 |
| ReCLIP | 学一个 projection space 修正 CLIP 图文错位，生成伪标签后进行 cross-modal self-training，并迭代优化。 |
| LaFTer | Label-free tuning。只调很少参数，用自动生成文本和无标签图像集微调 VLM。 |
| CPL | Candidate Pseudolabel Learning。不直接给单一伪标签，而是保留候选伪标签集合，降低早期伪标签错误风险。 |
| DPA | Dual Prototypes Alignment。文本原型和图像原型互相校正，使伪标签更可靠。 |
| UEO | Universal Entropy Optimization。通过样本级置信度控制熵优化，同时调整 text prompt 与 visual affine transformation。 |
| TFUP | Training-free unsupervised prompt。尽量不训练，用 CLIP 相似度、置信度和原型分数选择可靠预测。 |
| Feng et al. | 解耦类别知识和域先验，增强 label-free 适应能力。 |

SFF-PSUDA、SFF-OSUDA、SFF-OPSUDA 则分别处理目标域是类别子集、目标域含未知类、源目标开放部分重叠等情况。相关方法如 UEO、UOTA、CLIPXpert 主要依赖熵优化、轻量 adapter、自适应阈值、SVD 特征过滤等方式区分 known/unknown 或筛选目标域实际出现类别。

#### 5.2.2 SDF-CSUDA / PSUDA / OSUDA

SDF 比 SFF 多了源模型，因此有更多任务知识可用。

| 方法 | 原理 |
|---|---|
| DIFO | 使用 frozen multimodal foundation model，结合 prompt learning、knowledge distillation 和 regularization 做 source-free adaptation。 |
| Co-learn++ | 让源模型、目标模型和 CLIP 协同产生伪标签。源模型提供任务知识，CLIP 提供语义知识。 |
| CDBN | 注入源域类别语义，保持 class information，同时增强目标域预测准确性和多样性。 |
| BBC | Black-box DA with CLIP。源模型可能只是 cloud API，方法结合 API 输出和 CLIP 生成 joint labels，再用 kNN 保持结构修正伪标签。 |
| ProDe | Proxy Denoising。构造可靠 proxy 修正 noisy VLM predictions，并在 PS/OS 中过滤 irrelevant categories 或 unknown samples。 |
| DUET | Dual-perspective pseudo labeling 与 uncertainty-aware exploration/exploitation，结合 Tsallis mutual information 优化目标域适应。 |

伪标签是 DA 中非常关键但也非常危险的机制。它不是“模型猜了就当真”，而是将模型高置信预测作为带噪弱监督。伪标签成立的前提是模型在目标域上已有部分可靠判断，并且必须配合高置信筛选、类别平衡、一致性约束、原型校正、熵控制等机制。否则错误伪标签会自我强化，导致 confirmation bias。

### 5.3 DA Discussion 与批判性理解

DA discussion 的核心判断是：CLIP-based DA 正在从 data-driven adaptation 转向 knowledge- and prior-driven alignment。传统 DA 强依赖源域标签数据和目标域无标签数据做分布对齐；CLIP-based DA 则越来越依赖 CLIP 预训练语义、文本类别原型、zero-shot 预测、prompt、蒸馏和目标域无标签数据。

文章中一个重要趋势是：source-free 方法与 source-available 方法的性能差距在缩小。SDF/SFF 方法在某些数据集上能够逼近甚至超过部分 SA 方法。但这不能简单理解为 source-free 方法已经解决问题，而应理解为 CLIP 预训练先验非常强。source-free 的“源”不在当前任务数据中，而在 foundation model 的预训练阶段中。

另一个趋势是 prompt tuning 很主流但很敏感。prompt 能快速适配目标域，但如果目标域伪标签错误、类别分布偏斜或域差异过大，prompt 容易被带偏。更稳定的方法通常会加入 distillation、consistency regularization、frozen CLIP constraint 或 uncertainty filtering。

ViT-B/16 backbone 的优势也需要注意。很多性能差异并不是 DA 算法本身带来的，而是 CLIP ViT 的预训练表示本身比传统 ResNet 强得多。因此阅读 CLIP-based DA 论文时必须看强 baseline：zero-shot CLIP、linear probing、简单 prompt tuning、简单 adapter tuning。如果没有这些比较，方法贡献可能被夸大。

---

## 6. 常用数据集与评价指标

### 6.1 Multi-domain datasets

多域数据集是 DG/DA 最常用 benchmark。一个数据集内部有多个 domain，每个 domain 通常类别相同但图像分布不同。DG 中通常选择一个或多个源域训练，剩余域直接测试；DA 中源域有标签、目标域无标签，训练时联合使用，测试在目标域进行。

| 数据集 | 常见域 | 特点 |
|---|---|---|
| PACS | Photo / Art painting / Cartoon / Sketch | 经典 DG 小数据集，类别少、域风格常见，CLIP 方法已接近饱和。 |
| VLCS | VOC / LabelMe / Caltech / SUN | 老牌 DG benchmark，规模和难度相对有限。 |
| Office-Home | Art / Clipart / Product / Real-World | DA/DG 都常用，难度中等，比 PACS 更接近日常物体分类。 |
| DomainNet | Real / Clipart / Painting / Sketch / Infograph / Quickdraw | 大规模、多类别、多域，尤其 quickdraw 和 infograph 很难，更能检验真实泛化能力。 |
| TerraIncognita | 不同野外相机地点 | 更接近真实环境变化，不只是风格迁移。 |
| Office-31 / VisDA | Amazon/DSLR/Webcam 或 synthetic-to-real | 更常见于 DA，尤其用于经典 UDA 和 synthetic-to-real 适应。 |

因此，PACS、VLCS 更适合做快速 sanity check；DomainNet、TerraIncognita、医学、遥感等复杂数据更适合检验真实域泛化。

### 6.2 Single-domain datasets

single-domain datasets 本身不一定有多个显式 domain，但可以通过 variants 构造分布偏移评估。典型例子是 ImageNet 及其变体：

| 数据集 | 用途 |
|---|---|
| ImageNet | 基础自然图像分类，常用于 zero-shot 或 few-shot prompt learning |
| ImageNetV2 | 测试 ImageNet 分布轻微变化下的鲁棒性 |
| ImageNet-Sketch | 测试从自然图像到素描域的泛化 |
| ImageNet-A | 包含对模型更困难的自然样本，测试鲁棒性 |
| ImageNet-R | 包含绘画、卡通、雕塑等 rendition 风格，测试风格泛化 |

这类数据集尤其适合分析 CLIP 的 zero-shot robustness 和 prompt learning 对分布外样本的影响。

### 6.3 Accuracy 与 HOS

在 CS 和 PS 场景中，目标域没有真正 unknown 类，因此常用 accuracy。CS 是源域和目标域类别完全一致；PS 中目标域类别是源域子集，目标域样本仍然都属于源域已知类，所以通常也用 ACC。

在 OS 和 OPS 场景中，目标域包含源域未见类，因此必须评价 unknown detection。常用指标是 HOS：

\[
HOS=
\frac{
2\times ACC_{\text{known}}\times ACC_{\text{unknown}}
}{
ACC_{\text{known}}+ACC_{\text{unknown}}
}
\]

其中 \(ACC_{\text{known}}\) 是共享类别上的分类准确率，\(ACC_{\text{unknown}}\) 是 unknown 样本被正确判为 unknown 的准确率。HOS 是调和平均，因此只做好 known 分类或只做好 unknown 检测都不够。如果 \(ACC_{\text{known}}\) 很高但 \(ACC_{\text{unknown}}\) 很低，HOS 会被拉低；反之亦然。

还要注意不同论文计算 \(ACC_{\text{known}}\) 的方式可能不同。有些用 per-class accuracy，先算每类准确率再平均，可以减少大类支配；有些直接按所有 known 样本整体 accuracy 计算。因此不同论文的 HOS 和 ACC 不一定完全可比。

---

## 7. Challenges and Future Directions

文章最后提出五个未来方向：realistic scenarios、LLM knowledge、multimodal fusion、interpretability、catastrophic forgetting。整体方向正确，但写法比较模板化，需要批判性理解。

**Realistic Scenarios** 指从 SA+CS 走向 SF、PS、OS、OPS。早期方法假设源域数据可用、源目标类别一致，这在真实部署中不够现实。真实场景可能源数据因隐私或版权不可访问，目标域类别不完整或有新类，类别长尾严重，部署时分布持续变化。因此 SF、PS、OS、OPS 确实更有研究价值。但文章没有深入展开哪些真实任务最关键，例如医疗跨医院、遥感跨传感器、自动驾驶跨天气城市、工业质检跨设备等。

**LLM Knowledge** 的动机是 CLIP 主要来自图文配对监督，虽然图文对齐强，但缺少更复杂的上下文推理和属性知识。LLM 可以生成类别属性、风格描述、域描述、未知类语义和更丰富的 prompt。但 LLM 可能 hallucinate，生成的属性也未必与视觉证据一致。未来真正难点不是简单把 GPT-4、Gemini、LLaMA 接进来，而是如何验证 LLM 知识、控制幻觉、保持图文对齐可靠性，并保证推理效率。

**Multimodal Fusion** 指 CLIP-based DG/DA 目前主要关注 image-text，但真实应用可能有视频、音频、LiDAR、雷达、GPS、医学结构化数据、传感器时间序列、多光谱遥感等。融合更多模态可以提供互补信息，比如夜间自动驾驶中摄像头弱而雷达强。但难点在于模态缺失、模态噪声、模态质量不平衡、跨模态不一致和动态权重分配。真正有价值的方向是根据当前域和样本动态判断哪个模态更可信，而不是简单拼接特征。

**Interpretability** 对 CLIP-powered 模型尤其重要。CLIP 的文本解释看似自然，但模型真实决策可能仍依赖背景、水印、风格或数据集偏差。可解释性不只是 attention map 或 saliency map，还要满足 faithfulness，即解释必须忠实于模型真实决策；也要满足 cross-domain consistency，即同一类别在不同域中的解释应关注相同语义区域，而不是不同域的 spurious feature。在医疗、金融、自动驾驶等场景中，可解释性是可信部署的基础。

**Catastrophic Forgetting** 是 fine-tuning CLIP 时的核心问题。CLIP 本来有强大的 zero-shot 和 open-vocabulary 能力，但微调到某个目标域后，可能目标域性能提高，原始通用语义能力下降。传统方法用 regularization、rehearsal、memory module、parameter isolation 等缓解；CLIP-based 方法中 prompt tuning、adapter、LoRA、MoA、ProGrad、KgCoOp、PromptSRC 等都可以理解为在“适配当前任务”和“保留 CLIP 原有知识”之间做折中。未来更真实的问题是 continual domain adaptation：目标域不是一次性变化，而是长期、增量、连续变化。

---

## 8. 全文总结与个人评价

这篇文章最大的贡献是 taxonomy 和 benchmark/metric 整理。它把 CLIP-based DG/DA 方法放进了 source accessibility、source number 和 label-space relationship 三个维度中，使读者能快速判断一个方法到底解决什么问题：是 DG 还是 DA，是 SA 还是 SF，是 SS 还是 MS，是 CS、PS、OS 还是 OPS。

方法层面，文章展示了 CLIP-based DG/DA 的共同趋势：从传统视觉特征对齐转向语义驱动的跨域泛化与适应。Prompt learning、adapter、distillation、pseudo-labeling、entropy optimization、LLM semantics、diffusion synthesis 等机制，本质上都在试图更好地利用或保护 CLIP 的图文语义先验。

但是文章也有不足。很多模型介绍过于罗列，缺少对关键假设、失败条件和方法边界的分析。最后的 future directions 比较模板化，像是把当前 AI 领域常见关键词集中列出。更重要的是，文章没有充分批判 CLIP 预训练数据对 benchmark 结果的影响。所谓 source-free 不是没有数据先验，而是依赖 foundation model 预训练。

**笔记层面的补充判断。** 这篇 survey 没有充分讨论 CLIP 预训练分布与 benchmark 之间的关系。一个可能解释是：PACS、VLCS 这类小型数据集的类别和风格高度常见，可能与 CLIP 预训练图文分布存在大量相似模式；因此 CLIP-based 方法在这些数据集上的高准确率，未必完全来自新的 DG/DA 机制。DomainNet 类别更多、域更复杂，尤其 quickdraw 和 infograph 更偏离普通互联网自然图像，因此更容易暴露 CLIP 语义先验的边界。这个判断需要实验验证，不能当作文章结论，但读这类结果时应该始终记住。

最终可以把本文浓缩为一句话：

> CLIP-powered DG/DA 的核心不是传统意义上的“重新发明域泛化算法”，而是围绕 CLIP 的图文语义先验，设计 prompt、adapter、蒸馏、伪标签和语义约束，使模型在源域、目标域、未知域、未知类之间尽量保持语义稳定；但真实复杂场景中的泛化、未知类识别、预训练分布依赖、benchmark 熟悉度和持续适应问题，仍然没有被根本解决。
