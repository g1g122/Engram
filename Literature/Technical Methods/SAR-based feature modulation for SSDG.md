# Similar Average Representation-Guided Feature Modulation for Domain-Agnostic SSDG

[Paper Link](https://doi.org/10.48550/arXiv.2503.20897)

## 目录

[1. 文章定位：SSDG 里的伪标签质量与无标签利用率问题](#1-文章定位ssdg-里的伪标签质量与无标签利用率问题)

[2. 背景脉络：DG、SSL、SSDG 方法版图](#2-背景脉络dgsslssdg-方法版图)

[3. FixMatch 作为底座：伪标签与一致性正则](#3-fixmatch-作为底座伪标签与一致性正则)

[4. SAR：从 Class Prototype 到 Similar Average Representation](#4-sar从-class-prototype-到-similar-average-representation)

[5. Feature Modulator：用 SAR 调制样本特征](#5-feature-modulator用-sar-调制样本特征)

[6. 训练流程：弱增强伪标签、强增强一致性与四个 Loss](#6-训练流程弱增强伪标签强增强一致性与四个-loss)

[7. 实验结果与证据链](#7-实验结果与证据链)

[8. 我的理解：它对 domain-agnostic 表征学习的启发](#8-我的理解它对-domain-agnostic-表征学习的启发)

[Appendix. 符号速查](#appendix-符号速查)

---

## 1. 文章定位：SSDG 里的伪标签质量与无标签利用率问题

这篇文章处理的是 **Semi-Supervised Domain Generalization, SSDG**。训练时有多个 source domains，每个 source domain 里只有少量有类别标签的数据，剩下大量样本没有类别标签；测试时面对一个训练阶段没见过的 target domain。它不是普通 DG，因为标签很少；也不是普通 SSL，因为训练和测试之间存在 domain shift。

论文采用的 shift 假设是：

\[
P(X) \text{ changes across domains, while } P(Y) \text{ remains the same.}
\]

也就是说，图像外观、风格、背景、纹理、采集环境会变，但类别空间保持一致。比如 source domains 里有 photo、cartoon、sketch，target domain 仍然识别同一组类别，只是视觉分布变了。

这里的 **domain-agnostic** 需要谨慎理解。它不是说整个实验流程完全不知道 domain。实验仍然需要 source/target domain 划分、leave-one-domain-out protocol、按 domain 统计结果，训练实现里也可能按 source domain 做采样。它强调的是：**domain label 不进入方法的核心计算**。模型不使用 domain-aware prototype、不构造 domain-specific task、不用 domain mask，也不依赖无标签样本的 domain label 来生成伪标签。

这篇文章真正抓住的是 SSDG 里的一个矛盾：

\[
\text{pseudo-label accuracy} \quad \text{vs.} \quad \text{unlabeled data utilization}
\]

高阈值可以让伪标签更准，但大量无标签样本被丢掉；低阈值可以使用更多无标签样本，但错误伪标签会把模型带偏。作者的主线是：先用 SAR-guided feature modulation 提高伪标签质量，再用 uncertainty 和 loss scaling 让更多低一些置信度但相对可靠的无标签样本参与训练。

---

## 2. 背景脉络：DG、SSL、SSDG 方法版图

### DG: Domain Generalization

| 模型/方向 | 想解决的问题 | 核心思想 | 怎么实现/关键机制 | 优点/局限 |
|---|---|---|---|---|
| IRM | 模型容易学到只在某些 domain 有效的捷径特征 | 真正稳定的特征，应该能让同一个分类器在所有 source domains 上都好用 | 按 domain 分别计算 risk，并加入 IRM penalty，惩罚同一个 classifier 在不同 domain 上不是共同最优的情况 | 目标直接指向跨域稳定关系；但优化困难，而且 domain 分组参与 loss 构造 |
| Meta-learning DG | 普通训练没有模拟泛化到未知域的过程 | 训练时把部分 source domains 临时当作未见域，让模型练习跨域泛化 | 每轮把 source domains 拆成 meta-train 和 meta-test；先在 meta-train 上更新，再用 meta-test loss 约束更新后仍能泛化 | 训练目标贴近 DG；但实现复杂，依赖 domain 划分，训练成本较高 |
| Data augmentation DG | 模型依赖颜色、纹理、背景、风格等 domain-specific 信息 | 人为制造更多 domain/style 变化，让表面特征不再可靠 | 对输入图像做强增强、风格扰动、Fourier mixing、style transfer、adversarial augmentation 等 | 简单直观；但增强是否覆盖真实测试域不确定，增强过强也可能破坏语义 |
| Feature augmentation / MixStyle | 不只在图像层增强，也想在特征层模拟 domain 变化 | 特征统计量如 mean/std 往往携带 style/domain 信息，混合这些统计量可生成新风格 | 在中间 feature map 上计算每个样本的通道均值和方差，与另一样本的统计量线性混合，再还原 feature；类别标签保持不变 | 开销小，能在特征空间制造风格变化；但主要改变 style statistics，不一定处理更复杂的 shift |
| Adversarial feature learning | 特征里残留 domain 信息，分类器会利用 domain shortcut | 让特征能预测类别，但不能预测 domain | 加 domain classifier；feature extractor 一边帮助 class classifier，一边通过 adversarial loss 或 gradient reversal 欺骗 domain classifier | 显式压制 domain 信息；但需要 domain label，对抗训练不稳定，也可能误删有用类别信息 |

### SSL: Semi-Supervised Learning

| 模型/方向 | 想解决的问题 | 核心思想 | 怎么实现/关键机制 | 优点/局限 |
|---|---|---|---|---|
| Consistency regularization | 无标签样本没有真实标签，但仍应提供训练约束 | 同一张图经过合理扰动后，预测应该一致 | 对无标签样本做不同增强或扰动，要求模型输出相近；预测不一致时产生 consistency loss | 不需要真实标签；但增强若改变语义，或模型初期预测很差，会带来错误约束 |
| Entropy minimization | 无标签样本上模型预测太犹豫，决策边界可能穿过高密度区域 | 模型应对真实样本给出明确预测 | 对无标签样本输出分布加熵惩罚，使预测从平坦分布变得尖锐 | 能推动类别簇形成；但容易强化错误自信 |
| Pseudo-labeling | 无标签样本不能直接监督训练 | 模型自己的高置信预测可以暂时当标签 | 对无标签样本预测类别分布，若最大概率够高，就把 argmax 类别当 pseudo-label，用 CE loss 训练 | 简单有效；但伪标签错了会自我强化 |
| MixMatch | 想综合利用增强、伪标签和 MixUp | 多个增强预测平均后生成软标签，再用 MixUp 平滑训练 | 对无标签样本做多次增强，平均预测并 sharpen，得到软伪标签；再把有标签和无标签样本一起 MixUp | 利用信息充分；但流程较复杂 |
| ReMixMatch | MixMatch 仍不够充分利用增强和类别分布信息 | 强增强和分布对齐能进一步提升 SSL | 在 MixMatch 基础上加入 strong augmentation、augmentation anchoring、distribution alignment 等机制 | 效果更强；但组件多，训练复杂 |
| FixMatch | 如何从无标签样本中获得可靠训练信号 | 高置信预测可作为伪标签；合理增强不应改变类别 | 用弱增强预测无标签样本，只有置信度超过阈值时才采纳为伪标签；再用该伪标签监督同一样本的强增强版本，使模型对强扰动保持一致 | 简单强力；但固定高阈值会丢掉很多无标签样本，降低阈值又增加伪标签噪声 |
| FreeMatch | FixMatch 固定阈值不适应训练阶段和类别差异 | 阈值应随模型学习状态自适应变化 | 根据训练过程中的模型置信度、类别学习情况动态调整 pseudo-label threshold | 能使用更多合适样本；但阈值估计本身也依赖模型状态 |
| FlexMatch | 不同类别学习进度不同，统一阈值会让难类长期缺少伪标签 | 课程式伪标签，不同类别阈值不同 | 估计每个类别的学习状态；学得慢的类别适当降低阈值，让更多该类无标签样本参与训练 | 缓解类别学习不均衡；但仍主要处理 SSL 内部问题，不直接处理 domain shift |
| FullMatch | 低置信无标签样本不能可靠给正标签，但仍有可用信息 | 不确定它是什么时，至少可以学习它不是什么 | 使用 EML 和 ANL；高置信样本做正向伪标签，低置信样本可通过 negative learning 排除明显不可能类别 | 更充分利用低置信样本；但机制复杂，错误负标签也会伤害训练 |
| HyperMatch | 伪标签有 clean/noisy 之分，直接丢 noisy 会浪费信息 | clean 和 noisy pseudo-label 都可用，但监督强度不同 | 把伪标签分成可靠和噪声两类；clean 用较强监督，noisy 用更鲁棒的对比或聚类约束 | 不完全浪费 noisy 样本；但依赖 clean/noisy 区分质量 |

### SSDG: Semi-Supervised Domain Generalization

| 模型/方向 | 想解决的问题 | 核心思想 | 怎么实现/关键机制 | 优点/局限 |
|---|---|---|---|---|
| StyleMatch | FixMatch 没有专门处理 domain/style shift，少标签时还容易过拟合 | 在 FixMatch 上加入 stochastic classifier 和 multi-view/style consistency | 分类器权重建模成高斯分布，训练时采样分类器减少过拟合；同时构造不同 style/view，要求预测一致 | 能增强跨风格稳定性；但机制依赖 domain/style 组织，训练比 FixMatch 复杂 |
| FBCSA | SSDG 中伪标签容易受 domain shift 影响 | 伪标签不仅要分类器自信，还要在特征空间和语义结构上合理 | 用 feature-based conformity 检查无标签样本特征是否符合对应类别原型；用 semantic alignment 让不同 domain 的同类语义对齐，通常构造 domain-aware class prototypes | 能提高伪标签质量；但实现中利用 domain label/domain-aware 结构 |
| MultiMatch | 把所有 source domains 混在一起会模糊各域差异，影响伪标签质量 | 每个 domain 是一个任务，同时学一个 global task | 将不同 source domain 视作不同 task，为每个 domain 学局部任务，再用 global task 学跨域共性 | 同时保留域内特性和跨域共性；但需要知道样本属于哪个 domain 来划分任务 |
| Joint domain-aware label / dual-classifier 方法 | 单一分类器生成伪标签可能不可靠 | 用 domain-aware label 机制和双分类器共同修正伪标签 | 一个分类器负责伪标签，另一个提供辅助判断；结合样本与 domain-aware class representation bank 的相似度来修正预测 | 伪标签更稳；但依赖 domain-aware representation bank，训练结构更复杂 |
| Known/unknown classes SSDG | 无标签或测试数据中可能有训练时未知类别 | SSDG 不只要处理 domain shift，还要区分 known 与 unknown classes | 在半监督域泛化框架下显式建模已知类和未知类，避免把未知类强行分到已知类别 | 处理更开放的场景；但问题设定和本文主线不同 |
| DGWM | 不同 domain 下同一分类器权重可能不都合适，伪标签会受 domain 偏移影响 | 用 domain 信息动态调制共享分类器权重 | 从某个 domain 的 batch 特征均值得到 domain-level information vector，再生成 \(C \times d\) 的 soft mask；用 mask 逐元素调制分类头权重 \(W\)，再用调制后的分类器生成伪标签/训练 | 按 domain 适配分类器，提高伪标签质量；但核心机制需要 domain-level 信息 |

---

## 3. FixMatch 作为底座：伪标签与一致性正则

FixMatch 是这篇文章的半监督底座。它有两个思想：pseudo-labeling 和 consistency regularization。

对一张无标签图 \(u\)，FixMatch 先做弱增强，得到模型预测。如果最高概率超过固定阈值，比如 0.95，就把该类别作为 pseudo-label。然后对同一张图做强增强，用刚才的 pseudo-label 作为训练目标。如果强增强后预测不一致，就产生 loss，反向传播修正模型。

这个设计背后的假设是：模型对轻微扰动下的样本已经很自信时，这个预测大概率可信；同时，同一张图经过合理强增强后，类别不应该改变。

问题在于固定阈值会造成取舍。阈值高，伪标签准，但 keep rate 低，大量无标签样本不用；阈值低，keep rate 高，但错误伪标签变多。到了 SSDG 里，domain shift 还会进一步降低伪标签稳定性，因为模型可能把 domain-specific appearance 当成类别线索。

本文的改动不是抛弃 FixMatch，而是替换和增强 FixMatch 中最脆弱的部分：伪标签生成和无标签 loss 权重。

---

## 4. SAR：从 Class Prototype 到 Similar Average Representation

每个类别的 class prototype 是有标签样本在当前 feature space 中的平均特征。若 \(f\) 是 feature extractor，类别 \(c\) 的有标签特征集合为 \(K_c\)，则：

\[
P_c = \frac{1}{|K_c|}\sum_{i=1}^{|K_c|}K_c[i]
\]

这里的平均就是逐特征维度求平均。若 ResNet-18 输出 512 维特征，那么 \(P_c\) 也是 512 维向量。

SAR 的关键是：它不是只使用本类 prototype，而是根据类别 prototype 之间的 cosine similarity，对所有类别 prototype 做加权平均：

\[
R_c =
\frac{\sum_{j=1}^{C} sim_{cj} \cdot P_j}
{\sum_{j=1}^{C} sim_{cj}}
\]

其中 \(sim_{cj}\) 是 \(P_c\) 和 \(P_j\) 的相似度。也就是说，每个类别 \(c\) 都有一个 \(R_c\)，而 \(R_c\) 是所有类别 prototype 的加权平均；只是和 \(c\) 越相似的类别权重越大。

直观上：

\[
P_{cat} = \text{cat 自己的类中心}
\]

\[
R_{cat} = \text{cat prototype + 相似类别 prototype 的加权混合}
\]

这使 SAR 带有相似类别上下文。比如 cat 和 dog 在视觉上接近，\(R_{cat}\) 可能会比普通 prototype 更明显地暴露 cat/dog 的相似区域，从而迫使后续 classifier 学习更细的 class-specific 差异。

SAR 本身不是可学习参数，不是 optimizer 直接更新的 \(nn.Parameter\)。它是由当前 feature extractor 对有标签数据提取出的 prototype 动态计算出来的统计表示。因为 \(f\) 在训练中会变，所以 \(P_c\)、prototype 相似度和 \(R_c\) 也会随训练变化。论文算法中每个 epoch 会重新计算 prototypes 和 SAR。

---

## 5. Feature Modulator：用 SAR 调制样本特征

Feature Modulator 是放在 feature extractor 和 classifier 之间的可学习门控矩阵。原始路径是：

\[
x \rightarrow f(x)=z \rightarrow C(z)
\]

本文改成：

\[
x \rightarrow f(x)=z \rightarrow M(z, R)=Z_m \rightarrow C(Z_m)
\]

设一共有 \(C\) 个类别，特征维度为 \(d\)。对一个样本的特征 \(z \in \mathbb{R}^{1 \times d}\)，先复制 \(C\) 份得到：

\[
Z \in \mathbb{R}^{C \times d}
\]

所有类别的 SAR 堆成：

\[
R \in \mathbb{R}^{C \times d}
\]

Feature modulator 是：

\[
M \in \mathbb{R}^{C \times d}
\]

调制公式为：

\[
Z_m = M \odot Z + (1-M)\odot R
\]

对单个类别 \(c\) 来看，就是：

\[
z_{m,c}=M_c \odot z + (1-M_c)\odot R_c
\]

所以 \(M\) 对单一类别而言可以理解为一个 feature-wise gate 向量；把所有类别的 gate 堆起来，就是 \(C \times d\) 的矩阵。若 \(M_c\) 的某一维接近 1，这一维主要保留样本自己的特征；若接近 0，这一维更多混入该类的 SAR。

这里的关键不是简单把样本往一个类别拉，而是对同一个样本同时做 \(C\) 次假设式调制：

\[
\text{mod to class 1},\text{ mod to class 2},\ldots,\text{ mod to class C}
\]

这不是说样本已经属于这些类别，而是在问：如果把它往某个类别的 SAR 方向调制，它在 classifier 看来是否更像这个类别？

因此，一个样本会得到 \(C\) 个 modulated features。每个 modulated feature 再经过同一个 classifier，classifier 又输出 \(C\) 类概率。于是得到一个 \(C \times C\) 的预测矩阵 \(S_i\)：

\[
S_i =
\begin{bmatrix}
\text{mod to class 1 的 C 类预测}\\
\text{mod to class 2 的 C 类预测}\\
\vdots\\
\text{mod to class C 的 C 类预测}
\end{bmatrix}
\]

行表示往哪个类别调制，列表示 classifier 预测哪个类别。它的对角线：

\[
diag(S_i)
\]

表示：

\[
\text{mod to class } c \rightarrow \text{predict class } c \text{ 的分数}
\]

本文最终用这个对角线来做分类和伪标签。对于无标签样本，\(\arg\max diag(S_i)\) 就是它的 pseudo-label 候选。

作者声称这一机制能让 classifier 更关注 class-specific features。更准确地说，不是 classifier 内部显式知道什么是猫脸、狗鼻子，而是训练机制给它制造了压力：输入不再是干净的原始特征，而是与相似类别 SAR 混合后的特征；如果 classifier 只依赖猫狗共有的 shared features 或 domain/style features，就难以在调制后的特征上稳定区分类别。为了降低 loss，它必须学习更独特、更稳定的类别判别线索。

---

## 6. 训练流程：弱增强伪标签、强增强一致性与四个 Loss

训练中真正被 optimizer 更新的是 feature extractor \(f\)、feature modulator \(M\) 和 classifier \(C\)。SAR 不直接反传更新，而是由当前 \(f\) 提取的有标签特征统计出来，并随 \(f\) 的变化重新计算。

有标签样本提供两类监督。第一类是普通分类监督：

\[
L_s = NLL(y_{pred}, y)
\]

这里 \(y_{pred}\) 来自 \(diag(S_i)\)。第二类是 diagonal maximizing loss：

\[
L_d = MSE(diag(S_i), col_{max})
\]

\(col_{max}\) 是 \(S_i\) 每一列的最大值。这个 loss 的直觉是：对某个类别而言，该类别的最高预测分数应该出现在 mod to this class 的那一行。它主要训练 \(M\) 学会让“往哪个类别调制”这件事在 classifier 看来确实对应那个类别。

无标签样本沿用 FixMatch 的弱增强/强增强逻辑，但伪标签不是从普通 classifier 输出直接来，而是从 feature modulation 后的 \(diag(S_i)\) 来。弱增强无标签样本会多次 forward，用 MC dropout 估计不确定性。论文中使用 5 次 forward：

\[
D_\mu = mean(diag(S_i))
\]

\[
D_\sigma = std(diag(S_i))
\]

伪标签为：

\[
\tilde{y} = \arg\max(D_\mu)
\]

最高平均置信度为：

\[
p_{max} = \max(D_\mu)
\]

伪标签类别上的不确定性为：

\[
\sigma = D_\sigma(\tilde{y})
\]

然后 loss scaling 为：

\[
l_{scale} = \mathbf{1}\{p_{max}-\sigma>\tau\}\cdot Q(p_{max})
\]

\[
Q(x)=\exp(x^3-1)
\]

这个设计不只看置信度，也扣掉预测波动。如果一个 pseudo-label 平均置信度高但多次 dropout 预测很不稳定，它仍然可能被拒绝。若通过阈值，\(Q(p_{max})\) 会让高置信伪标签影响更大、低一些置信度的伪标签影响更小。

强增强无标签样本用弱增强路径生成的 pseudo-label 训练：

\[
L_u = NLL(y_{upred}, \tilde{y})\cdot l_{scale}
\]

同时还有无标签版本的 diagonal loss：

\[
L_{ud} = MSE(diag(S_{ui}), col_{umax})\cdot l_{scale}
\]

最终 loss 是：

\[
L = L_s + L_u + \beta L_d + \gamma L_{ud}
\]

论文中使用 \(\beta=1\)、\(\gamma=0.5\)。

这套训练流程的本质是：有标签数据建立类别结构和训练基础分类能力；SAR 把类别结构变成可调制的类别代表；弱增强无标签样本生成更可靠的伪标签和不确定性；强增强无标签样本提供一致性训练；loss scaling 控制伪标签对梯度的影响。

---

## 7. 实验结果与证据链

主结果显示，本文方法相对 FixMatch 有稳定提升。四个 benchmark 包括 PACS、OfficeHome、VLCS 和 DigitsDG，实验设定是每类 5 个或 10 个有标签样本，其余训练样本作为无标签数据。

| Setting | Method | PACS | OfficeHome | VLCS | DigitsDG |
|---|---|---:|---:|---:|---:|
| 5 labels/class | FixMatch | 73.4 | 55.1 | 69.9 | 56.0 |
| 5 labels/class | Ours | 75.4 | 57.2 | 75.3 | 59.7 |
| 10 labels/class | FixMatch | 76.6 | 57.8 | 70.0 | 66.4 |
| 10 labels/class | Ours | 78.7 | 61.0 | 75.5 | 73.0 |

相对 FixMatch，5 labels/class 设置平均提升约 4.3%，10 labels/class 设置平均提升约 3.3%。提升最明显的是 VLCS、DigitsDG 和 OfficeHome，PACS 上提升较小，而且不是所有 setting 都绝对第一。例如 5 labels PACS 上 StyleMatch 更高，5 labels DigitsDG 上 FBCSA 更高。

Figure 2 的核心证据是三件事同时变好：

| Method | Average Accuracy | Keep Rate | PL Accuracy |
|---|---:|---:|---:|
| FixMatch | 57.8 | 63.5 | 87.1 |
| Ours | 61.0 | 67.6 | 90.4 |

这支持作者的中心主张：不是单纯使用更多无标签数据，也不是单纯提高伪标签准确率，而是同时提高 pseudo-label accuracy 和 keep rate。

消融实验也围绕这个主张展开。Feature modulator、SAR、variance initialization、diagonal loss、uncertainty score 和 loss scaling 都被单独或组合比较。比较重要的结论是：variance initialization 比普通随机初始化好；SAR 比普通 average prototype 好；diagonal loss 对训练 feature modulator 有帮助；单纯降低阈值会引入噪声，只有与 loss scaling 和 uncertainty 结合才更稳。

可视化证据主要来自 t-SNE。补充材料中的 Figure 5 和 Figure 6 对比了 FixMatch 的 feature extractor features 和本文的 modulated features。作者观察到：同一类别内部的 domain-wise separation 变弱，不同类别之间的 class separation 变强。这是对“减少 domain-specific 信息、增强 class-discriminative 信息”的间接支持。

但这篇文章没有直接把 \(M\) 的每个维度可视化成语义含义，也没有证明 \(M_c\) 接近 0 的维度一定对应背景、纹理或风格。关于 feature-wise gate 的解释更像是设计假设加消融和 t-SNE 支持，而不是强可解释性证明。

---

## 8. 我的理解：它对 domain-agnostic 表征学习的启发

这篇文章和纯监督 DG 的设定不同。它的主战场不是“有完整标签时如何学 invariant representation”，而是“标签很少时如何更可靠地使用无标签数据”。因此，如果一个项目中每个训练样本都有 class label，那么这篇文章的 SSDG 问题设定并不完全对应。

但它仍然有一个值得吸收的机制：**不使用 domain label 进入核心计算，也可以通过类别结构来压制 domain noise**。它没有显式按 domain 对齐，也没有为每个 domain 学 mask，而是从 class prototype 出发，用 SAR 描述相似类别结构，再通过 feature modulation 让样本特征向更稳定的类别代表靠近。

对 domain-agnostic 表征学习来说，这个思路有两个启发。第一，类别结构本身可以作为去 domain 的支点。只要 prototype 是跨 source domains 聚合出来的，它天然会平均掉一部分 domain-specific variation。第二，相似类别不一定只是干扰，也可以被用来制造更难的分类压力，迫使模型区分真正 class-specific 的特征，而不是依赖类别之间共有的粗糙特征。

我对这篇文章的保留是：它把 \(M\) 解释成压制 domain-specific features、保留 class-specific features，但证据主要是性能、消融和 t-SNE。若要把这个思想迁移到更强调机制解释的项目里，可能需要额外验证 \(M\) 到底在压制什么，例如可视化 gate 分布、做 domain predictability test、或者比较调制前后特征中的 domain 信息可分性。

---

## Appendix. 符号速查

| 符号 | 含义 |
|---|---|
| \(x_i^d, y_i^d\) | domain \(d\) 中第 \(i\) 个有标签样本及其类别标签 |
| \(u_i^d\) | domain \(d\) 中第 \(i\) 个无标签样本 |
| \(f\) | feature extractor |
| \(C\) | classifier，也表示类别数时需结合上下文区分 |
| \(z=f(x)\) | 单个样本的 instance-specific feature |
| \(P_c\) | 类别 \(c\) 的 class prototype，有标签样本特征均值 |
| \(sim_{cj}\) | 类别 \(c\) 和类别 \(j\) 的 prototype cosine similarity |
| \(R_c\) | 类别 \(c\) 的 Similar Average Representation |
| \(R\) | 所有类别 SAR 堆成的矩阵，形状 \(C \times d\) |
| \(M\) | feature modulator / modulating matrix，形状 \(C \times d\) |
| \(Z\) | 将单个样本特征 \(z\) 复制 \(C\) 份后的矩阵 |
| \(Z_m\) | 调制后的特征矩阵，形状 \(C \times d\) |
| \(S_i\) | 一个样本经过所有类别调制后再分类得到的 \(C \times C\) 预测矩阵 |
| \(diag(S_i)\) | mod to class \(c\) 后 predict class \(c\) 的分数集合 |
| \(col_{max}\) | \(S_i\) 每一列的最大值 |
| \(D_\mu, D_\sigma\) | 多次 MC dropout forward 后，\(diag(S_i)\) 的均值和标准差 |
| \(\tilde{y}\) | 无标签样本的 pseudo-label |
| \(l_{scale}\) | 无标签 loss 的缩放权重 |
| Keep Rate | 被用于无标签 loss 的无标签样本比例 |
| PL Accuracy | pseudo-label accuracy，伪标签准确率 |
