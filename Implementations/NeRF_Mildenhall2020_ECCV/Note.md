# NeRF: Neural Radiance Fields

[Paper Link](https://doi.org/10.48550/arXiv.2003.08934)

本文档旨在提供对 NeRF 原理、数学公式及其物理含义的严谨技术解析。内容涵盖场景表示、体积渲染积分、分层采样策略、位置编码机制以及坐标系变换等核心模块。

## 1. 场景表示 (Scene Representation)

NeRF 将静态 3D 场景表示为一个连续的 5D 函数 $F_\Theta$:
$$ F_\Theta : (\mathbf{x}, \mathbf{d}) \rightarrow (\mathbf{c}, \sigma) $$

### 输入与输出

- **输入**：
  - $\mathbf{x} = (x, y, z)$：三维空间坐标。
  - $\mathbf{d} = (\theta, \phi)$：二维观察方向（在实际实现中通常使用归一化的三维笛卡尔向量 $\mathbf{d} = (d_x, d_y, d_z)$）。
- **输出**：

  - $\mathbf{c} = (r, g, b)$：该点发射的颜色辐射（Radiance）。
  - $\sigma$：体积密度（Volume Density），表示光线在该点被阻挡的微分概率。

### 依赖关系 (View Dependence)

网络结构设计显式地约束了 $\sigma$ 和 $\mathbf{c}$ 的依赖关系：

1. **$\sigma(\mathbf{x})$ 仅依赖于位置 $\mathbf{x}$**：

   - 物理意义：物体的几何形状和透光性是其固有属性，不随观察角度改变。这保证了多视角几何的一致性（Multi-view Consistency）。
2. **$\mathbf{c}(\mathbf{x}, \mathbf{d})$ 依赖于位置 $\mathbf{x}$ 和方向 $\mathbf{d}$**：
   - 物理意义：模拟非朗伯体（Non-Lambertian）效应，如镜面反射（Specular Highlight）。同一空间点从不同角度观察可能呈现不同颜色。
   - 实现细节：方向 $\mathbf{d}$ 仅在 MLP 的最后层注入，确保其不影响 $\sigma$ 的预测。

---

## 2. 体积渲染 (Volume Rendering)

NeRF 通过可微的体积渲染公式将 3D 表示投影为 2D 图像。

### (1) 连续积分方程

沿相机光线 $\mathbf{r}(t) = \mathbf{o} + t\mathbf{d}$ 的颜色 $C(\mathbf{r})$ 定义为：
$$ C(\mathbf{r}) = \int_{t_n}^{t_f} T(t) \cdot \sigma(\mathbf{r}(t)) \cdot \mathbf{c}(\mathbf{r}(t), \mathbf{d}) \, dt $$

其中 **累积透射率 (Accumulated Transmittance)** $T(t)$ 定义为：
$$ T(t) = \exp \left( -\int_{t_n}^{t} \sigma(\mathbf{r}(s)) \, ds \right) $$

**物理含义解析**：

- **$T(t)$**：表示光线从起点 $t_n$ 传播到 $t$ 未发生碰撞（未被遮挡）的概率。

  - 当路径上 $\sigma$ 较大时，积分值增大，$\exp(-\int)$ 迅速衰减至 0。
  - 这项是处理**遮挡关系**的关键：若光线在 $t$之前已被阻挡，$T(t) \approx 0$，则 $t$ 处的颜色 $\mathbf{c}$ 对最终积分无贡献。
- **$\sigma(\mathbf{r}(t))$**：在该点发生碰撞的概率密度。
- **$\mathbf{c}(\mathbf{r}(t), \mathbf{d})$**：该点在给定观察方向下的颜色。

### (2) 离散化近似

为了计算机数值求解，需将积分离散化。NeRF 不直接使用固定的 $N$ 个点进行积分，而是在每个区间内采用**随机采样**，以保证连续空间的覆盖（详见第 3 节采样策略）。

对于对应的光线 $\mathbf{r}$，我们将积分范围 $[t_n, t_f]$ 划分为 $N$ 个均匀区间，第 $i$ 个采样点 $t_i$ 从每个区间内均匀随机采样：
$$ t_i \sim \mathcal{U} \left[ t_n + \frac{i-1}{N}(t_f - t_n), \,\, t_n + \frac{i}{N}(t_f - t_n) \right] $$

离散化的颜色贡献公式如下：
$$ \hat{C}(\mathbf{r}) = \sum_{i=1}^{N} T_i \cdot (1 - \exp(-\sigma_i \delta_i)) \cdot \mathbf{c}_i $$

- **$\mathbf{r}$**：射线（Ray），即从相机原点出发穿过像素中心的那条射线。
- **$t_i$**：第 $i$ 个采样点的深度值。
- **$T_i$**：离散形式的透射率，即 $T_i = \exp(-\sum_{j=1}^{i-1} \sigma_j \delta_j)$。
- **$\delta_i$**：采样点之间的距离，$t_{i+1} - t_i$。
- **$\alpha_i = 1 - \exp(-\sigma_i \delta_i)$**：第 $i$ 个区间的**不透明度 (Opacity)**。
  - 在 $\sigma_i \delta_i$ 较小的情况下，根据泰勒展开 $1 - e^{-x} \approx x$，该项近似于连续形式中的 $\sigma(t) dt$。
  - 使用 $\exp$ 形式保证了即使在采样间隔较大时，$\alpha_i$ 也严格限制在 $[0, 1]$ 范围内，提高了数值稳定性。

### (3) 端到端可微性

整个渲染过程仅涉及加减乘除与指数运算，均由基本初等函数构成。因此，渲染出的像素颜色 $\hat{C}(\mathbf{r})$ 对 MLP 的权重参数 $\Theta$ 是**完全可微**的。这使得可以通过计算预测图像与真实图像的误差（Loss），通过反向传播（Backpropagation）直接优化网络参数。

---

## 3. 采样策略 (Sampling Strategies)

### (1) Stratified Sampling (分层随机采样)

为了使 MLP 学习到连续的空间表示而非离散点，NeRF 在粗采样阶段引入了随机扰动：
$$ t_i \sim \mathcal{U} \left[ t_n + \frac{i-1}{N}(t_f - t_n), \,\, t_n + \frac{i}{N}(t_f - t_n) \right] $$
即将光线均分为 $N$ 个区间，在每个区间内**均匀随机**采样一个点。

- **作用**：将离散采样转化为对连续积分的无偏估计。**这是 Coarse 网络的采样策略**，通过这种"盲搜"来大致定位场景中物体的位置。

### (2) Hierarchical Volume Sampling (分层采样 - Coarse to Fine)

单纯的均匀采样效率低下，因为光线上大部分区域是空的。NeRF 设计了 **Coarse-to-Fine** 的双阶段采样策略：**用 Coarse 网络的预测结果来指导 Fine 网络的采样**。

**流程**：

1. **Coarse Network 推理（基于 Stratified Sampling）**：

   - 输入：$N_c$ 个分层随机采样点（即第 (1) 步产生的点）。

   - 输出：计算这些点的权重 $w_i = T_i (1 - \exp(-\sigma_i \delta_i))$。
2. **构建 PDF (概率密度函数)**：

   - 将权重归一化：$\hat{w}_i = w_i / \sum w_j$。此分布反映了光线上"物体存在"的概率。
   - **对应的连续形式**：
     $$ p(t) = \frac{w(t)}{\int_{t_n}^{t_f} w(s) \, ds}, \quad \text{其中 } w(t) = T(t)\cdot\sigma(t) $$

3. **Inverse Transform Sampling (逆变换采样)**：

   - 根据 $\hat{w}_i$ 构建累积分布函数 (CDF)。
   - 在 CDF 上进行**分层随机采样**（Stratified Sampling on CDF），反查得到 $N_f$ 个新采样点。这些点会自动集中在 PDF 较高（即物体表面）的区域。
4. **Fine Network 推理**：
   - 输入：将 Coarse 阶段的均匀采样点 $N_c$ 与新采样的重要性采样点 $N_f$ **合并并重新排序**（Union & Sort）。
   - **关键点**：Fine Network 并不只是看 $N_f$ 个点，而是把 $N_c$ 个"侦察点"和 $N_f$ 个"精修点"全都要了（共 $N_c + N_f$ 个点）。这就保证了 Fine Network 既覆盖了全图（来自 $N_c$），又在细节处有高分辨率（来自 $N_f$）。
   - 输出：Fine 网络的输出用于最终的图像渲染。

**注意**：Coarse 和 Fine 网络的推理是串行的（Coarse 结果指导 Fine 采样），但训练是同时的。

---

## 4. 位置编码 (Positional Encoding)

### 频谱偏置 (Spectral Bias) 问题

深度神经网络倾向于学习低频函数，难以捕捉高频细节（如纹理、锐利边缘）。直接输入低维坐标 $(x,y,z)$ 会导致渲染结果模糊。

### 解决方案

将输入坐标 $\mathbf{p}$ 映射到高维频域空间：

$$ \gamma(p) = (\sin(2^0 \pi p), \cos(2^0 \pi p), \dots, \sin(2^{L-1} \pi p), \cos(2^{L-1} \pi p)) $$

- **位置 $\mathbf{x}$**：使用 $L=10$（60维特征），保留几何细节。
- **方向 $\mathbf{d}$**：使用 $L=4$（24维特征），处理较平滑的视点变化。

此操作并非简单的特征增加，而是强制网络在不同的频率尺度上关注输入，从而能够拟合高频函数。

---

## 5. 训练与损失函数

### 损失函数

使用均方误差 (MSE) 直接比较渲染颜色与真实像素颜色：

$$ L = \sum_{\mathbf{r} \in \mathcal{R}} \left[ \| \hat{C}_c(\mathbf{r}) - C_{gt}(\mathbf{r}) \|_2^2 + \| \hat{C}_f(\mathbf{r}) - C_{gt}(\mathbf{r}) \|_2^2 \right] $$

- **$L_{total} = L_{coarse} + L_{fine}$**。
- Coarse 网络的 Loss 必须保留，因为 Fine 网络的采样依赖于 Coarse 网络输出的权重分布。若 Coarse 网络未经过训练，其输出的 PDF 将是均匀或随机的，导致 Fine 网络无法获得有效的采样点。
- 梯度分别回传至 Coarse 和 Fine 网络，互不干扰。

---

## 6. 坐标系与空间变换

### 6.1 相机位姿 (Camera Extrinsics)

数据集提供的 `transform_matrix` 为 **Camera-to-World (c2w)** 矩阵（$4 \times 4$）：

$$ \mathbf{T} = \begin{bmatrix} \mathbf{R} & \mathbf{t} \\ \mathbf{0} & 1 \end{bmatrix} $$

- **$\mathbf{R} (3 \times 3)$**：旋转矩阵。定义相机坐标系的轴在世界坐标系中的方向。
- **$\mathbf{t} (3 \times 1)$**：平移向量。定义相机光心在世界坐标系中的位置。

### 6.2 旋转矩阵 $\mathbf{R}$ 的行与列：严格数学定义

设世界坐标系的三个基向量为 $\mathbf{e}_x, \mathbf{e}_y, \mathbf{e}_z$，相机坐标系的三个基向量为 $\mathbf{c}_x, \mathbf{c}_y, \mathbf{c}_z$。

c2w 旋转矩阵的每一个元素是相机轴和世界轴的**点积（投影）**：

$$R = \begin{bmatrix}
\mathbf{c}_x \cdot \mathbf{e}_x & \mathbf{c}_y \cdot \mathbf{e}_x & \mathbf{c}_z \cdot \mathbf{e}_x \\
\mathbf{c}_x \cdot \mathbf{e}_y & \mathbf{c}_y \cdot \mathbf{e}_y & \mathbf{c}_z \cdot \mathbf{e}_y \\
\mathbf{c}_x \cdot \mathbf{e}_z & \mathbf{c}_y \cdot \mathbf{e}_z & \mathbf{c}_z \cdot \mathbf{e}_z
\end{bmatrix}$$

#### 列 = 相机坐标轴

第 0 列：

$$\text{col}_0 = \begin{bmatrix} \mathbf{c}_x \cdot \mathbf{e}_x \\ \mathbf{c}_x \cdot \mathbf{e}_y \\ \mathbf{c}_x \cdot \mathbf{e}_z \end{bmatrix}$$

三个元素都包含 $\mathbf{c}_x$（相机 X 轴）。这一列就是**相机 X 轴在世界坐标系中的方向向量**。同理，第 1 列是相机 Y 轴，第 2 列是相机 Z 轴。

#### 行 = 世界坐标分量

第 0 行：

$$\text{row}_0 = \begin{bmatrix} \mathbf{c}_x \cdot \mathbf{e}_x & \mathbf{c}_y \cdot \mathbf{e}_x & \mathbf{c}_z \cdot \mathbf{e}_x \end{bmatrix}$$

三个元素都包含 $\mathbf{e}_x$（世界 X 轴）。当计算 $\mathbf{p}_w = R \cdot \mathbf{p}_c$ 时，第 0 行与 $\mathbf{p}_c$ 的点积直接决定结果的**世界 X 分量**。同理，第 1 行决定世界 Y 分量，第 2 行决定世界 Z 分量。

**总结**：c2w 矩阵是相机和世界坐标系之间的"翻译合同"。在这个矩阵里，**列属于相机，行属于世界**。

### 6.3 LLFF 坐标系转换

#### 问题

LLFF 数据集的相机坐标系定义为 $(\text{down}, \text{right}, \text{backwards})$，即相机的 X 轴朝下、Y 轴朝右、Z 轴朝后。而 NeRF 的 `get_rays` 假设相机坐标系为 $(\text{right}, \text{up}, \text{backwards})$。

#### 为什么交换列

`get_rays` 在相机坐标系下构造光线方向 $((i - W/2)/f, -(j - H/2)/f, -1)$，其中第 0 个分量对应相机 X 轴（假设为 right），第 1 个分量对应相机 Y 轴（假设为 up）。这些方向随后通过 $R$ 的列向量转到世界坐标系：

$$ \mathbf{d}_{\text{world}} = R \cdot \mathbf{d}_{\text{cam}} = d_x \cdot \text{col}_0 + d_y \cdot \text{col}_1 + d_z \cdot \text{col}_2 $$

如果 $R$ 的列含义仍然是 LLFF 的 $(\text{down}, \text{right}, \text{backwards})$，那么 $d_x$（本应乘以 right 方向）会乘以 down 方向，光线方向就完全错了。

因此需要**交换列**，使 c2w 矩阵的列含义与 `get_rays` 的相机坐标系假设一致：

$$R_{\text{NeRF}} = \begin{bmatrix} | & | & | \\ \text{right} & \text{up} & \text{backwards} \\ | & | & | \end{bmatrix} = \begin{bmatrix} | & | & | \\ \text{col}_1^{\text{LLFF}} & -\text{col}_0^{\text{LLFF}} & \text{col}_2^{\text{LLFF}} \\ | & | & | \end{bmatrix}$$

即：新 col 0 = 旧 col 1（right），新 col 1 = $-$旧 col 0（$-$down = up），新 col 2 不变（backwards）。

#### 行交换 vs 列交换

某些实现脚本中对 LLFF 数据做的是**行交换**而非列交换。这是因为那些实现同时改变了世界坐标系和相机坐标系的约定——LLFF 的世界坐标系也遵循 $(\text{down}, \text{right}, \text{backwards})$ 排列，行交换将其重排为 $(\text{right}, \text{up}, \text{backwards})$。

两种做法（行交换 vs 列交换）在各自管线内都是自洽的，但不能混用。关键是保证 c2w 矩阵的列含义与 `get_rays` 的假设匹配。

### 6.4 叉积与坐标系手性

在构建正交坐标系时（例如 `_average_poses` 中从平均方向构造正交基），叉积顺序必须遵循 **xyz 正循环**：

$$ \mathbf{x} \times \mathbf{y} = \mathbf{z}, \quad \mathbf{y} \times \mathbf{z} = \mathbf{x}, \quad \mathbf{z} \times \mathbf{x} = \mathbf{y} $$

**逆序则取负**：$\mathbf{y} \times \mathbf{x} = -\mathbf{z}$，以此类推。

若坐标轴为 $(\text{right}=x, \text{up}=y, \text{backwards}=z)$，则求 right 时应使用 $\text{right} = \text{up} \times \text{backwards}$（即 $y \times z = x$）。写反为 $\text{backwards} \times \text{up}$（即 $z \times y = -x$）会得到 $-\text{right}$，构造出的矩阵行列式为 $-1$（反射矩阵而非旋转矩阵），导致场景被镜像翻转。

**注意**：c2w 矩阵第 2 列存储的是 backwards 方向（$+z$），不是 forward 方向（$-z$）。变量命名时需注意，否则容易在推导叉积顺序时搞反。

### 6.5 世界原点的定义

原点的位置取决于数据来源：

1. **合成数据（如 Blender/Unity）**：
   - 原点由建模者决定，通常是物体的中心。相机围绕原点旋转。

2. **真实数据（如 LLFF/COLMAP）**：
   - 使用 SfM（Structure from Motion）算法（如 COLMAP）估算位姿。
   - COLMAP 通常以第一张照片为基准（位置设为原点，朝向设为主轴）。
   - **归一化 (Normalization)**：代码通常会进行 `recenter_poses`（计算所有相机的平均位姿，将其变换为坐标原点）和缩放，将场景约束在合理范围内。

### 6.6 场景边界处理 (Scene Bounds)

根据场景类型选择不同的空间划分策略：

1. **有界场景（如 Lego）**：
   - 适用于 $360°$ 围绕拍摄。
   - 设定固定的近平面 $t_n$ 和远平面 $t_f$（例如 $[2.0, 6.0]$），仅在此范围内采样。
   - 假设背景为白色或黑色，不需要对无限远进行建模。

2. **无界场景（如 LLFF）**：
   - 适用于 Forward-Facing（前向）拍摄。
   - 场景深度范围极大（$[t_n, \infty]$）。
   - 需要 **NDC（Normalized Device Coordinates）变换**（详见 6.7 节）。

### 6.7 NDC 变换（Normalized Device Coordinates）

#### 动机

对于前向场景，深度范围从近平面 $n$ 到无穷远。如果在原始空间均匀采样，绝大部分采样点会浪费在远处的空区域。NDC 变换将无限深的视锥体映射到有限的 $[-1, 1]^3$ 立方体中，使得均匀采样自动在近处更密集。

#### 对坐标系的隐式要求

NDC 变换公式硬编码了以下假设：

- 分量 0（$x$）对应图像**水平方向**，公式中除以图像宽度 $W$
- 分量 1（$y$）对应图像**竖直方向**，公式中除以图像高度 $H$
- 分量 2（$z$）对应**深度方向**，相机看向 $-z$

因此，在应用 NDC 之前，世界坐标系中光线的各分量必须满足上述排列。这也是 LLFF 坐标转换不可省略的原因——如果不转换，$x$ 分量是 down 方向而非水平方向，NDC 公式会将"往下的距离"当作"水平距离"来投影。

#### 数学推导

给定光线 $\mathbf{r}(t) = \mathbf{o} + t\mathbf{d}$，首先将光线原点平移到近平面 $z = -n$：

$$ t_0 = -\frac{n + o_z}{d_z}, \quad \mathbf{o}' = \mathbf{o} + t_0 \mathbf{d} $$

此时 $o'_z = -n$。然后对原点进行透视投影：

$$o'_x = -\frac{2f}{W} \cdot \frac{o'_x}{o'_z}, \quad o'_y = -\frac{2f}{H} \cdot \frac{o'_y}{o'_z}, \quad o'_z = 1 + \frac{2n}{o'_z}$$

方向的 NDC 变换为：

$$d'_x = -\frac{2f}{W} \left( \frac{d_x}{d_z} - \frac{o'_x}{o'_z} \right), \quad d'_y = -\frac{2f}{H} \left( \frac{d_y}{d_z} - \frac{o'_y}{o'_z} \right), \quad d'_z = -\frac{2n}{o'_z}$$

变换后的光线在 NDC 空间中，$z'$ 从 $0$（近平面）线性变化到 $1$（无穷远），因此采样时设 $t_n = 0, t_f = 1$ 即可。NDC 空间中的均匀采样等价于原始空间中的**视差（disparity，$1/z$）线性采样**，自动在近处分配更多采样点。

---

## 7. 螺旋渲染路径 (Spiral Render Path)

训练完成后，需要生成一条相机轨迹来渲染新视角视频。对于前向场景，常用的是螺旋路径。

### 构造方法

1. **计算平均位姿**：从所有训练相机中计算平均位置和朝向，作为螺旋的中心。

2. **确定 look-at 目标**：在平均相机的**前方**（沿 $-z$ 方向）偏移一个焦距深度，将目标放在场景中心附近。注意 c2w 矩阵第 2 列是 backwards（$+z$），所以目标位置应为：
   $$ \mathbf{target} = \mathbf{position} - \mathbf{backwards} \cdot d_{\text{focal}} $$
   如果误用 $+$ 号，目标会跑到相机背后，导致渲染相机"转过身"去看，产生翻转的图像。

3. **在局部坐标系中生成偏移**：螺旋偏移量 $(\cos\theta, \sin\theta, \ldots)$ 是在相机的局部坐标系（right, up, backwards）中定义的。必须通过平均位姿的旋转矩阵将其转换到世界坐标系：
   $$ \mathbf{p}_{\text{new}} = \mathbf{p}_{\text{avg}} + R_{\text{avg}} \cdot \mathbf{offset}_{\text{local}} $$
   如果直接将偏移加在世界坐标的 $x, y, z$ 上，螺旋平面不会对齐相机的视平面，渲染结果会出现畸变。

4. **构造 look-at 矩阵**：对每个螺旋位置，构造一个从新位置看向目标点的 c2w 矩阵。
