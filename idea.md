为了突破纯序列模型的“0.65瓶颈”，不能简单地将两个蛋白质的嵌入向量进行平均后拼接。平均池化（Mean Pooling）会抹平关键的局部结合位点信息。两个蛋白质是否相互作用，往往取决于它们表面特定的几个残基（“热点”残基）。因此，架构设计的核心必须是保留残基级别的分辨率，并在模型内部显式模拟两个序列之间的相互作用。我们提出 **SE-CAI (Siamese ESM-2 Cross-Attention Interaction)** 架构。


## 5.1 架构详解
该架构由三个主要模块组成：孪生编码器、交叉注意力交互模块、分类预测头。


### 5.1.1 模块一：孪生ESM-2编码器
输入是一对序列 $S_A$ 和 $S_B$。由于它们都是蛋白质（物理化学性质相同），我们使用**共享权重**的ESM-2模型来处理它们——这不仅减少了一半的参数量，还起到了正则化的作用。

$$\mathbf{H}_A = \text{ESM2}(S_A) \in \mathbb{R}^{L_A \times d}$$
$$\mathbf{H}_B = \text{ESM2}(S_B) \in \mathbb{R}^{L_B \times d}$$

其中：
- $L_A, L_B$ 分别为序列长度；
- $d=1280$ 为ESM-2 650M模型的隐藏层维度；
- 提取的是**最后一层所有Token的隐藏状态**（而非仅仅是 `<cls>` 标记）。


### 5.1.2 模块二：交叉注意力交互模块 (Cross-Attention)
这是本架构的创新核心。传统双塔模型的交互仅发生在最后的全连接层，而SE-CAI引入基于Transformer解码器机制的交叉注意力，让蛋白质A的每个残基去“查询”蛋白质B中的所有残基，反之亦然。


#### 步骤1：定义投影矩阵
查询（Query）、键（Key）、值（Value）矩阵通过线性变换得到：
$$Q_A = \mathbf{H}_A W_Q, \quad K_B = \mathbf{H}_B W_K, \quad V_B = \mathbf{H}_B W_V$$
$$Q_B = \mathbf{H}_B W_Q, \quad K_A = \mathbf{H}_A W_K, \quad V_A = \mathbf{H}_A W_V$$


#### 步骤2：计算交叉注意力
蛋白质A的残基查询蛋白质B的残基（$A \to B$），并加权得到融合特征；反之同理（$B \to A$）：

$$\text{Attn}_{A \to B} = \text{Softmax}\left(\frac{Q_A K_B^T}{\sqrt{d_k}}\right) V_B$$
$$\text{Attn}_{B \to A} = \text{Softmax}\left(\frac{Q_B K_A^T}{\sqrt{d_k}}\right) V_A$$

其中：
- $\text{Attn}_{A \to B} \in \mathbb{R}^{L_A \times d}$ 代表“注入了蛋白质B信息的蛋白质A特征”；
- 若蛋白质A的某个区域与B有强相互作用，注意力机制会赋予该区域更高权重。


### 5.1.3 模块三：特征融合与分类头
经过交互后，对特征进行聚合（保留最显著的结合位点特征），并构建最终特征向量用于分类。


#### 步骤1：特征聚合
同时使用**最大池化（Max Pooling）**和**平均池化（Mean Pooling）**：
$$\mathbf{v}_A = \text{Concat}\left( \text{MaxPool}(\text{Attn}_{A \to B}), \text{MeanPool}(\text{Attn}_{A \to B}) \right)$$
$$\mathbf{v}_B = \text{Concat}\left( \text{MaxPool}(\text{Attn}_{B \to A}), \text{MeanPool}(\text{Attn}_{B \to A}) \right)$$


#### 步骤2：构建最终特征向量
通过启发式匹配组合原始、差异和相似性特征：
$$\mathbf{v}_{final} = \text{Concat}\left( \mathbf{v}_A, \mathbf{v}_B, \mathbf{v}_A - \mathbf{v}_B, \mathbf{v}_A \odot \mathbf{v}_B \right)$$


#### 步骤3：分类预测
将 $\mathbf{v}_{final}$ 送入多层感知机（MLP）：
$$\text{MLP} = \text{Linear} \to \text{BatchNorm} \to \text{GELU} \to \text{Dropout} \to \text{Linear} \to \text{Sigmoid}$$

最终输出蛋白质相互作用的概率。


## 5.2 训练策略：参数高效微调 (PEFT/LoRA)
ESM-2 650M包含6.5亿参数，全量微调显存需求高且易在小数据集上发生**灾难性遗忘**。因此采用 **LoRA (Low-Rank Adaptation)** 技术：

冻结预训练模型权重 $W_0$，仅训练新加入的低秩矩阵 $A$ 和 $B$：
$$W = W_0 + \Delta W = W_0 + BA$$

其中：
- $B \in \mathbb{R}^{d \times r}, A \in \mathbb{R}^{r \times k}$（秩 $r$ 通常设为8或16）；
- 仅在Attention层的Query和Value投影矩阵中注入LoRA适配器，可训练参数量减少至原模型的**0.1%左右**；
- 支持单卡GPU大Batch训练，同时保留泛化能力。


## 5.3 损失函数设计：针对加权指标的优化
项目评估指标是 **Accuracy、Precision、Recall、F1的加权和**。标准二元交叉熵（BCE）主要优化Accuracy，对F1和Recall关注不足，因此采用 **Focal Loss**。


### Focal Loss原理
通过降低易分类样本的权重，专注于难分类样本（Hard Examples）：
$$L_{FL}(p_t) = -\alpha_t (1-p_t)^\gamma \log(p_t)$$

其中：
- $\gamma$ 为聚焦参数（通常设为2）；
- $\alpha_t$ 为平衡正负样本权重的参数。


### 阈值优化
验证阶段通过**网格搜索（Grid Search）**寻找最佳分类阈值 $\tau$：
- 默认阈值为0.5；
- 若模型倾向于高Precision低Recall，调低阈值（如0.4）可提升Recall和F1，从而提高总加权得分。