# 蛋白质相互作用预测项目

## 📋 项目简介

本项目的目标是预测两个蛋白质是否会发生相互作用（Protein-Protein Interaction, PPI）。

**任务类型**: 二分类
**输入**: 两个蛋白质的氨基酸序列
**输出**: 0（不相互作用）或 1（相互作用）

## 📊 数据集说明

### 文件列表

| 文件名 | 说明 | 样本数 | 是否含label |
|-------|------|--------|------------|
| `train.csv` | 训练集 | 11,436 | ✅ |
| `valid.csv` | 验证集 | 2,287 | ✅ |
| `test.csv` | 测试集 | 1,525 | ❌ |
| `sample_submission.csv` | 提交格式示例 | 5 | - |
| `dataset_info.json` | 数据集统计信息 | - | - |

### 数据格式

#### 训练集和验证集 (train.csv, valid.csv)

```csv
protein_A,protein_B,sequence_A,sequence_B,label
9606.ENSP00000000233,9606.ENSP00000019317,MGLTVSALFS...,MTECFLPPTS...,1
9606.ENSP00000252172,9606.ENSP00000260682,MLWLALGPFP...,MACYIYQLPS...,0
```

**字段说明**:
- `protein_A`: 蛋白质A的ID
- `protein_B`: 蛋白质B的ID
- `sequence_A`: 蛋白质A的氨基酸序列
- `sequence_B`: 蛋白质B的氨基酸序列
- `label`: **标签** (1=相互作用, 0=不相互作用)

#### 测试集 (test.csv)

```csv
protein_A,protein_B,sequence_A,sequence_B
9606.ENSP00000178640,9606.ENSP00000256458,MLWLALGPFP...,MACYIYQLPS...
```

**注意**: 测试集**没有label列**，需要你的模型预测！

## 🎯 任务要求

### 1. 训练模型
使用 `train.csv` 训练你的模型

### 2. 本地验证
使用 `valid.csv` 调整超参数、验证模型性能

### 3. 生成预测
对 `test.csv` 中的每个蛋白质对进行预测

### 4. 提交结果

创建 `submission.csv` 文件，格式如下：

```csv
protein_A,protein_B,prediction
9606.ENSP00000178640,9606.ENSP00000256458,1
9606.ENSP00000253727,9606.ENSP00000257724,0
9606.ENSP00000215587,9606.ENSP00000216277,1
...
```

**重要**:
- 文件名必须是 `submission.csv`
- 必须包含3列：`protein_A`, `protein_B`, `prediction`
- `prediction` 必须是 0 或 1
- 必须包含测试集的所有1,525个样本
- 顺序不重要，但蛋白质对必须匹配

参考 `sample_submission.csv` 查看格式示例。

## 📈 评分标准

你的提交将通过以下**加权多指标**进行评分：

| 指标 | 权重 | 说明 |
|-----|------|------|
| **Accuracy** | 30% | 预测正确的比例 |
| **Precision** | 20% | 预测为正样本中真正为正的比例 |
| **Recall** | 20% | 真正样本中被正确预测的比例 |
| **F1-Score** | 30% | Precision和Recall的调和平均 |

**最终得分**:
```
Score = Accuracy × 0.3 + Precision × 0.2 + Recall × 0.2 + F1 × 0.3
```

得分范围: 0.0 ~ 1.0 (越高越好)