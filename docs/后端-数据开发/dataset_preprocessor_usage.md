# 多数据集基因对齐预处理使用说明

## 1. 功能用途

多个单细胞 `.h5ad` 数据集的基因列表往往不完全一致。如果直接合并，表达矩阵的列含义无法对应，后续统一检索、建索引或分析都会产生问题。

本功能用于将多个 `.h5ad` 数据集按基因名对齐，生成一个新的合并数据集，并输出可读报告与结构化处理记录。

典型使用场景：

- 将多个来源的单细胞数据集整理成一个统一数据集
- 解决不同数据集基因维度不一致的问题
- 为后续统一 ANN 检索、RAG 检索或数据分析准备输入文件
- 展示“跨数据集预处理与基因对齐”的完整处理流程


## 2. 前端使用方式

管理员登录后进入：

```text
数据管理中心
```

点击顶部按钮：

```text
基因对齐预处理
```

在弹窗中完成以下配置：

| 配置项 | 说明 | 推荐默认值 |
|---|---|---|
| 源数据集 | 选择至少 2 个原始 `.h5ad` 数据集 | 选择两个已有数据集 |
| 输出数据集名称 | 新生成的 `.h5ad` 文件名，不需要写后缀 | `joint_aligned` |
| 基因对齐方式 | `inner` 或 `outer` | `inner` |
| min cells | 基因至少在多少个细胞中被检测到才保留 | `3` |
| min genes | 细胞至少检测到多少个基因才保留 | `200` |
| normalize total | 是否进行总量归一化 | 默认开启 |
| log1p 转换 | 是否进行 `log(1 + x)` 转换 | 默认开启 |

点击“开始预处理”后，系统会读取源数据集、清洗并对齐基因，完成后新数据集会自动出现在数据集列表中。


## 3. inner 与 outer 的区别

假设两个数据集的基因列表如下：

```text
数据集 A：Gene1, Gene2, Gene3
数据集 B：Gene2, Gene3, Gene4
```

### inner：只保留共同基因

```text
结果：Gene2, Gene3
```

优点：结果更稳妥，所有保留下来的基因在每个数据集中都存在。

适合：保守分析、避免大量补 0 的场景。

### outer：保留全部基因

```text
结果：Gene1, Gene2, Gene3, Gene4
```

某个数据集缺失的基因表达值会补 0。

优点：保留信息更多。

适合：希望保留所有基因、并能接受缺失值补 0 的场景。


## 4. 输出结果

假设输出数据集名称填写为：

```text
joint_aligned
```

系统会生成：

```text
data/joint_aligned.h5ad
data/joint_aligned_report.json
```

其中：

- `joint_aligned.h5ad` 是对齐并合并后的新数据集
- `joint_aligned_report.json` 是结构化处理记录，可在前端查看可读报告并导出 PDF

处理报告包含：

| 字段 | 说明 |
|---|---|
| `join` | 使用的基因对齐方式 |
| `n_datasets` | 源数据集数量 |
| `total_cells` | 合并后的细胞总数 |
| `aligned_genes` | 对齐后保留的基因数量 |
| `datasets` | 每个源数据集处理前后的细胞数和基因数 |


## 5. 命令行用法

前端按钮底层调用的是 `backend/dataset_preprocessor.py`。也可以直接使用命令行脚本：

```bash
.venv/bin/python scripts/prepare_datasets.py \
  data/liver.h5ad \
  data/liver_IFALD.h5ad \
  -o data/joint_aligned.h5ad \
  --report data/joint_aligned_report.json \
  --join inner
```

常用参数：

```text
--join inner       只保留共同基因
--join outer       保留全部基因，缺失补 0
--min-cells 3      基因过滤阈值
--min-genes 200    细胞过滤阈值
--normalize-total  执行总量归一化
--log1p            执行 log1p 转换
```


## 6. 后端接口

前端调用接口：

```http
POST /api/datasets/preprocess
```

请求体示例：

```json
{
  "source_datasets": ["liver", "liver_IFALD"],
  "output_name": "joint_aligned",
  "join": "inner",
  "min_cells": 3,
  "min_genes": 200,
  "normalize_total": true,
  "log1p": true
}
```

响应示例：

```json
{
  "status": "preprocessed",
  "dataset_id": "joint_aligned",
  "output_path": "data/joint_aligned.h5ad",
  "report_path": "data/joint_aligned_report.json",
  "report": {
    "join": "inner",
    "n_datasets": 2,
    "total_cells": 100000,
    "aligned_genes": 18000
  }
}
```


## 7. 常见问题

### 为什么推荐先用 inner？

`inner` 只保留所有数据集共同拥有的基因，结果更容易解释，也能减少补 0 带来的影响。

### 为什么生成后的数据集可能基因数变少？

如果选择 `inner`，系统会取多个数据集的基因交集。源数据集差异越大，共同基因可能越少。

### 为什么处理很慢？

`.h5ad` 文件可能很大。预处理需要读取表达矩阵、过滤细胞和基因、重新排列基因维度并写出新文件，因此比普通检索更耗时。

### 新生成的数据集能继续建索引吗？

可以。新数据集会出现在数据管理列表中，之后可以像普通数据集一样构建 ANN 索引。
