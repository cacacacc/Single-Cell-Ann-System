# 单细胞高维向量 ANN 检索系统

面向单细胞高维向量数据的近似最近邻检索系统。核心目标是读取数据、提取向量、构建 ANN 索引，并通过 Web 页面提供 Top-K 相似细胞检索。


## 快速上手

1. 克隆并进入项目目录：
```bash
git clone https://github.com/cacacacc/Single-Cell-Ann-System.git
cd Single-Cell-Ann-System
```

2. 创建 Python 环境并安装依赖（`requirements.txt`）
```bash
pip install -r requirements.txt
```

3. 下载数据，重命名为`liver.h5ad`后放入 `data/` 目录，下载链接如下：`https://datasets.cellxgene.cziscience.com/10cc50a0-af80-4fa1-b668-893dd5c0113a.h5ad`。
4. 启动项目根目录的app.py：
```bash
python app.py
```
5. 打开前端页面进行检索，默认服务地址：

```text
http://127.0.0.1:5000
```


## 核心功能

### 1. 单细胞数据读取

- 支持读取 `.h5ad` 格式的单细胞数据文件
- 使用 `scanpy.read_h5ad()` 加载 AnnData 数据
- 提取表达矩阵 `X`
- 提取降维向量 `obsm/X_pca`
- 提取细胞元数据 `obs`

### 2. 细胞向量提取

- 将单细胞表达矩阵或 PCA 结果转换为 NumPy 数组
- 统一转换为 `float32` 格式
- 为后续 ANN 索引构建提供标准输入

### 3. ANN 索引构建

- 支持使用 FAISS 或 HNSWLIB 构建近似最近邻索引，NumPy 暴力检索可作为 recall 基准
- 支持配置索引类型与距离度量（`flat`/`ivf_flat`/`hnsw`、`l2`/`cosine`/`ip`）
- 支持 HNSW/IVF 关键参数配置（`M`/`ef_*`/`nlist`/`nprobe`）
- 支持输入查询向量并返回 Top-K 相似细胞
- 返回相似细胞的内部编号和距离
- 支持索引文件保存与加载，减少重复构建时间

### 4. Web API 服务

- 使用 Flask 或 FastAPI 搭建后端服务
- 提供 `/api/search` 检索接口
- 接收前端传入的细胞编号和 Top-K 参数
- 调用底层 ANN 检索逻辑
- 返回 JSON 格式的检索结果

### 5. 前端结果展示

- 提供简洁的 Web 查询页面
- 支持输入查询细胞编号
- 支持设置 Top-K 数量
- 使用表格展示相似细胞结果
- 可扩展使用 ECharts 展示散点图或简单可视化结果

## 项目结构

```text
.
├── app.py                    # 外层入口
├── backend/
│   ├── data_reader.py        # 数据读取与向量提取模块
│   ├── ann_indexer.py        # ANN 索引构建与检索模块
│   └── app.py                # Web API 服务入口
├── data/
│   └── liver.h5ad            # 示例单细胞数据文件
├── docs/
│   └── 项目说明.md           # 详细说明文档
├── static/                   # 静态资源（如 CSS）
├── templates/                # 前端页面模板
├── tests/
└── requirements.txt          # Python 依赖
```


如需完整接口说明、协作规范与扩展规划，请查看：`docs/项目说明.md`。
