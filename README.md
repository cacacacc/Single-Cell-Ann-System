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
注意，windows下FAISS不会安装，建议使用conda进行安装：
```bash
conda install -c conda-forge faiss-cpu
```
3. 下载数据，重命名为`liver.h5ad`后放入 `data/` 目录，下载链接如下：`https://datasets.cellxgene.cziscience.com/10cc50a0-af80-4fa1-b668-893dd5c0113a.h5ad`。
4. 启动项目根目录的app.py：
```bash
python app.py
```
5. 打开前端页面并登录系统，默认服务地址：

```text
http://127.0.0.1:5000
```

开发默认管理员账号：

```text
账号：admin
密码：Admin@123456
```

首次运行会自动在 `data/users.sqlite3` 中创建用户表和默认管理员。正式部署前请通过环境变量修改 `SECRET_KEY`、`DEFAULT_ADMIN_USERNAME`、`DEFAULT_ADMIN_PASSWORD`，并及时修改默认密码。


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
- 支持配置索引类型与距离度量（`flat`/`ivf_flat`/`hnsw`/`pq`、`l2`/`cosine`/`ip`）
- 支持 HNSW/IVF/PQ 关键参数配置（`M`/`ef_*`/`nlist`/`nprobe`/`pq_m`/`pq_nbits`）
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

### 6. 用户信息模块

- 支持账号密码注册、登录、退出和会话保持
- 密码使用 Werkzeug 哈希存储，用户数据默认持久化到 SQLite
- 登录用户可以查看可视化系统、相似检索、数据集列表、性能评测和个人信息
- 普通用户可以修改个人资料和登录密码
- 管理员可以查看用户列表、添加用户、调整角色、启用/禁用账号、重置密码和删除用户
- 管理员权限保护数据上传、数据删除、索引构建和索引删除等系统管理操作

## 项目结构

```text
.
├── app.py                    # Flask Web 服务主入口
├── backend/
│   ├── data_reader.py        # 数据读取与向量提取模块
│   ├── ann_indexer.py        # ANN 索引构建与检索模块
│   └── user_store.py         # 用户注册、登录、权限与 SQLite 持久化模块
├── data/
│   ├── liver.h5ad            # 示例单细胞数据文件
│   └── users.sqlite3         # 自动生成的用户数据库
├── docs/
│   └── 项目说明.md           # 详细说明文档
├── static/                   # 静态资源（如 CSS）
├── templates/                # 前端页面模板
├── tests/
└── requirements.txt          # Python 依赖
```


如需完整接口说明、协作规范与扩展规划，请查看：`docs/项目说明.md`。
