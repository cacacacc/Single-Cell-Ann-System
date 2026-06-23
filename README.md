# 单细胞高维向量 ANN 检索系统

面向单细胞高维向量数据的近似最近邻检索系统。核心目标是读取 `.h5ad` 单细胞数据、构建 ANN 索引，并通过 Web 页面提供 Top-K 相似细胞检索、UMAP 可视化、多算法性能评测，以及基于 RAG 的 AI 细胞助手智能问答。


## 快速上手

1. 克隆并进入项目目录：
```bash
git clone https://github.com/cacacacc/Single-Cell-Ann-System.git
cd Single-Cell-Ann-System
```

2. 创建 Python 环境并安装依赖：
```bash
pip install -r requirements.txt
pip install flask flask-cors python-dotenv anndata numpy pandas scipy
```

Windows 下 FAISS 需要通过 conda 安装：
```bash
conda install -c conda-forge faiss-cpu
```
> 如需使用 HNSWLIB 后端，还需安装：`pip install hnswlib`

3. 配置环境变量（AI 问答功能需要）：
```bash
# Windows
copy .env.example .env

# Mac / Linux
cp .env.example .env
```
用文本编辑器打开 `.env`，填入大模型 API Key（推荐智谱 GLM，注册即有免费额度）。

4. 下载示例数据，重命名为 `liver.h5ad` 后放入 `data/` 目录：

```text
https://datasets.cellxgene.cziscience.com/10cc50a0-af80-4fa1-b668-893dd5c0113a.h5ad
```

5. 启动服务：
```bash
python app.py
```

6. 打开浏览器访问：

```text
http://127.0.0.1:5000
```

默认管理员账号：

```text
账号：admin
密码：Admin@123456
```

首次运行会自动在 `data/users.sqlite3` 中创建用户表和默认管理员。正式部署前请通过 `.env` 文件修改 `SECRET_KEY` 和管理员凭据。


## 核心功能

### 1. 单细胞数据读取与管理

- 支持读取 `.h5ad` 格式的单细胞数据文件
- 内存高效加载：仅提取 `obsm`（降维向量）和 `obs`（元数据），不展开完整表达矩阵
- 支持多种向量表示：`X_pca`、`X_umap`、`X_tsne` 及原始表达矩阵 `X`
- 多数据集管理：支持上传、列出、删除多个数据集，自动校验文件格式和内容有效性
- 提供细胞分页查询和元数据字段浏览

### 2. ANN 索引构建与检索

- 支持三种后端：FAISS、HNSWLIB、NumPy（暴力检索）
- 支持五种索引类型：`flat`（精确）、`ivf_flat`（倒排）、`hnsw`（图索引）、`pq`（乘积量化）、`brute`（暴力）
- 支持三种距离度量：`l2`（欧氏距离）、`cosine`（余弦距离）、`ip`（内积）
- 支持 HNSW/IVF/PQ 关键参数配置（`M`/`ef_*`/`nlist`/`nprobe`/`pq_m`/`pq_nbits`）
- 每个数据集支持多个命名索引，按需构建和切换
- 支持索引文件保存与加载，减少重复构建时间
- 支持按细胞元数据字段（如细胞类型、组织来源）过滤检索结果

### 3. UMAP 降维可视化

- 基于 ECharts 的交互式散点图，支持数据缩放和平移
- 随机采样展示（可配置采样数量和随机种子）
- 按类别着色（如细胞类型），自动生成分类图例和分布统计
- 查询细胞高亮显示（涟漪动画效果），检索结果以不同颜色标注
- 支持按指定细胞 ID 批量查询 UMAP 坐标

### 4. 多算法性能评测

- 自动对比五种算法：暴力精确检索、FAISS-HNSW、HNSWLIB、FAISS-IVF、FAISS-PQ
- 以暴力检索为基准，计算各算法在不同 K 值下的 recall@K 曲线
- 柱状图展示平均查询延迟对比，折线图展示召回率曲线
- 评测结果持久化保存（最近 50 条），支持历史查看

### 5. RAG AI 细胞助手

- 自然语言问答：输入生物学问题，系统自动检索相似细胞并结合 LLM 生成专业回答
- 基于 ChromaDB 的向量数据库：将细胞 PCA 向量、元数据和高表达基因写入持久化向量库
- 多种检索策略：细胞向量检索、Embedding API 文本检索、关键词匹配检索
- SSE 流式回答：大模型回答逐字实时推送，前端实现打字机效果
- 多轮对话：支持会话上下文，每轮自动注入检索到的细胞数据作为上下文
- 对话历史持久化：支持 SQLite 持久化存储 + 内存缓存双存储
- 会话管理：支持会话列表、查看、重命名、删除
- LLM 角色预设：生信分析专家、严谨数据统计员、通俗科普助手
- RAG 溯源：展示检索到的相似细胞及其高表达基因，便于验证回答依据
- 支持智谱 GLM 和 OpenAI 兼容接口（DeepSeek、阿里通义等）

### 6. Web API 服务

- 使用 Flask 搭建后端服务，提供 40+ 个 REST API 接口
- CORS 支持，允许前端跨域访问
- 完整的认证体系：注册、登录、会话保持、角色权限控制
- 接收前端参数（数据集 ID、细胞编号、Top-K 数量、索引参数等），调用底层模块执行检索
- 返回 JSON 格式的检索结果，包含耗时统计和索引配置信息

### 7. 前端结果展示

- 深色科技风大屏界面，基于 Bootstrap 5 + ECharts 构建
- 系统概览仪表盘：展示数据集规模、索引状态、查询趋势等统计信息
- 数据管理中心：支持拖拽上传数据集、批量管理、ANN 索引构建和索引生命周期管理
- 相似检索页面：集成细胞浏览器、UMAP 散点图、Top-K 结果表格、CSV 导出
- 性能评测页面：可视化对比多种 ANN 算法的延迟和召回率
- AI 细胞助手页面：流式聊天界面，支持 Markdown 渲染和来源引用卡片
- 个人信息页面：编辑资料、修改密码、查看搜索快照与一键重跑

### 8. 用户信息模块

- 支持账号密码注册、登录、退出和会话保持（7 天有效期）
- 密码使用 Werkzeug `pbkdf2:sha256` 哈希存储，持久化到 SQLite
- 两种角色：普通用户和管理员
- 普通用户：查看可视化、执行检索、查看评测、维护个人资料
- 管理员：管理用户、上传/删除数据集、构建/删除索引、初始化/清空向量库
- 搜索快照：每次搜索自动保存参数快照，支持在个人中心查看和一键重跑
- 对话历史：AI 聊天消息持久化到 SQLite，支持跨会话查看

### 9. 数据集预处理

- 支持加载、清洗和对齐多个 `.h5ad` 数据集
- 每个数据集独立执行：去重 obs/var 名称、QC 过滤（低基因细胞和低表达基因）、可选标准化和 log1p 变换
- 基因对齐合并：支持 `inner`（交集）和 `outer`（并集）两种模式
- 输出合并后的 AnnData 文件和 JSON 处理报告


## 项目结构

```text
.
├── app.py                          # Flask Web 服务主入口（40+ 路由）
├── backend/
│   ├── __init__.py                 # 模块懒加载
│   ├── data_reader.py              # 数据读取（DataLoader）与多数据集管理（DatasetManager）
│   ├── ann_indexer.py              # ANN 索引构建与检索（FAISS/HNSWLIB/NumPy）
│   ├── vector_store.py             # ChromaDB 向量数据库封装
│   ├── llm_client.py               # 大模型 API 客户端（智谱 GLM / OpenAI 兼容）
│   ├── rag_engine.py               # RAG 检索增强生成引擎
│   ├── prompt_builder.py           # Prompt 工程与角色预设
│   ├── dataset_preprocessor.py     # 多数据集预处理管线
│   └── user_store.py               # 用户认证、权限与 SQLite 持久化
├── data/
│   ├── liver.h5ad                  # 示例单细胞数据（69032 细胞）
│   ├── users.sqlite3               # 自动生成的用户数据库
│   └── benchmark_history.json      # 性能评测历史记录
├── indexes/                        # ANN 索引文件（按数据集子目录存放）
├── chroma_db/                      # ChromaDB 持久化数据
├── docs/
│   ├── 项目说明.md                 # 详细说明文档
│   ├── 后端-数据开发/              # DataLoader 模块文档
│   ├── 后端-算法开发/              # ANN 算法设计与使用文档
│   ├── 后端-Web接口开发/           # API 接口使用文档
│   └── 后端-RAG开发/               # RAG/AI 问答模块文档
├── scripts/
│   └── prepare_datasets.py         # 数据集预处理 CLI 工具
├── static/
│   └── css/style.css               # 深色科技风主题样式
├── templates/
│   ├── base.html                   # 页面基础布局（侧边栏 + 顶栏）
│   ├── index.html                  # 系统概览仪表盘
│   ├── login.html                  # 登录/注册页面
│   ├── data_manage.html            # 数据管理中心
│   ├── search.html                 # 相似检索页面
│   ├── benchmark.html              # 性能评测页面
│   ├── chat.html                   # AI 细胞助手页面
│   ├── profile.html                # 个人信息页面
│   └── users.html                  # 用户管理页面（管理员）
├── tests/                          # 单元测试
├── .env.example                    # 环境变量配置模板
└── requirements.txt                # Python 依赖
```


## 环境变量配置

所有配置项均可通过 `.env` 文件或系统环境变量设置。完整配置模板见 `.env.example`。

### 大模型 API

| 变量 | 说明 | 默认值 |
|---|---|---|
| `LLM_PROVIDER` | 厂商：`zhipu`（推荐）或 `openai` | 自动检测 |
| `ZHIPU_API_KEY` | 智谱 GLM API Key | — |
| `OPENAI_API_KEY` | OpenAI 兼容 API Key | — |
| `LLM_MODEL` | 聊天模型名称 | `glm-4-flash` / `gpt-3.5-turbo` |
| `LLM_BASE_URL` | 自定义 API 地址 | 厂商默认 |
| `LLM_EMBEDDING_MODEL` | Embedding 模型 | `embedding-3` / `text-embedding-3-small` |
| `LLM_MAX_TOKENS` | 单次最大生成 token 数 | `1024` |
| `LLM_TEMPERATURE` | 生成温度 0~1 | `0.7` |
| `LLM_TIMEOUT` | API 超时秒数 | `60` |

### ANN 索引参数

| 变量 | 说明 | 默认值 |
|---|---|---|
| `CELL_INDEX_BACKEND` | 后端：`auto`/`faiss`/`hnswlib`/`numpy` | `auto` |
| `CELL_INDEX_TYPE` | 类型：`flat`/`ivf_flat`/`hnsw`/`pq`/`brute` | `flat` |
| `CELL_INDEX_METRIC` | 度量：`l2`/`cosine`/`ip` | `l2` |
| `CELL_INDEX_NLIST` | IVF 分桶数 | `100` |
| `CELL_INDEX_NPROBE` | IVF 探测数 | `10` |
| `CELL_INDEX_M` | HNSW M 参数 | `16` |
| `CELL_INDEX_EF_CONSTRUCTION` | HNSW 构建参数 | `200` |
| `CELL_INDEX_EF_SEARCH` | HNSW 查询参数 | `50` |
| `CELL_INDEX_PQ_M` | PQ 子空间数 | `8` |
| `CELL_INDEX_PQ_NBITS` | PQ 编码位数 | `8` |

### 系统配置

| 变量 | 说明 | 默认值 |
|---|---|---|
| `SECRET_KEY` | Flask Session 加密密钥 | `dev-secret-change-me` |
| `PORT` | 服务端口 | `5000` |
| `DEFAULT_ADMIN_USERNAME` | 默认管理员账号 | `admin` |
| `DEFAULT_ADMIN_PASSWORD` | 默认管理员密码 | `Admin@123456` |
| `DEFAULT_ADMIN_NAME` | 默认管理员姓名 | `系统管理员` |
| `DEFAULT_ADMIN_EMAIL` | 默认管理员邮箱 | `admin@example.com` |
| `DEFAULT_NEW_USER_PASSWORD` | 创建用户默认密码 | `Nankai@123` |
| `USER_DB_PATH` | 用户数据库路径 | `data/users.sqlite3` |


## 详细文档

如需完整接口说明、模块设计、协作规范与扩展规划，请查看：

| 文档 | 内容 |
|---|---|
| [项目说明](docs/项目说明.md) | 项目背景、技术栈、团队分工、系统流程、API 接口、进度跟踪 |
| [数据模块](docs/后端-数据开发/data_module_usage.md) | DataLoader 和 DatasetManager 接口说明 |
| [ANN 算法设计](docs/后端-算法开发/ann_algorithm_design.md) | ANN 算法选型与设计说明 |
| [ANN 模块使用](docs/后端-算法开发/ann_module_usage.md) | ANNIndexer 代码示例与参数配置 |
| [Web 接口](docs/后端-Web接口开发/api_usage.md) | 全部 REST API 接口参考 |
| [RAG 模块](docs/后端-RAG开发/rag_module_usage.md) | ChromaDB 向量库、LLM 接入、RAG 引擎使用 |


## 许可证

本项目仅用于课程学习、实验展示与教学交流。
