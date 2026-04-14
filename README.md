# 🪐 Gemma 4 Agentic RAG 

![License](https://img.shields.io/badge/License-MIT-blue.svg)
![Python](https://img.shields.io/badge/Python-3.10%2B-green.svg)
![Architecture](https://img.shields.io/badge/Architecture-Multi--Agent-orange.svg)

**Gemma 4 Agentic RAG** 是一个企业级、支持本地私有化部署的多模态智能检索增强生成系统。本项目主导研发了基于图状态机（LangGraph）的智能路由编排逻辑，深度整合多模态解析、混合检索与四级级联爬虫设计，为复杂业务场景提供高召回、高置信度且自带防幻觉溯源的专家级问答服务。

---

## ✨ 核心特性 (Core Features)

- 👁️ **数据接入与多模态解析引擎**: 针对复杂 PDF/图像采用 **MinerU** 与 **RapidOCR** 进行高保真版面分析与图文提取；针对 Office 全家桶接入 **LlamaIndex** 原生解析器，实现多源异构数据的标准化切块。
- 🔍 **混合检索与深度重排架构**: 构建基于 **Milvus + ES** 的 BM25 稀疏检索与 BGE-M3 稠密向量双路混合召回引擎。配合 **bge-ranker** 实现交叉重排与局部悬崖过滤，大幅提升检索精度。
- 🌐 **全域级联联网检索与动态反思 (Reflection)**: 设计 `SearXNG -> Crawl4AI -> Tavily -> Exa` 四级级联策略。引入动态反思模块，基于原子词覆盖率自主决策熔断或触发补偿检索，确保数据的实时性与饱和度。
- 🕸️ **图状态机调度与智能路由**: 利用 **LangGraph** 思想构建节点流转图，实现本地/联网/多模态意图的智能路由分发。模型端基于 **llama.cpp** 对 **Gemma 4 E4B** 进行量化部署，支持多线程调度与工具调用。
- 🛡️ **高置信度溯源生成**: 结合 **Small-to-Big** 策略增强上下文，并在 System Prompt 中建立强约束机制，强制 LLM 输出携带精准引用的 Markdown 锚点（如 `[文件X, 第N页]`），消除信息拼凑导致的大模型幻觉问题。

---

## 🐍 环境隔离部署 (Dual-Environment Isolation)

为了彻底解决 **MinerU (视觉解析)** 与 **Gemma 4 (业务逻辑)** 之间复杂的底层依赖冲突，本项目采用了**双虚拟环境并行架构**：
1. `venv_gemma`: 承载主 UI (Streamlit)、RAG 引擎、意图路由器及所有 Web 业务。
2. `venv_mineru`: 专门负责复杂 PDF 与图像的深度 OCR 文字识别任务。

### 🚀 快速本地启动 (1-Click Setup)

本项目依赖 GPU 加速，请严格按照以下步骤准备依赖：

1. **准备离线包**: 前往 PyTorch 官网下载匹配您本地 CUDA 版本的 `torch` 和 `torchvision` 的 `.whl` 文件 (对应gemma_4和minerU开头文件夹的README.md中有版本说明)
2. **分发依赖**: 
   - 将 Gemma 环境所需的 `.whl` 放入 `gemma_4_dependencies/` 目录。
   - 将 MinerU 环境所需的 `.whl` 放入 `minerU_dependencies/` 目录。
3. **执行初始化**:
   - **Windows**: 双击运行项目根目录下的 `setup.bat`。
   - **Linux/Mac**: 在终端运行 `bash setup.sh`。
4. 脚本将自动构建双环境并在完成后自动拉起 Web UI。

*(注意：请确保 `config/config.yaml` 中的 `mineru.exe_path` 指向 `venv_mineru/Scripts/magic-pdf.exe`)*

---

## ⚙️ 配置说明 (Configuration)

### 1. `.env` (本机独有环境)
请复制 `.env.example` 并重命名为 `.env`。在此配置：
- **API 密钥**: 联网检索 API Keys（不填则自动降级至 SearXNG）。
- **网络代理**: 配置 `PROXY_URL` 与 `NO_PROXY`。
- **运行端口**: `STREAMLIT_SERVER_PORT`（默认 8501）。

### 2. `config/config.yaml` (全局业务策略)
- **LLM 参数**: 上下文长度 (`n_ctx`)、温度 (`temperature`) 与量化模型路径。
- **数据库**: Milvus 向量数据库连接 URI。
- **RAG 策略**: 检索 Top-K、重排阈值与分块逻辑。

---

## 🐳 Docker 基础设施部署 (Docker Infrastructure)

我们在 `docker_yaml` 目录下提供了模块化的容器编排方案，用于一键部署基础设施：

### 1. 向量数据库 (Milvus)
负责处理高维向量存储与检索。
```bash```
cd docker_yaml/milvus_docker
docker-compose up -d

### 2. 联网检索获取url的工具 (SearXNG)
```bash
cd docker_yaml/serxng_docker
docker-compose up -d
