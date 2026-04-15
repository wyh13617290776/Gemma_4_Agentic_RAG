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

## 📦 模型与核心引擎准备 (Models & Core Engines)

在启动环境部署之前，请务必完成以下核心组件的下载与放置：

1. **llama.cpp 引擎**:
   - 请前往官方仓库下载适合您本地硬件配置的 `llama.cpp`， [下载llama](https://github.com/ggml-org/llama.cpp/releases?q=b8708&expanded=true) ，Windows环境推荐下载 `Windows x64 (Vulkan)` 版本。
   - 将解压后的核心可执行文件（如 `llama-server.exe` ）和相关的 `ddl` 文件的文件夹 (如 `llama/llama-b8708` 包含 `*.exe` 和 `*.ddl` ) 直接放置在 **项目根目录** 下。

2. **核心模型权重 (Models)**:
   - LLM模型及组件 **gemma 4** 模型和多模态组件 **mmproj**，此处仅提供一个我自己的下载组合参考：[gemma-4-E4B-it-Q4_K_M](https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF/blob/main/gemma-4-E4B-it-Q4_K_M.gguf) 和 [mmproj-F16](https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF/blob/main/mmproj-F16.gguf)，放在项目根目录新建的 `models` 文件夹中。
   - 系统依赖稠密向量检索模型 **bge-m3** [下载bge-m3](https://huggingface.co/BAAI/bge-m3/tree/main) 
   - **⚠️注意**: 由于 Pytorch 版本支持问题，需将 `bge-m3` 中的 `pytorch_model.bin` 文件替换成：`model.safetensors` [下载model.safetensors](https://huggingface.co/trollathon/bge-m3-safetensors/blob/main/model.safetensors) 
   - 交叉重排模型 **bge-ranker** [下载bge-ranker](https://huggingface.co/BAAI/bge-reranker-v2-m3/tree/main)
   - 请自行下载并存放在项目根目录新建的 `models` 文件夹中。

   - **MinerU 多模态解析模型自动下载**: 
     一键跨平台下载脚本。
   - **操作步骤**: 直接在项目根目录下运行（无需手动激活虚拟环境）：
     ```bash
     python download_minerU.py
     ```
   - **自动化逻辑**: 
     - **自动识别**: 脚本会自动检测并进入根目录下的 `venv_gemma` 虚拟环境。
     - **自愈检查**: 自动校验 `torch` 等 GPU 加速库是否正确安装，自动安装相关依赖包。
     - **加速下载**: 自动调用 **清华镜像源 (TUNA)** 极速安装下载器，模型将下载至 `models/minerU` 目录下。
   - **⚠️ 注意**: 运行前请**关闭全局代理/梯子**，否则 ModelScope 可能会因网络策略导致下载失败。

---

## 🐍 环境隔离部署 (Dual-Environment Isolation)

为了彻底解决 **MinerU (视觉解析)** 与 **Gemma 4 (业务逻辑)** 之间复杂的底层依赖冲突，本项目采用了**双虚拟环境并行架构**：
1. `venv_gemma`: 承载主 UI (Streamlit)、RAG 引擎、意图路由器及所有 Web 业务。
2. `venv_mineru`: 专门负责复杂 PDF 与图像的深度 OCR 文字识别任务。

### 🚀 快速本地启动 (1-Click Setup)

本项目依赖 GPU 加速，请严格按照以下步骤准备依赖：

1. **准备离线包**: 前往 PyTorch 官网下载匹配您本地 CUDA 版本的 `torch` 和 `torchvision` 的 `.whl` 文件 (对应文件夹的README.md中有版本说明)
2. **分发依赖**: 
   - 将 Gemma 环境所需的 `.whl` 放入 `gemma_4_dependencies/` 目录。
   - 将 MinerU 环境所需的 `.whl` 放入 `minerU_dependencies/` 目录。
3. **执行初始化**:
   - **Windows**: 双击运行项目根目录下的 `setup.bat`。
   - **Linux/Mac**: 在终端运行 `bash setup.sh`。
4. 脚本将自动构建双环境并在完成后自动拉起 Web UI。

*(注意：请确保 `config/config.yaml` 中的 `mineru.exe_path` 指向 `venv_mineru/Scripts/magic-pdf.exe`)*

### ⚙️ MinerU 引擎配置 (MinerU Configuration)

在完成 `venv_mineru` 环境初始化后，必须配置 MinerU 的模型路径与硬件加速开关。
请在你的用户根目录下找到（或新建）配置文件 `C:\Users\[用户名]\magic-pdf.json`（Linux/Mac 为 `~/.magic-pdf.json`），并参考你的项目绝对路径进行如下修改：

Windows 环境下的配置：
```json
{
  "models-dir": "F:/Code_Programming/Gemma/Gemma_Agent_Project/models/minerU/models",
  "device-mode": "cuda",
  "layout-config": {
    "model": "doclayout_yolo"
  },
  "layoutreader-model-dir": "F:/Code_Programming/Gemma/Gemma_Agent_Project/models/minerU/models/ReadingOrder/layout_reader",
  "table-config": {
    "model": "rapid_table",
    "enable": true,
    "max_table_shape": 2048,
    "is_table_recog_enable": true,
    "max_time": 600
  }
}
```
*(注意：请确保上方 JSON 中的 `models-dir` 和 `layoutreader-model-dir` 路径与你本机实际的模型存放绝对路径保持一致)*

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

## 🐳 Docker 部署 (Docker Infrastructure)

我们在 `docker_yaml` 目录下提供了模块化的容器编排方案，用于一键部署基础设施：

### 1、向量数据库 (Milvus) 
负责处理高维向量存储与检索。
```bash
cd docker_yaml/milvus_docker
docker-compose up -d
```

### 2、url获取工具 (SearXNG)
联网检索时第一步获取url的工具。
```bash
cd docker_yaml/serxng_docker
docker-compose up -d
```

> **⚠️ SearXNG 核心配置修改说明**:
> 在通过 `docker-compose up -d` 首次启动 SearXNG 后，系统会在同级目录下生成一个 `searxng` 文件夹。为了确保 Agent 能够通过 API 获取结构化的搜索数据，请打开并编辑 `docker_yaml/serxng_docker/searxng/settings.yml` 文件。
> 找到 `search` 节点下的 `formats` 配置，并在其中手动加上 `- json`：
> ```yaml
> search:
>   formats:
>     - html
>     - json  # 👈 必须加上这一行
> ```
> 保存文件后，请重启 SearXNG 容器使配置生效。
> 重启命令：
> ```bash cd docker_yaml/serxng_docker docker-compose restart searxng```

---

## 📧 联系与交流 (Contact)

如果你对本项目有任何疑问、发现 Bug，或者对 Agentic RAG、多模态大模型落地有更好的想法，欢迎通过以下方式与我交流：

- **Email**: [wyh37133@gmail.com]
- **GitHub**: [wyh13617290776](https://github.com/wyh13617290776)
- **技术探讨**: 欢迎在项目的 [Issues](https://github.com/wyh13617290776/Gemma_4_Agentic_RAG/issues) 提出你的宝贵意见！

---

## 📄 开源协议 (License)

本项目采用 [MIT License](LICENSE) 开源协议。
你可以自由地使用、修改和分发本项目的代码，但请保留原作者的版权声明。
