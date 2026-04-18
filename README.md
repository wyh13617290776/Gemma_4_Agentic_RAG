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

## 🖥️ Windows 前置环境准备 (Windows Prerequisites)

对于 Windows 系统的用户，为了确保 GPU 加速正常运行以及后续 Docker 容器的顺利部署，在开始一切操作之前，请务必确认已安装以下基础运行环境：

1. **CUDA Toolkit (GPU 加速核心)**:
   - 请先打开终端 (CMD/PowerShell) 输入 `nvidia-smi` 查看您的显卡支持的最高 CUDA 版本（例如我的电脑右上角显示的就是 `CUDA Version: 12.6`）。
   - 前往 NVIDIA 官网下载对应的安装包（例如对应 12.6 版本下载 `cuda_12.6.0_560.76_windows.exe` [下载](https://developer.nvidia.com/cuda-12-6-0-download-archive)）并完成默认安装。
2. **WSL2 与 硬件虚拟化 (Hardware Virtualization)**:
   - **状态自查**：右键任务栏 -> 任务管理器 -> 性能 -> CPU，查看右下角是否显示 **“虚拟化：已启用”**。
   - **开启方式**：若显示“已禁用”，需重启电脑并进入 BIOS（通常开机按 `F2`、`Del` 或 `F12`）：
     - **Intel 平台**：找到 `Intel Virtualization Technology` 或 `VT-x`，设为 `Enabled`。
     - **AMD 平台**：找到 `SVM Mode` 或 `Secure Virtual Machine`，设为 `Enabled`。
   - **系统安装**：在管理员权限终端运行 `wsl --install`。安装后必须**重启电脑**。
   - **疑难排解**：如遇 `0x800701bc` 等报错，请手动安装内核更新包（例如 `wsl.2.6.3.0.x64.msi` [下载](https://github.com/microsoft/WSL/releases/tag/2.6.3)）。
3. **Docker Desktop**:
   - 负责运行后续的 Milvus 向量数据库和 SearXNG 检索工具。
   - 下载并安装 `Docker Desktop for Windows-x86_64` [下载](https://docs.docker.com/desktop/setup/install/windows-install/)，安装完成后请在软件设置 (Settings) 中确认已勾选 "Use the WSL 2 based engine" (Windows11 家庭版会默认勾选)。

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

本系统采用模块化配置架构，所有核心敏感信息与业务策略均收拢于 `.env` 及 `config/` 目录下的三大 YAML 文件中：

### 1. `.env` (本机独有环境与密钥)
请复制 `.env.example` 并重命名为 `.env`。在此配置：
- **云端大模型 API Keys**: 配置 `SILICONFLOW_API_KEY` (DeepSeek, [点击注册试用](https://cloud.siliconflow.cn/)) 与 `DASHSCOPE_API_KEY` (Qwen, [点击注册试用](https://bailian.console.aliyun.com/))。
- **联网检索 API Keys**: 配置 `SERPER_API_KEY` ([点击注册试用](https://serper.dev/)), `TAVILY_API_KEY` ([点击注册试用](https://tavily.com/)), `EXA_API_KEY` ([点击注册试用](https://exa.ai/))（不填则系统自动降级使用本地 SearXNG + Crawl4AI）。
- **网络代理**: `PROXY_URL` 与本地白名单 `NO_PROXY` (必须包含 localhost 和 127.0.0.1 避免本地服务被拦截，注：开个VPN使用规则模式)。
- **运行端口**: `STREAMLIT_SERVER_PORT`（默认 8501）。

### 2. `config/config.yaml` (全局基础设施与业务策略)
- **底层引擎与物理路径**: llama-server 路径、MinerU (magic-pdf) 引擎路径、BGE Embedding 及 Reranker 权重模型路径。
- **基础设施**: Milvus 向量数据库的 URI 及维度 (`dim`) 配置。
- **RAG 引擎策略**: BM25 本地节点路径、双路召回规模 (`fusion_top_k`)、BGE 二次精排规模 (`rerank_top_k`) 以及最终注入上下文的片段数 (`similarity_top_k`)。
- **检索引擎与反思机制**: SearXNG 本地地址、多阶梯搜索引擎抓取上限，以及反思引擎的质量过滤阈值（最小有效链接数、文本长度、关键词覆盖率）。
- **记忆流控**: 短期记忆保留窗口 (`max_window`) 与后台静默摘要的触发阈值 (`summary_threshold`)。

### 3. `config/model_router.yaml` (大模型路由与端点映射)
- **默认选择**: 定义系统启动时的默认模型 (`default_model`)。
- **异构资源池定义**: 基于 YAML 锚点模板，统一配置本地 Gemma、硅基流动 (DeepSeek 家族) 及阿里云 (Qwen 纯文本/视觉/Flash极速池) 的端点信息 (`base_url`, `protocol`)。
- **能力探针与适配器**: 为模型分配 `adapter`（标准或思考型）、标记是否支持深度思考 (`dynamic_thinking`) 及多模态特性声明。
- **专属覆写**: 为特定模型定义专属的 `generation_cfg`，覆盖全局参数。

### 4. `config/llm_tasks.yaml` (系统级 Agent 任务采样模板)
- **LLM 对话基准模板**: 按派系（A类 Gemma、B类 Qwen、C类 DeepSeek）定义基础对话的推荐采样参数（`temperature`, `top_p`, `max_tokens`, `top_k`）。
- **内部神经中枢调优**: 为系统后台运行的 Agent 任务（如 `intent_routing` 意图分发、`query_transformation` 复杂查询提纯、`memory_summary` 长期记忆压缩）配置低发散度的专属严谨参数，确保逻辑执行的确定性。

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
负责处理无追踪的本地联网搜索服务。
```bash
cd docker_yaml/serxng_docker
docker-compose up -d
```

> **⚠️ SearXNG 核心配置修改说明**:
> 在通过 `docker-compose up -d` 首次启动 SearXNG 后，系统会在同级目录下生成一个 `searxng` 文件夹。为了确保 Agent 能够通过 API 获取结构化的搜索数据，请务必执行以下步骤：
> 
> 1. 编辑配置文件 `docker_yaml/serxng_docker/searxng/settings.yml`。
> 2. 找到 `search` 节点下的 `formats` 配置，并在其中手动加上 `- json`：
>    ```yaml
>    search:
>      formats:
>        - html
>        - json  # 👈 必须加上这一行
>    ```
> 3. **执行重启命令使配置生效**:
>    ```bash
>    docker-compose restart searxng
>    ```

---

## 📧 联系与交流 (Contact)

如果你对本项目有任何疑问、发现 Bug，或者对 Agentic RAG、多模态大模型落地有更好的想法，欢迎通过以下方式与我交流：

- **Email**: [wyh37133@gmail.com](wyh37133@gmail.com)
- **GitHub**: [wyh13617290776](https://github.com/wyh13617290776)
- **技术探讨**: 欢迎在项目的 [Issues](https://github.com/wyh13617290776/Gemma_4_Agentic_RAG/issues) 提出你的宝贵意见！

---

## 📄 开源协议 (License)

本项目采用 [MIT License](LICENSE) 开源协议。
你可以自由地使用、修改和分发本项目的代码，但请保留原作者的版权声明。
