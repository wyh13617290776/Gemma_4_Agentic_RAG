"""
=============================================================================
核心入口文件：Gemma 4 商业级智能知识库 Agent (Web UI)
功能：配置中心化管理、模型专属参数自适应、RAG全局调优、多模态流式调度
=============================================================================
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import logging
import os
# 核心魔法：强行劫持该进程内所有的 Hugging Face 请求到国内高速镜像站
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import base64
import yaml
import threading
import time
import math
import httpx
import asyncio
from dotenv import load_dotenv

from streamlit.runtime.scriptrunner import add_script_run_ctx
from audio_recorder_streamlit import audio_recorder
from pymilvus import connections

# 优化 PyTorch 显存分配策略，防止 OOM
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ==========================================
# 👑 导入内部核心业务组件 (统一配置中心)
# ==========================================
from core.config import (
    CFG, ROUTER, TASKS, PROMPTS, ROUTER_PATH,
    get_model_generation_params, 
    save_sys_config, 
    save_model_override
)
from core.llm_gateway import LLMGateway
from core.hardware import HardwareManager, is_port_in_use
from core.database import EmbeddingService
from tools.doc_parser import process_and_embed_documents
from memory.chat_memory import MemoryManager
from agents.router import IntentRouter
from core.rag_engine import RAGPipeline
from tools.web_retriever import UltimateWebRetriever
from agents.orchestrator import GraphOrchestrator
from core.multimodal_engine import MultimodalEngine

# 加载环境变量
load_dotenv()

# ==========================================
# 全局日志中心 (Logger) 初始化
# ==========================================
# 配置标准日志格式
logging.basicConfig(
    level=logging.INFO,  # 设定基础日志级别
    format="%(asctime)s - 🚀 %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(), # 默认输出到控制台
        # logging.FileHandler("agent_system.log", encoding="utf-8") # 取消注释此行，即可自动把日志写入文件！
    ]
)
logger = logging.getLogger("AgenticRAG")
logger.info("系统日志模块初始化完毕。")

# ==========================================
# 全局算力锁与缓存单例
# ==========================================
@st.cache_resource
def get_llm_lock():
    """获取全局算力锁，防止并发对话打满 GPU 显存"""
    return threading.Lock()
llm_lock = get_llm_lock()

@st.cache_resource
def get_web_retriever():
    """全局缓存联网检索器，避免反复初始化浏览器内核"""
    return UltimateWebRetriever()
web_retriever = get_web_retriever()

# =========================================================
# 资源生命周期看门狗 (后台静默运行)
# =========================================================
@st.fragment(run_every=60) # 每 60 秒检查一次
def memory_lifecycle_watchdog():
    # 设定空闲回收阈值，例如 5 分钟 (300秒)
    IDLE_THRESHOLD = 300 
    if EmbeddingService._is_loaded:
        idle_duration = EmbeddingService.get_idle_time()
        if idle_duration > IDLE_THRESHOLD:
            EmbeddingService.unload()
            st.toast("💡 系统进入节能模式：长时间未操作，向量引擎显存已自动释放。", icon="♻️")
# 在页面顶部调用，让它启动
memory_lifecycle_watchdog()

# ==========================================
# 页面基础配置与 CSS 注入
# ==========================================
st.set_page_config(page_title="商业级 Agentic RAG", page_icon="🪐", layout="wide", initial_sidebar_state="expanded")

def local_css(file_name):
    if os.path.exists(file_name):
        with open(file_name, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        st.error(f"⚠️ 未找到 CSS 样式文件: {file_name}")

local_css(os.path.join("assets", "css", "style.css"))

# ==========================================
# 全局状态初始化 (UI 状态机)
# ==========================================
if "uploader_key" not in st.session_state: 
    st.session_state.uploader_key = 1
if "enable_web_search" not in st.session_state: 
    st.session_state.enable_web_search = True 
if "last_audio_hash" not in st.session_state:
    st.session_state.last_audio_hash = None

st.title("🪐 Agentic RAG")

# 加上 @st.dialog 装饰器，它就会变成一个带有灰色背景遮罩的中央弹窗
@st.dialog("🛡️ 本地模型环境深度安检", width="large")
def validate_local_environment_modal(server_info, cfg):
    """屏幕中央弹出的安检模态框"""
    missing_deps = []
    
    # 进度提示
    status_text = st.empty()
    progress_bar = st.progress(0)
    
    # --- 步骤 1: 检查模型权重与多模态组件 ---
    status_text.info("📦 正在核验模型权重及多模态组件...")
    time.sleep(0.5) 
    progress_bar.progress(20)
    
    # 1.1 检查基础大语言模型权重 (.gguf)
    model_path = server_info.get("model_path", "")
    if not model_path or not os.path.exists(model_path):
        missing_deps.append("基础模型权重 (.gguf)")
    
    # 1.2 检查多模态视觉权重 (mmproj-*.gguf)
    mmproj_path = server_info.get("mmproj_path", "")
    if not mmproj_path or not os.path.exists(mmproj_path):
        missing_deps.append("多模态视觉权重 (mmproj-*.gguf)")
    
    # --- 步骤 2: 探测推理引擎 ---
    status_text.info("⚙️ 正在递归扫描项目目录寻找 llama-server.exe...")
    progress_bar.progress(50)
    
    search_root = os.getcwd() 
    found_exe = False
    found_dlls = False
    
    exe_path_from_cfg = cfg.get("llm_server", {}).get("server_exe_path", "")
    if exe_path_from_cfg and os.path.exists(exe_path_from_cfg):
        found_exe = True
        target_dir = os.path.dirname(exe_path_from_cfg)
        if any(f.lower().endswith(('.dll')) for f in os.listdir(target_dir)):
            found_dlls = True
    
    if not found_exe:
        for root, dirs, files in os.walk(search_root):
            if "llama" in root.lower():
                if "llama-server.exe" in files:
                    found_exe = True
                    if any(f.lower().endswith(('.dll')) for f in files):
                        found_dlls = True
                    break
    
    progress_bar.progress(90)
    
    if not found_exe:
        missing_deps.append("推理核心引擎 (缺少 llama-server.exe)")
    elif not found_dlls:
        missing_deps.append("引擎运行依赖 (缺失 *.dll 库文件)")

    progress_bar.progress(100)
    
    # --- 步骤 3: 渲染最终结果并控制状态 ---
    if len(missing_deps) == 0:
        status_text.success("✅ 本地模型环境安检通过！引擎准备就绪。")
        # 记录通过状态
        st.session_state["env_check_passed"] = True
        # 自动重启本地模型服务
        # 去掉原先的手动按钮，给用户 1.5 秒的视觉确认时间，然后自动重载整个页面关闭弹窗
        time.sleep(1.5)
        st.rerun()
    else:
        status_text.error("❌ 本地模型缺失组件，无法启动！")
        st.error(f"缺失清单：{', '.join(missing_deps)}")
        st.info("💡 建议：请关闭弹窗，在上方下拉列表中切换至【云端模型】体验。")
        st.session_state["env_check_passed"] = False

# =========================================================
# 模块一：定时刷新探针 (仅用于本地硬件监控)
# =========================================================
@st.fragment(run_every=5)
def render_local_monitor(port):
    if is_port_in_use(port):
        st.success("💻 运行模式: 本地私有化计算 (服务在线 🟢)")
        st.markdown("#### 📊 硬件资源实时监控")
        metrics = HardwareManager.get_system_metrics()
        col1, col2, col3 = st.columns(3)
        col1.metric("CPU 占用", f"{metrics['cpu_percent']:.1f}%")
        col2.metric("系统内存", f"{metrics['ram_percent']:.1f}%")
        if metrics["has_gpu"]:
            col3.metric(
                label=f"显存 (使用: {metrics['vram_used_gb']:.1f}G)", 
                value=f"{metrics['vram_percent']:.1f}%",
                delta=f"核心负载: {metrics['gpu_util']}%", 
                delta_color="off",
                help="💡 **核心负载 (GPU Utilization)**\n\n表示当前显卡计算核心的忙碌程度：\n\n- **低负载 (0%~10%)**：模型处于待机状态（仅占用显存，未进行计算）。\n- **高负载 (80%~100%)**：引擎正在全速推理生成 Token 或进行多模态降维处理。\n\n*注：显存占用高但核心负载低属于正常待机现象。*"
            )
    else:
        st.error("⚠️ 本地服务已离线 (被手动关闭或崩溃)")
        if st.button("手动拉起本地模型服务", use_container_width=True):
            st.session_state['llm_started_this_session'] = False
            st.rerun()

# 云端 API 连通性探活
@st.cache_data(ttl=30, show_spinner=False)
def check_cloud_api_health(protocol, base_url, key_env):
    api_key = os.getenv(key_env, "")
    if not api_key or api_key == "sk-dummy" or len(api_key) < 10:
        return "🔴 API Key 未配置或无效"
    
    try:
        if protocol == "openai":
            # =========================================================
            # 针对阿里云，完全模拟官网 SDK 的 POST 调用行为
            # =========================================================
            if base_url and "dashscope.aliyuncs.com" in base_url:
                test_url = f"{base_url.rstrip('/')}/chat/completions"
                payload = {
                    "model": "qwen-plus",
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 1
                }
                res = httpx.post(
                    test_url,
                    headers={"Authorization": f"Bearer {api_key}"}, 
                    json=payload
                )
            # =========================================================
            # SiliconFlow：GET 探测
            # =========================================================
            else:
                url = f"{base_url.rstrip('/')}/models" if base_url else "https://api.openai.com/v1/models"
                res = httpx.get(url, headers={"Authorization": f"Bearer {api_key}"}, timeout=5.0)
            
            # =========================================================
            # 统一状态码精细化判定
            # =========================================================
            if res.status_code == 200:
                return "🟢 在线 (验证通过)"
            elif res.status_code == 401:
                return "🔴 鉴权失败 (Key 错误)"
            elif res.status_code == 400:
                return "🟡 参数错误 (HTTP 400)"
            elif res.status_code == 429:
                return "🟡 触发限流 (频率过高)"
            elif res.status_code >= 500:
                return f"🔴 云端异常 (HTTP {res.status_code})"
            else:
                return f"🟡 状态异常 (HTTP {res.status_code})"

    except httpx.ConnectTimeout:
        return "🔴 连接超时 (请检查网络)"
    except httpx.ConnectError:
        return "🔴 网络不可达 (请检查代理或域名)"
    except Exception as e:
        return f"🔴 探测异常 ({str(e)[:20]}...)"
        
    return "⚪ 状态未知"

# =========================================================
# 模块二：主调度台与模型自动联动引擎
# =========================================================
def render_dashboard():
    # 1. 获取原始模型列表
    raw_models = list(ROUTER.get("models", {}).keys())
    default_model = ROUTER.get("default_model", "gemma-4-local")
    
    # 2. 获取当前应该处于激活状态的模型
    current_active = st.session_state.get("STREAMLIT_ACTIVE_MODEL", default_model)

    # ============================================================
    # 动态置顶策略 (Dynamic Top-Pinning)
    # 将当前选中的模型强行抽取并插入到列表的最顶部
    # 效果：打开下拉框时永远不需要长跨度滚动，彻底粉碎底层 DOM 卸载的白屏 Bug
    # ============================================================
    if current_active in raw_models:
        raw_models.remove(current_active)
        raw_models.insert(0, current_active)

    if "STREAMLIT_ACTIVE_MODEL" not in st.session_state:
        st.session_state["STREAMLIT_ACTIVE_MODEL"] = default_model
    
    # 图标格式化钩子
    def format_model_label(model_id):
        if not model_id: return "⚠️ 未知模型"
        is_local = "127.0.0.1" in ROUTER.get("models", {}).get(model_id, {}).get("base_url", "")
        return f"💻 {model_id}" if is_local else f"☁️ {model_id}"
    
    # 核心下拉框直接在页面顶部渲染
    selected_model = st.selectbox(
        "当前激活的大模型：", 
        options=raw_models,               # 传入经过动态置顶处理的列表
        format_func=format_model_label,
        key="STREAMLIT_ACTIVE_MODEL"
    )

    model_info = ROUTER["models"].get(selected_model, {})
    model_changed = selected_model != st.session_state.get("prev_selected_model", "")

    # --- 核心联动：模型自适应参数与能力判定 ---
    if "last_synced_model" not in st.session_state or st.session_state.last_synced_model != selected_model:
        m_params = get_model_generation_params(selected_model)
        st.session_state["cur_temp"] = float(m_params.get("temperature", 1.0))
        st.session_state["cur_top_p"] = float(m_params.get("top_p", 0.95))
        st.session_state["cur_top_k"] = int(m_params.get("top_k", 64))
        st.session_state["cur_max_tokens"] = int(m_params.get("max_tokens", 4096))
        
        m_features = model_info.get("features", {})
        st.session_state["enable_multimodal"] = m_features.get("enable_multimodal", True)
        st.session_state.last_synced_model = selected_model

    # C. 深度思考能力探针 
    adapter_type = str(model_info.get("adapter", "")).lower().strip()
    supports_thinking = adapter_type in ["deepseek", "local_think"] or bool(model_info.get("dynamic_thinking", False))
    st.session_state["supports_thinking"] = supports_thinking
    
    if not supports_thinking:
        st.session_state["enable_thinking"] = False
    elif model_changed:
        st.session_state["enable_thinking"] = model_info.get("features", {}).get("enable_thinking", True)

    # 运行模式自检 (本地/云端)
    if model_changed:
        st.session_state['llm_started_this_session'] = False 
        st.session_state["prev_selected_model"] = selected_model

    os.environ["STREAMLIT_ACTIVE_MODEL"] = selected_model
    
    server_info = model_info.get("server_info", {})
    port = int(server_info.get("port", 8000))
    is_local_mode = "127.0.0.1" in model_info.get("base_url", "") or "localhost" in model_info.get("base_url", "") or model_info.get("key_env") == "NONE"

    # ==============================================================
    # 折叠面板移到最下面，仅用于包裹硬件状态和探活日志
    # ==============================================================
    with st.expander("⚙️ 核心引擎调度与硬件监控面板", expanded=False):
        if is_local_mode:
            # 节点1：主动检测与弹窗触发
            # 👑 核心关卡：读取安检状态
            is_env_ready = st.session_state.get("env_check_passed", False)
            
            # ==========================================================
            # 🛑 拦截区：未通过安检
            # ==========================================================
            if not is_env_ready:
                st.warning("⚠️ 本地模型环境尚未通过安检，暂时无法使用。")
                st.session_state["disable_chat_input"] = True

                # 自动触发弹窗逻辑
                if not st.session_state.get("auto_check_triggered", False):
                    st.session_state["auto_check_triggered"] = True
                    validate_local_environment_modal(server_info, CFG)
                
                # 手动触发按钮
                if st.button("🛡️ 立即开始环境深度安检", use_container_width=True, type="primary"):
                    validate_local_environment_modal(server_info, CFG)
                
                # 🛑 核心修复：在此处物理退出函数
                # 只要安检没过，下面的 HardwareManager 和 render_local_monitor 永远不会被执行
                return
            
            # ==========================================================
            # 🟢 放行区：安检已通过
            # ==========================================================
            else:
                # 🟢 环境已通过安检，确保输入框解锁，执行原有逻辑
                st.session_state["disable_chat_input"] = False

                # 1. 安全点火逻辑
                is_online = is_port_in_use(port)
                if not is_online and not st.session_state.get('llm_started_this_session', False):
                    with st.spinner("🚀 检测到本地模式已启用，正在自动点火拉起引擎..."):
                        # 获取线程锁
                        with llm_lock: 
                            # 双重检查锁 (Double-Checked Locking)
                            if is_port_in_use(port):
                                st.session_state['llm_started_this_session'] = True
                                st.rerun()
                            else:
                                st.session_state['llm_started_this_session'] = True
                                st.toast("正在启动本地模型服务...", icon="🚀")
                                try:
                                    HardwareManager.start_llm_service()
                                    # 死等逻辑
                                    timeout = 30
                                    while not is_port_in_use(port) and timeout > 0:
                                        time.sleep(1)
                                        timeout -= 1
                                    if timeout > 0:
                                        st.toast("大模型点火成功！", icon="🔥")
                                        st.rerun() 
                                    else:
                                        st.session_state['llm_started_this_session'] = False
                                        st.error("启动超时，请检查控制台。")
                                except Exception as e:
                                    st.session_state['llm_started_this_session'] = False
                                    st.error(f"启动异常: {e}")
                
                # 3. 核心修复：本地监控雷达被严格隔离在放行区内！
                # 这样未安检时就绝对看不到那个带蓝框的“服务已离线”提示了。
                # 调用被精准剥离的 5 秒刷新组件
                render_local_monitor(port)
        
        # ==========================================================
        # ☁️ 云端模式处理区
        # ==========================================================
        else:
            # 只要用户切到云端模型，立刻解除聊天框的锁定状态
            st.session_state["disable_chat_input"] = False

            st.info(f"☁️ 运行模式: 云端 API ({selected_model})")
            if is_port_in_use(port):
                with st.spinner("♻️ 检测到切换至云端，正在强杀本地进程以释放显存..."):
                    HardwareManager.stop_llm_service()
                    st.toast("本地引擎已关闭，显存已释放", icon="♻️")

            st.session_state['llm_started_this_session'] = False
            
            st.markdown("#### 📊 引擎状态")
            # 核心修复：云端连通性缓存机制 (中央控制)
            health_cache_key = f"api_health_{selected_model}"
            
            # 只有当：1. 发生了模型切换  或者 2. 缓存里还没测过这个模型 时，才去真实发 HTTP 请求
            if model_changed or health_cache_key not in st.session_state:
                with st.spinner("📡 正在探测云端连通性..."):
                    status_text = check_cloud_api_health(
                        protocol=model_info.get("protocol"),
                        base_url=model_info.get("base_url"),
                        key_env=model_info.get("key_env")
                    )
                    st.session_state[health_cache_key] = status_text
            
            st.metric("云端连通性", st.session_state[health_cache_key])

# 渲染顶部折叠面板
render_dashboard()

# ==========================================
# 业务组件初始化
# ==========================================
memory = MemoryManager(st.session_state, max_window=CFG["memory"]["max_window"])
router = IntentRouter()
rag_engine = RAGPipeline()
multimodal_engine = MultimodalEngine()
orchestrator = GraphOrchestrator(router, rag_engine, web_retriever)

# ==========================================
# 辅助弹窗
# ==========================================
@st.dialog("🗂️ 上传文件并入库", width="large")
def upload_file_dialog():
    st.markdown("上传文档或代码，系统将自动进行【智能分流解析】并入库。")
    supported_formats = ["pdf", "doc", "docx", "xls", "xlsx", "ppt", "pptx", "txt", "md", "csv", "py", "json"]
    uploaded_files = st.file_uploader("拖拽文件到此处", type=supported_formats, accept_multiple_files=True)
    
    if st.button("🚀 开始解析并入库", use_container_width=True):
        try:
            # 动作前：装载或刷新 Embedding 模型的存活时间
            EmbeddingService.load(device="cuda")
            # 执行解析
            process_and_embed_documents(uploaded_files)
            st.success("解析入库完成！系统已恢复对话模式，您可以关闭此弹窗开始提问。")
        except Exception as e:
            # 失败立即释放显存
            EmbeddingService.unload(reason="immediate")
            st.error(f"入库失败: {e}")


# ==========================================
# 侧边栏：多会话与参数调优面板
# ==========================================
with st.sidebar:
    # 顶部 Logo 与系统名称
    st.image("assets/logo.svg", width=50)
    st.markdown("### 工作台")
    
    # 开启新会话逻辑：清空历史消息和摘要
    if st.button("➕ 开启新会话", use_container_width=True, type="primary"):
        st.session_state.messages = []
        st.session_state.current_summary = ""
        st.rerun()
        
    st.divider()
    
    # --- 控制面板 1：模型专属微调 ---
    active_m = st.session_state.get("STREAMLIT_ACTIVE_MODEL", "gemma-4-local")
    st.markdown(f"#### ⚙️ {active_m} 生成参数")
    
    st.session_state["cur_temp"] = st.slider(
        "模型发散度 (Temperature)", 
        0.0, 2.0, st.session_state.get("cur_temp", 1.0), 0.1,
        help="推荐 1.0。值越高回答越有创意；值越低越严谨。"
    )
    st.session_state["cur_max_tokens"] = st.select_slider(
        "最大输出长度 (Max Tokens)", 
        options=[1024, 2048, 4096, 8192], 
        value=st.session_state.get("cur_max_tokens", 4096)
    )

    with st.expander("🔬 高级采样控制", expanded=False):
        # 保留 Top-P 的帮助按钮
        st.session_state["cur_top_p"] = st.slider(
            "Top-P (核采样)", 
            0.1, 1.0, st.session_state.get("cur_top_p", 0.95), 0.05,
            help="核采样阈值。值越低回答越死板，值越高越有创造性。"
        )
        # 保留 Top-K 的帮助按钮
        st.session_state["cur_top_k"] = st.number_input(
            "Top-K", 
            1, 100, st.session_state.get("cur_top_k", 64), 1,
            help="绝对截断。限制模型每一步只能在概率最高的 K 个词汇中挑选。"
        )

    c1, c2 = st.columns(2)
    with c1:
        if st.button("🔄 加载模板", use_container_width=True, help="载入官方推荐参数"):
            m_type = ROUTER.get("models", {}).get(active_m, {}).get("type", "gemma")
            template = TASKS.get(f"llm_chat_{m_type}", {})
            if template:
                st.session_state["cur_temp"] = float(template.get("temperature", 1.0))
                st.session_state["cur_top_p"] = float(template.get("top_p", 0.95))
                st.session_state["cur_max_tokens"] = int(template.get("max_tokens", 4096))
                if "extra_body" in template: st.session_state["cur_top_k"] = int(template["extra_body"].get("top_k", 64))
                st.toast(f"✅ 已加载 {m_type} 系列推荐模板", icon="🔄")
                st.rerun()

    with c2:
        if st.button("💾 保存配置", use_container_width=True, type="primary"):
            # 仅保存当前模型的专属微调
            c_params = {
                "temperature": st.session_state["cur_temp"],
                "top_p": st.session_state["cur_top_p"],
                "top_k": st.session_state["cur_top_k"],
                "max_tokens": st.session_state["cur_max_tokens"]
            }
            save_model_override(active_m, c_params)
            st.success(f"✅ {active_m} 专属配置已固化")

    st.divider()
    
    # --- 控制面板 2：系统全局基础设施 ---
    st.markdown("#### 🔍 RAG 引擎调优")
    
    # 第一层：检索与重排规模 (并排显示，节省空间)
    col1, col2 = st.columns(2)
    with col1:
        CFG["rag"]["retrieval"]["fusion_top_k"] = st.number_input(
            "初筛召回池 (Fusion)", 
            min_value=10, max_value=100, 
            value=int(CFG["rag"]["retrieval"].get("fusion_top_k", 40)),
            step=5,
            help="双路检索第一轮打捞的片段数量。数值越大，找得越全，但速度变慢。"
        )
    with col2:
        CFG["rag"]["retrieval"]["rerank_top_k"] = st.number_input(
            "最终精排数 (Rerank)", 
            min_value=3, max_value=30, 
            value=int(CFG["rag"]["retrieval"].get("rerank_top_k", 15)),
            step=1,
            help="送往 BGE 模型进行二次排序的片段数。决定了精排的深度。"
        )
    
    # 第二层：RAG 最终注入浓度补齐
    CFG["rag"]["similarity_top_k"] = st.slider(
        "最终注入提示词的片段数（知识提取浓度）", 
        min_value=1, max_value=10, 
        value=int(CFG["rag"].get("similarity_top_k", 5)), 
        step=1,
        help="经过重排后，真正进入提示词的文本块数量。如果模型容易被杂讯干扰，请调低此值；如果回答不全，请调高。"
    )
    
    st.divider()

    st.markdown("#### 🧠 记忆流控阈值")
    
    CFG["memory"]["max_window"] = st.slider(
        "短期记忆保留轮数", 
        min_value=1, max_value=10, 
        value=int(CFG["memory"].get("max_window", 3)), 
        step=1,
        help="数值越大，模型能记住越久远的上下文，但极其消耗 GPU 显存和推理速度。1 轮 = 1 次问+答。"
    )
    
    CFG["memory"]["summary_threshold"] = st.number_input(
        "触发记忆压缩阈值 (轮)", 
        min_value=3, max_value=20, 
        value=int(CFG["memory"].get("summary_threshold", 5)), 
        step=1,
        help="当对话总轮数达到此阈值时，将在后台启动静默线程，自动把旧对话压缩成精华摘要。"
    )

    st.divider()
    st.markdown("#### 👁️ 多模态特性")
    st.session_state["enable_multimodal"] = st.toggle(
        "开启图像/音视频解析", 
        value=st.session_state.get("enable_multimodal", True),
        help="关闭后，系统将拒绝解析所有附件，可节省显存并加快推理。"
    )
    
    st.divider()

    # 此按钮专门保存 RAG 和 记忆 参数
    if st.button("💾 保存RAG和Memory参数", use_container_width=True):
        try:
            save_sys_config(CFG)
            st.success("✅ RAG 引擎与记忆流控参数已保存至 config.yaml")
        except Exception as e:
            st.error(f"❌ 保存失败: {e}")
            
    # --- 状态雷达 ---
    @st.fragment(run_every=3)
    def render_health_radar():
        st.caption("实时系统状态：")
        active_model = st.session_state.get("STREAMLIT_ACTIVE_MODEL", "gemma-4-local")
        s_info = ROUTER["models"].get(active_model, {}).get("server_info", {})
        port = int(s_info.get("port", 8000))
        llm_host = s_info.get("host", "127.0.0.1")

        @st.cache_data(ttl=3, show_spinner=False)
        def check_llm_health(host, p):
            """HTTP 业务层探活：不仅端口要通，服务还必须正常响应 200"""
            try:
                url = f"http://{host}:{p}/v1/models"
                response = httpx.get(url, timeout=0.5)
                return response.status_code == 200
            except Exception:
                return False

        @st.cache_data(ttl=5, show_spinner=False)
        def check_milvus_health(url):
            """Milvus 业务探活"""
            try:
                connections.connect(alias="health_check", uri=url, timeout=0.5)
                connections.disconnect("health_check")
                return True
            except Exception:
                return False
        
        is_local = "127.0.0.1" in ROUTER["models"].get(active_model, {}).get("base_url", "")
        if is_local:
            if check_llm_health(llm_host, port):
                st.success("🟢 本地推理引擎在线")
            else:
                st.error("🔴 本地推理引擎离线")
        else:
            status_text = st.session_state.get(f"api_health_{active_model}", "⚪ 状态未知")
            if "🟢" in status_text:
                st.success(f"🟢 云端在线 ({active_model})")
            else:
                st.warning(f"🔴 云端异常: {status_text.replace('🔴 ', '')}")

        if check_milvus_health(CFG["milvus"]["uri"]):
            st.info("🟢 Milvus 向量库连通")
        else:
            st.error("🔴 Milvus 库失联，请检查 Docker")
    
    render_health_radar()


# ==========================================
# 交互组件区 (上传、思考、联网、附件、语音)
# ==========================================
with st.container(border=True):
    cols = st.columns([1.2, 1.2, 1.2, 1.2, 0.8, 2.7])
    
    with cols[0]:
        if st.button("🗂️ 上传本地资产", use_container_width=True):
            upload_file_dialog() 
            
    with cols[1]:
        # 👑 核心自适应：动态禁用/启用“深度思考”按钮
        can_think = st.session_state.get("supports_thinking", False)
        btn_type = "primary" if st.session_state.get("enable_thinking", False) else "secondary"
        help_text = "开启/关闭深度思考过程展示" if can_think else "⚠️ 当前选择的大模型不支持深度思考"
        
        if st.button("🧠 深度思考", type=btn_type, use_container_width=True, disabled=not can_think, help=help_text):
            st.session_state.enable_thinking = not st.session_state.enable_thinking
            st.rerun()
            
    with cols[2]:
        web_btn_type = "primary" if st.session_state.enable_web_search else "secondary"
        if st.button("🌐 联网搜索", type=web_btn_type, use_container_width=True):
            st.session_state.enable_web_search = not st.session_state.enable_web_search
            st.rerun()
            
    with cols[3]:
        # 附件弹窗：支持图/文/音/视
        with st.popover("📎 附加文件", use_container_width=True):
            chat_media = st.file_uploader(
                "支持多模态", 
                type=["png", "jpg", "jpeg", "webp", "wav", "mp4"], 
                label_visibility="collapsed", 
                key=f"media_uploader_{st.session_state.uploader_key}"
            )
            if chat_media:
                st.success(f"已附加: {chat_media.name}")
                
    with cols[4]:
        # Level 3：录音麦克风集成
        audio_bytes = audio_recorder(text="", icon_size="2x", icon_name="microphone", key="voice_recorder")
        
    with cols[5]:
        st.markdown("<div style='margin-top: 10px; color: #808495; font-size: 0.85em;'>💡 提示：点击左侧麦克风可直接发送语音指令。</div>", unsafe_allow_html=True)

st.divider() 

# ====================================================
# 历史消息渲染引擎
# ====================================================
for message in memory.get_ui_messages():
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            # 渲染用户端多模态内容
            if isinstance(message["content"], list):
                for item in message["content"]:
                    if item["type"] == "text":
                        st.markdown(item["text"])
                    elif item["type"] == "image_url":
                        st.image(item["image_url"]["url"], width=250)
                    elif item["type"] == "audio":
                        st.audio(item["audio"])
                    elif item["type"] == "video":
                        st.video(item["video"])
            else:
                st.markdown(message["content"])
        else:
            # 渲染助手端的思考过程与正文
            thought = message.get("thought", "")
            content = message.get("content", "")
            if thought.strip():
                with st.expander("🧐 思考过程"):
                    st.markdown(f"<div style='color:gray; font-size:0.85em;'>{thought}</div>", unsafe_allow_html=True)
            if content.strip():
                st.markdown(content)
                
                # 拦截 JSON 数据渲染 Echarts/Plotly 的钩子预留
                if "```json\n{" in content and "bar_chart" in content.lower():
                    # 未来可以在这里接入 JSON 转换并用 st.plotly_chart 渲染
                    pass

# ====================================================
# 核心对话与处理触发器
# ====================================================
# 节点 2：UI 视觉锁定
# 从全局状态机获取锁定标识
is_locked = st.session_state.get("disable_chat_input", False)
placeholder = "⚠️ 请先完成本地模型安检或切换云端模型" if is_locked else "请向您的知识库提问："
user_prompt = st.chat_input(placeholder, disabled=is_locked)

# 检查是否触发了新的语音录制 (防重复哈希校验)
voice_triggered = False
if audio_bytes is not None:
    current_audio_hash = hash(audio_bytes)
    if current_audio_hash != st.session_state.last_audio_hash:
        voice_triggered = True
        st.session_state.last_audio_hash = current_audio_hash

# 当用户敲击回车 或 录制完新语音时触发全链路
if user_prompt or voice_triggered:
    # 智能唤醒
    # 只要用户开始说话，就确保 Embedding 模型在内存中，如果之前被释放了则会自动重装
    with st.spinner("🧠 正在唤醒本地知识库引擎..."):
        EmbeddingService.load(device="cuda")

    # 节点 3：逻辑终极卫兵
    # 如果安检未通过，强制拦截本次执行，防止代码流向 LLMGateway 导致崩溃
    if st.session_state.get("disable_chat_input", False):
        st.error("操作被拦截：本地模型环境未就绪，请先执行安检或更换模型。")
        st.stop() # 立即停止当前脚本运行

    user_content = []
    has_media = False
    
    # 优先处理刚录制的语音
    if voice_triggered:
        base64_audio = base64.b64encode(audio_bytes).decode('utf-8')
        user_content.append({"type": "audio", "audio": f"data:audio/wav;base64,{base64_audio}"})
        # 如果用户光发语音没打字，给大模型补齐一个默认指令
        if not user_prompt:
            user_prompt = "请仔细听这段录音，并结合我的其他指令进行解答。"
        has_media = True

    # 处理上传的文件附件
    elif chat_media is not None:
        bytes_data = chat_media.getvalue()
        base64_data = base64.b64encode(bytes_data).decode('utf-8')
        file_ext = chat_media.name.split('.')[-1].lower()
        mime_type = chat_media.type
        
        if file_ext in ["png", "jpg", "jpeg", "webp"]:
            user_content.append({"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{base64_data}"}})
        elif file_ext == "wav":
            user_content.append({"type": "audio", "audio": f"data:{mime_type};base64,{base64_data}"})
        elif file_ext == "mp4":
            user_content.append({"type": "video", "video": f"data:{mime_type};base64,{base64_data}"})
        
        has_media = True
        st.session_state.uploader_key += 1
    
    # 组合最终的用户 Payload 存入记忆
    user_content.append({"type": "text", "text": user_prompt})
    memory.add_user_message(user_content)
    
    # 在前端立即渲染用户的输入
    with st.chat_message("user"):
        if chat_media: st.caption(f"📎 附带媒体: {chat_media.name}")
        if voice_triggered: st.caption("🎤 附带实时语音指令")
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        # 提取当前真正是否使用思考的标识
        use_thinking_for_this_turn = st.session_state.get("enable_thinking", False) and st.session_state.get("supports_thinking", False)
        messages_to_send = memory.get_llm_payload(user_content)

        media_context = ""
        if chat_media and st.session_state.get("enable_multimodal", True): 
            with st.spinner("👁️ 正在启动视觉前置信息提取..."):
                media_context = asyncio.run(multimodal_engine.process_files([chat_media]))
                if media_context: st.toast("✅ 视觉特征提取完成！", icon="👁️")
        
        # 将提取的文本与用户原话强行绑定，给状态机注入“上帝视角”
        full_user_query = user_prompt
        if media_context:
            full_user_query = f"{media_context}\n\n【用户针对附件的提问】：{user_prompt}"
        
        # =========================================================
        # 状态机编排接管：路由 -> 提纯 -> 并发检索 -> 提示词拼装
        # =========================================================
        with st.spinner("🧠 战略主脑已接管：正在执行图结构并发调度..."):
            final_state = asyncio.run(orchestrator.run(
                user_query=full_user_query, 
                has_media=has_media, 
                enable_web_search=st.session_state.enable_web_search
            ))
            intents = final_state.intents
            nodes = final_state.rag_nodes
            # 组合所有的系统提示词块
            system_content = "\n\n".join(final_state.system_content_blocks)

        # 组合终极 System Prompt = 所有前置意图上下文 + 长期记忆压缩
        final_system_prompt = system_content.strip() + "\n\n" + memory.get_summary_prompt()
        if final_system_prompt.strip():
            messages_to_send.insert(0, {"role": "system", "content": final_system_prompt})
        
        # 多模态数据全局拦截剥离 (防爆显存机制)
        # 如果当前回合没有任何多模态意图被触发，强制删除 Payload 中的多模态大文件
        multimodal_intents = {"analyze_image", "analyze_audio", "analyze_video"}
        if not any(i in multimodal_intents for i in intents):
            stripped_current_media = False
            for msg in messages_to_send:
                if isinstance(msg["content"], list):
                    original_len = len(msg["content"])
                    text_only = [item for item in msg["content"] if item["type"] == "text"]
                    msg["content"] = text_only
                    if msg is messages_to_send[-1] and original_len > len(text_only):
                        stripped_current_media = True
            if stripped_current_media:
                st.toast("♻️ 媒体附件与当前意图无关，已在底层剥离以节省 GPU 算力", icon="🍃")

        memory.add_assistant_message(thought="", content="")

        # =========================================================
        # 触发大模型生成 (由企业级网关接管，多厂商思维链统一剥离)
        # =========================================================
        with llm_lock:
            chat_gateway = LLMGateway(temperature=st.session_state.get("cur_temp"))
            
            stream_generator = chat_gateway.stream_invoke(
                messages=messages_to_send,
                top_p=st.session_state.get("cur_top_p"),
                max_tokens=st.session_state.get("cur_max_tokens"),
                extra_body={"top_k": st.session_state.get("cur_top_k")},
                request_thinking=use_thinking_for_this_turn
            )
            
            # UI 占位符准备
            think_status = None
            think_placeholder = None
            if use_thinking_for_this_turn:
                think_status = st.status("🧠 深度思考中...", expanded=True)
                think_placeholder = think_status.empty()
                
            answer_placeholder = st.empty()
            raw_thought = ""
            raw_content = ""
            thought_completed = False
            
            # 网关吐出的纯净标准数据
            for chunk_data in stream_generator:
                raw_thought += chunk_data["reasoning"]
                raw_content += chunk_data["content"]
                
                # 渲染逻辑
                if use_thinking_for_this_turn:
                    if raw_content and not thought_completed:
                        think_status.update(label="🧐 推理过程", state="complete", expanded=False)
                        thought_completed = True
                    if raw_thought and not thought_completed:
                        think_placeholder.markdown(f"<span style='color:gray;'>{raw_thought}</span>", unsafe_allow_html=True)
                    if raw_content:
                        answer_placeholder.markdown(raw_content)
                    
                    # 实时写入内存
                    memory.update_last_message(thought=raw_thought, content=raw_content)
                else:
                    answer_placeholder.markdown(raw_content)
                    memory.update_last_message(thought="", content=raw_content)
        
        # ==========================================================
        # 极简溯源 UI 渲染 (仅针对本地 RAG 召回的数据)
        # ==========================================================
        if "search_knowledge_base" in intents and nodes:
            with st.expander("📚 查看本地知识库溯源细节", expanded=False):
                for i, n in enumerate(nodes):
                    meta = n.node.metadata
                    # 概率换算: 使用 Sigmoid 函数将原始 Logit 分数映射到 0~1 的置信度
                    probability = 1 / (1 + math.exp(-n.score))
                    percentage_score = f"{probability * 100:.1f}%"
                    
                    file_type = meta.get("file_type", "txt").lower()
                    page_info = f"第 {meta.get('page_label', '?')} 页" if file_type == "pdf" else ""
                    
                    st.markdown(f"📄 **来源 {i+1}**：{meta.get('file_name', '未知')} ({page_info})")
                    st.markdown(f"🎯 **语义匹配度**：`{percentage_score}`")
                    st.markdown(f"> {meta.get('original_text', n.get_content())}")
                    st.divider()

        # ==============================================================
        # 异步线程：后台静默记忆摘要压缩
        # ==============================================================
        if use_thinking_for_this_turn and not thought_completed:
            memory.update_last_message(thought=raw_thought, content=raw_content)

        if memory.need_summarize():
            full_history_text = ""
            for m in st.session_state.messages[:-1]: 
                role = "用户" if m["role"] == "user" else "助手"
                if m["role"] == "user":
                    if isinstance(m["content"], list):
                        text_content = next((item["text"] for item in m["content"] if item["type"] == "text"), "[附带了多模态文件]")
                    else:
                        text_content = str(m["content"])
                else:
                    text_content = str(m["content"])
                full_history_text += f"{role}: {text_content}\n"
            
            summary_request = [
                {"role": "system", "content": PROMPTS["memory_summary_system"]},
                {"role": "user", "content": PROMPTS["memory_summary_user"].format(
                    old_summary=st.session_state.current_summary,
                    new_dialogue=full_history_text
                )}
            ]
            
            def background_summarize(request_payload, mem_obj):
                """后台独立线程，等待算力锁释放后进行静默摘要"""
                time.sleep(2.0)
                if llm_lock.acquire(blocking=False):
                    try:
                        # 实例化一个专门的后台网关 (上下文压缩)
                        bg_gateway = LLMGateway(task_name="memory_summary")
                        new_summary = bg_gateway.invoke(
                            messages=request_payload, 
                            stream=False
                        )
                        mem_obj.update_summary(new_summary)
                        print("✅ [后台任务] 长期记忆静默更新完成")
                    except Exception as e:
                        print(f"❌ [后台任务] 摘要生成失败: {e}")
                    finally:
                        llm_lock.release()
                else:
                    print("⚠️ [后台避让] 检测到大模型正忙，主动放弃本次摘要，算力让给主对话。")

            summary_thread = threading.Thread(target=background_summarize, args=(summary_request, memory))
            add_script_run_ctx(summary_thread) 
            summary_thread.start()
            st.toast("🔄 触发记忆阈值，将在后台空闲时静默压缩...", icon="🗄️")