"""
=============================================================================
核心入口文件：Gemma 4 商业级智能知识库 Agent (Web UI)
功能：前端渲染、多模态输入交互、意图路由分发、流式对话处理、后台静默记忆摘要
=============================================================================
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import os
# 核心魔法：强行劫持该进程内所有的 Hugging Face 请求到国内高速镜像站
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import base64
import threading
import time
import math
import httpx
from streamlit.runtime.scriptrunner import add_script_run_ctx

# 引入外部强化 UI 库
from audio_recorder_streamlit import audio_recorder
import plotly.graph_objects as go

from dotenv import load_dotenv
from openai import OpenAI

# 优化 PyTorch 显存分配策略，防止 OOM (Out Of Memory) 报错
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ==========================================
# 导入内部核心业务组件
# ==========================================
from core.config import CFG, PROMPTS, save_config
from core.hardware import HardwareManager, is_port_in_use
from core.database import EmbeddingService
from tools.doc_parser import process_and_embed_documents
from memory.chat_memory import MemoryManager
from agents.router import IntentRouter
from core.rag_engine import RAGPipeline
from tools.web_retriever import UltimateWebRetriever

import asyncio
from core.query_transformer import QueryTransformer

from agents.orchestrator import GraphOrchestrator
from core.multimodal_engine import MultimodalEngine

# 加载环境变量并初始化 OpenAI 兼容客户端
load_dotenv()
client = OpenAI(
    api_key="sk-local",
    base_url=f"http://{CFG['llm_server']['host']}:{CFG['llm_server']['port']}/v1", 
    # 新版 httpx 的终极防代理写法：显式清空代理，强行无视操作系统的 Clash 环境变量
    http_client=httpx.Client(proxy=None, trust_env=False) 
)

# 页面基础配置：启用宽屏模式，默认展开侧边栏
st.set_page_config(page_title="Gemma 4 商业级 Agent", page_icon="🪐", layout="wide", initial_sidebar_state="expanded")

def local_css(file_name):
    """
    读取外部 CSS 文件并注入到 Streamlit 页面中
    :param file_name: CSS 文件的路径
    """
    with open(file_name, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# 从外部加载极致美化样式
style_path = os.path.join("assets", "css", "style.css")
if os.path.exists(style_path):
    local_css(style_path)
else:
    st.error(f"⚠️ 未找到 CSS 样式文件: {style_path}")

# ==========================================
# 全局状态与核心单例初始化
# ==========================================
# 文件上传器刷新 Key
if "uploader_key" not in st.session_state: 
    st.session_state.uploader_key = 1
# 深度思考模式状态
if "enable_thinking" not in st.session_state: 
    st.session_state.enable_thinking = CFG["gemma_features"].get("enable_thinking", False)
# 联网搜索模式状态
if "enable_web_search" not in st.session_state: 
    st.session_state.enable_web_search = True 
# 防重复处理录音的标记
if "last_audio_hash" not in st.session_state:
    st.session_state.last_audio_hash = None

# 初始化无状态业务组件
memory = MemoryManager(st.session_state, max_window=CFG["memory"]["max_window"])
router = IntentRouter()
rag_engine = RAGPipeline()

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

multimodal_engine = MultimodalEngine()

orchestrator = GraphOrchestrator(router, rag_engine, web_retriever)

# ==========================================
# 弹窗逻辑：文件上传与解析工作流
# ==========================================
@st.dialog("🗂️ 上传文件并入库", width="large")
def upload_file_dialog():
    """渲染文件上传弹窗，接收文件并交由底层 Parser 处理后存入 Milvus"""
    st.markdown("上传文档或代码，系统将自动进行【智能分流解析】并入库。")
    supported_formats = ["pdf", "doc", "docx", "xls", "xlsx", "ppt", "pptx", "txt", "md", "csv", "py", "json"]
    uploaded_files = st.file_uploader("拖拽文件到此处，或点击上传", type=supported_formats, accept_multiple_files=True)
    
    if st.button("🚀 开始解析并入库", use_container_width=True):
        process_and_embed_documents(uploaded_files)


# ==========================================
# Level 1：侧边栏与多会话管理
# ==========================================
with st.sidebar:
    # 顶部 Logo 与系统名称
    st.image("assets/logo.svg", width=50)
    st.markdown("### Gemma 4 工作台")
    
    # 开启新会话逻辑：清空历史消息和摘要
    if st.button("➕ 开启新会话", use_container_width=True, type="primary"):
        st.session_state.messages = []
        st.session_state.current_summary = ""
        st.rerun()
        
    st.divider()
    
    # ---------------------------------------------------------
    # 控制面板 1：生成参数 (Generation)
    # ---------------------------------------------------------
    st.markdown("#### ⚙️ 大模型生成参数")
    
    # 温度：决定发散程度，Slider 最直观
    CFG["llm_generation"]["temperature"] = st.slider(
        "模型发散度 (Temperature)", 
        min_value=0.0, max_value=2.0, 
        value=float(CFG["llm_generation"].get("temperature", 1.0)), 
        step=0.1,
        help="推荐 1.0。值越高，回答越有创意；值越低，回答越严谨、机械。"
    )
    
    # 最大 Token：阶梯式选择，Select Slider 最合适
    CFG["llm_generation"]["max_tokens"] = st.select_slider(
        "最大输出长度 (Max Tokens)", 
        options=[1024, 2048, 4096, 8192], 
        value=CFG["llm_generation"].get("max_tokens", 4096)
    )

    # 展开的高级生成参数
    with st.expander("🔬 高级采样控制", expanded=False):
        # Top-P 截断
        CFG["llm_generation"]["top_p"] = st.slider(
            "Top-P (核采样)", 
            min_value=0.1, max_value=1.0, 
            value=float(CFG["llm_generation"].get("top_p", 0.95)), 
            step=0.05,
            help="核采样阈值。例如设为 0.8，意味着模型只会从累计概率达到 80% 的候选词中做选择。值越低回答越死板/确定，值越高越有创造性。通常不建议与 Temperature 同时大幅修改。"
        )
        # Top-K：数字输入框更精准
        CFG["llm_generation"]["top_k"] = st.number_input(
            "Top-K", 
            min_value=1, max_value=100, 
            value=int(CFG["llm_generation"].get("top_k", 64)), 
            step=1,
            help="绝对截断。强行限制模型每一步只能在概率最高的 K 个词汇中挑选。把这个值调低（比如 10），可以极其有效地防止模型产生幻觉或胡言乱语。"
        )

    st.divider()
    
    # ---------------------------------------------------------
    # 控制面板 2：RAG 引擎深度调优 (已合并知识浓度控制)
    # ---------------------------------------------------------
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

    # ---------------------------------------------------------
    # 控制面板 4：记忆管家引擎 (Memory)
    # ---------------------------------------------------------
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
    
    # ---------------------------------------------------------
    # 控制面板 3：多模态全局开关
    # ---------------------------------------------------------
    st.markdown("#### 👁️ 多模态特性")
    # 使用 Toggle 开关控制
    CFG["gemma_features"]["enable_multimodal"] = st.toggle(
        "开启图像/音视频解析", 
        value=CFG["gemma_features"].get("enable_multimodal", True),
        help="关闭后，系统将拒绝解析所有附件，可节省显存并加快推理。"
    )
    
    st.divider()
    
    # 💾 固化配置按钮
    if st.button("💾 将当前参数保存为默认", use_container_width=True):
        try:
            save_config(CFG)
            st.success("✅ 配置已永久保存至 config.yaml！")
        except Exception as e:
            st.error(f"❌ 保存失败: {e}")
    
    # ---------------------------------------------------------
    # 实时系统状态探活雷达 (严谨 HTTP 业务监控版)
    # ---------------------------------------------------------
    st.caption("实时系统状态：")
    
    # 核心防御：TTL 缓存。限制每 3 秒最多只真正发一次请求，绝不阻塞 UI
    @st.cache_data(ttl=3, show_spinner=False)
    def check_llm_health(host, port):
        """HTTP 业务层探活：不仅端口要通，服务还必须正常响应 200"""
        try:
            # 访问 OpenAI 兼容接口的 /v1/models 端点
            url = f"http://{host}:{port}/v1/models"
            # timeout=0.5 秒，足够本地或局域网响应了
            response = httpx.get(url, timeout=0.5)
            return response.status_code == 200
        except Exception:
            return False

    @st.cache_data(ttl=5, show_spinner=False)
    def check_milvus_health(url):
        """Milvus 业务探活"""
        try:
            from pymilvus import connections
            # 尝试建立短链接测试，如果能连上说明服务健康
            connections.connect(alias="health_check", uri=url, timeout=0.5)
            connections.disconnect("health_check")
            return True
        except Exception:
            return False

    # 1. 验证大模型
    llm_host = CFG['llm_server']['host']
    llm_port = CFG['llm_server']['port']
    if check_llm_health(llm_host, llm_port):
        st.success("🟢 推理引擎在线")
    else:
        st.error("🔴 推理引擎异常，请检查控制台日志！")
        
    # 2. 验证 Milvus
    if check_milvus_health(CFG["milvus"]["uri"]):
        st.info("🟢 Milvus 向量库连通")
    else:
        st.error("🔴 Milvus 库失联，请检查 Docker 服务！")

# ==========================================
# 主界面：环境自检与启动
# ==========================================
if "system_initialized" not in st.session_state:
    with st.spinner("系统初始化中... 正在装载底层环境 (初次启动可能需要十几秒)..."):
        with llm_lock:
            llm_port = HardwareManager.PORT
            # 检测本地大模型服务端口是否存活
            if not is_port_in_use(llm_port):
                st.toast("检测到大模型未启动，正在自动点火拉起引擎...", icon="🚀")
                HardwareManager.start_llm_service()
                
                # 死等大模型真正绑定端口，最多等待 30 秒
                timeout = 30
                while not is_port_in_use(llm_port) and timeout > 0:
                    time.sleep(1)
                    timeout -= 1
                    
                if timeout <= 0:
                    st.error("❌ 大模型拉起超时，请检查控制台日志！")
                else:
                    st.toast("大模型点火成功！", icon="🔥")
            else:
                st.toast("大模型引擎已在线，系统就绪！", icon="✅")
                
            # 拉起 BGE 向量化模型
            EmbeddingService.load(device="cuda")
        st.session_state.system_initialized = True

st.title("🪐 Gemma 4 E4B")

# ==========================================
# 交互组件区 (上传、思考、联网、附件、语音)
# ==========================================
with st.container(border=True):
    # 调整为 6 列，完美塞入语音组件
    cols = st.columns([1.2, 1.2, 1.2, 1.2, 0.8, 2.7])
    
    with cols[0]:
        if st.button("🗂️ 上传本地资产", use_container_width=True):
            upload_file_dialog() 
            
    with cols[1]:
        btn_type = "primary" if st.session_state.enable_thinking else "secondary"
        if st.button("🧠 深度思考", type=btn_type, use_container_width=True):
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
        # 提示文案垂直居中对齐
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
                
                # Level 2: 拦截 JSON 数据渲染 Echarts/Plotly 的钩子预留
                if "```json\n{" in content and "bar_chart" in content.lower():
                    # 未来可以在这里接入 JSON 转换并用 st.plotly_chart 渲染
                    pass


# ====================================================
# 核心对话与处理触发器
# ====================================================
user_prompt = st.chat_input("请向您的知识库提问：")

# 检查是否触发了新的语音录制 (防重复哈希校验)
voice_triggered = False
if audio_bytes is not None:
    current_audio_hash = hash(audio_bytes)
    if current_audio_hash != st.session_state.last_audio_hash:
        voice_triggered = True
        st.session_state.last_audio_hash = current_audio_hash

# 当用户敲击回车 或 录制完新语音时触发全链路
if user_prompt or voice_triggered:
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
        
    # 其次处理上传的文件附件
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
        # 刷新上传框
        st.session_state.uploader_key += 1
    
    # 组合最终的用户 Payload 存入记忆
    user_content.append({"type": "text", "text": user_prompt})
    memory.add_user_message(user_content)
    
    # 在前端立即渲染用户的输入
    with st.chat_message("user"):
        if chat_media: st.caption(f"📎 附带媒体: {chat_media.name}")
        if voice_triggered: st.caption("🎤 附带实时语音指令")
        st.markdown(user_prompt)

    # ====================================================
    # 后端大模型流式响应处理
    # ====================================================
    with st.chat_message("assistant"):
        use_thinking_for_this_turn = st.session_state.enable_thinking
        messages_to_send = memory.get_llm_payload(user_content)

        # =========================================================
        # 阶段零：本地 RapidOCR 物理脱水 (Media Dehydration)
        # =========================================================
        media_context = ""
        if chat_media: # 或者是 uploaded_files，看你 UI 里怎么存的
            with st.spinner("👁️ 正在启动 RapidOCR 视觉前置信息提取..."):
                # 注意：引擎要求传入 list 格式
                media_context = asyncio.run(multimodal_engine.process_files([chat_media]))
                if media_context:
                    st.toast("✅ 视觉特征脱水完成！", icon="👁️")

        # 将脱水文本与用户原话强行绑定，给状态机注入“上帝视角”
        full_user_query = user_prompt
        if media_context:
            full_user_query = f"{media_context}\n\n【用户针对附件的提问】：{user_prompt}"
        
        # =========================================================
        # 🚀 状态机编排接管：路由 -> 提纯 -> 并发检索 -> 提示词拼装
        # =========================================================
        with st.spinner("🧠 战略主脑已接管：正在执行图结构并发调度..."):
            # 一键流转状态机
            final_state = asyncio.run(orchestrator.run(
                user_query=full_user_query, # 传给它融合了 OCR 的巨型文本
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
        
        # 性能优化：媒体资源全局智能剥离 (防爆显存机制)
        # 如果当前回合没有任何多模态意图被触发，强制删除 Payload 中的多模态大文件
        multimodal_intents = {"analyze_image", "analyze_audio", "analyze_video"}
        if not any(i in multimodal_intents for i in intents):
            stripped_current_media = False
            for msg in messages_to_send:
                if isinstance(msg["content"], list):
                    original_len = len(msg["content"])
                    # 仅保留 text 类型的内容
                    text_only_content = [item for item in msg["content"] if item["type"] == "text"]
                    msg["content"] = text_only_content
                    if msg is messages_to_send[-1] and original_len > len(text_only_content):
                        stripped_current_media = True
            if stripped_current_media:
                st.toast("♻️ 媒体附件与当前意图无关，已在底层剥离以节省 GPU 算力！", icon="🍃")

        # 初始化 Assistant 回复占位
        memory.add_assistant_message(thought="", content="")

        # =========================================================
        # 🚀 触发大模型生成 (带全局算力锁保护)
        # =========================================================
        with llm_lock:
            stream = client.chat.completions.create(
                model="gpt-3.5-turbo", 
                messages=messages_to_send,
                stream=CFG["llm_generation"]["stream"],
                temperature=CFG["llm_generation"]["temperature"],
                top_p=CFG["llm_generation"]["top_p"],
                max_tokens=CFG["llm_generation"]["max_tokens"],
                stop=["<turn|>", "<|turn|>", "<eos>", "<end_of_turn>"],
                extra_body={"top_k": CFG["llm_generation"]["top_k"]}
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
            
            # 实时流式解析
            for chunk in stream:
                delta = chunk.choices[0].delta
                
                # 兼容不同后端的思维链 (Reasoning) 字段
                reasoning_chunk = getattr(delta, "reasoning_content", None)
                if not reasoning_chunk and hasattr(delta, "model_extra") and delta.model_extra:
                    reasoning_chunk = delta.model_extra.get("reasoning_content", "")
                reasoning_chunk = reasoning_chunk or ""
                content_chunk = delta.content or ""
                
                raw_thought += reasoning_chunk
                raw_content += content_chunk
                
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
                        summary_res = client.chat.completions.create(
                            model="gpt-3.5-turbo",
                            messages=request_payload,
                            stream=False 
                        )
                        new_summary = summary_res.choices[0].message.content
                        mem_obj.update_summary(new_summary)
                        print("✅ [后台任务] 长期记忆静默更新完成")
                    except Exception as e:
                        print(f"❌ [后台任务] 摘要生成失败: {e}")
                    finally:
                        llm_lock.release()
                else:
                    print("⚠️ [后台避让] 检测到大模型正忙，主动放弃本次摘要，让渡算力给主对话。")

            summary_thread = threading.Thread(target=background_summarize, args=(summary_request, memory))
            add_script_run_ctx(summary_thread) 
            summary_thread.start()
            
            st.toast("🔄 触发记忆阈值，将在后台空闲时静默压缩...", icon="🗄️")