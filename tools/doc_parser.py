import os
import tempfile
import hashlib
import subprocess
import shutil
from datetime import datetime
import streamlit as st

# 引入核心底层组件
from core.config import CFG
from core.hardware import HardwareManager
from core.database import EmbeddingService, DatabaseService

# 引入 LlamaIndex 自带的全能极速读取器
from llama_index.core import VectorStoreIndex, Document, StorageContext, Settings, SimpleDirectoryReader
from pymilvus import Collection, connections

import json
from llama_index.core import Document
from llama_index.core.node_parser import SentenceWindowNodeParser

import pickle
import logging
import gc

logger = logging.getLogger("AgenticRAG")

# ==========================================================
# 视觉解析依赖检测
# ==========================================================
# 全局绝对不实例化任何模型，只检查依赖是否就绪
try:
    from rapidocr_onnxruntime import RapidOCR
    from rapid_table import RapidTable, RapidTableInput
    HAS_OCR = True
    logger.info("✅ 视觉解析组件 (RapidOCR/Table) 依赖正常")
except Exception as e:
    HAS_OCR = False
    logger.warning(f"⚠️ 视觉解析组件未完整安装: {e}")
    
def extract_table_with_ocr(image_path):
    """
    终极版：提取文字，并完美重组为 HTML 表格格式！
    """
    if not HAS_OCR or not os.path.exists(image_path):
        return ""
    
    ocr_engine = None
    table_engine = None

    try:
        # 临时向系统申请内存拉起引擎
        logger.info(f"👁️ 侦测到复杂图片表格，正在临时向系统申请内存启动 Rapid 引擎...")

        # 1. 临时初始化纯文字扫描器
        ocr_engine = RapidOCR()
        # 2. 临时初始化表格重组器 (1.0.5 规范)
        table_config = RapidTableInput() 
        table_engine = RapidTable(table_config)
        
        # 第一步：硬核文字扫描 (拿到坐标点位和文字)
        ocr_result, _ = ocr_engine(image_path)
        
        if not ocr_result:
            return ""
            
        # 第二步：空间结构重组 (兼容不同版本的传参方式)
        try:
            table_result = table_engine(image_path, ocr_result=ocr_result)
        except TypeError:
            table_result = table_engine(image_path, ocr_result)
        
        # 第三步：极其严谨地提取 HTML 属性 (完美兼容 1.0.5 的对象返回格式)
        table_html_str = ""
        if hasattr(table_result, "pred_html") and table_result.pred_html:
            table_html_str = table_result.pred_html
        elif hasattr(table_result, "pred_htmls") and table_result.pred_htmls:
            table_html_str = table_result.pred_htmls[0]
        elif isinstance(table_result, tuple):
            table_html_str = str(table_result[0])
        elif isinstance(table_result, str):
            table_html_str = table_result
            
        return table_html_str
    except Exception as e:
        print(f"❌ RapidTable 重组表格失败: {e}")
        return ""
    
    finally:
        # 强制销毁实例，触发垃圾回收
        if ocr_engine is not None:
            del ocr_engine
        if table_engine is not None:
            del table_engine
            
        gc.collect() # 物理清空 RAM
        logger.info("♻️ 解析完毕，Rapid 引擎实例已销毁，内存已归还系统。")
# ==========================================================

# 👑 1. 初始化滑动窗口切分器 (大块包含前后各3句，小块是单句)
node_parser = SentenceWindowNodeParser.from_defaults(
    window_size=3, 
    window_metadata_key="window",
    original_text_metadata_key="original_text",
)

def process_mineru_to_documents(json_file_path: str, file_name: str, file_hash: str, upload_time: str):
    """
    解析 MinerU 生成的 JSON，遇到图片表格直接启动本地 OCR
    """
    documents = []
    # 获取 json 文件所在的绝对目录，因为里面的图片路径是相对的
    base_dir = os.path.dirname(json_file_path)
    
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    for block in data:
        block_type = block.get('type', '')
        valid_types = ['text', 'title', 'text_block', 'table', 'table_caption', 'equation', 'equation_caption']
        
        if block_type in valid_types:
            raw_content = ""
            
            if block_type == 'table':
                raw_content = block.get('html') or block.get('md') or block.get('text')
                if not raw_content:
                    res_dict = block.get('res', {})
                    if isinstance(res_dict, dict):
                        raw_content = res_dict.get('html') or res_dict.get('md') or res_dict.get('text')
                
                # =========================================================
                # 👑 发现 MinerU 摆烂截了图，立刻唤醒 OCR 引擎
                # =========================================================
                if not raw_content and 'img_path' in block:
                    img_rel_path = block.get('img_path')
                    # 拼出图片的真实本地绝对路径
                    full_img_path = os.path.join(base_dir, img_rel_path)
                    
                    st.toast(f"🧩 侦测到复杂图片表格，正在启动本地 OCR 强解...", icon="⚙️")
                    ocr_html = extract_table_with_ocr(full_img_path)
                    
                    if ocr_html:
                        # 成功将图片转为 HTML 数据！ f"***[本地OCR深度还原的表格数据]***\n{ocr_html}"
                        raw_content = f"{ocr_html}"
                    else:
                        # 兜底
                        raw_content = f"[表格以图片形式存在，路径：{img_rel_path}]"
                        
            elif block_type == 'equation':
                raw_content = block.get('latex') or block.get('md') or block.get('text') or ""
            else:
                raw_content = block.get('text') or block.get('md') or ""
                
            content = str(raw_content).strip()
            page_no = block.get('page_idx', 0) + 1  
            
            if content and content != "{}":
                doc = Document(
                    text=content,
                    metadata={
                        "file_name": file_name,
                        "file_hash": file_hash, 
                        "file_type": "pdf",
                        "page_label": str(page_no),
                        "upload_time": upload_time, # 写入上传时间
                        "parser": "MinerU_with_OCR", # 标记
                        "block_type": block_type
                    }
                )
                documents.append(doc)
                
    return documents

def get_file_hash(file_bytes):
    return hashlib.md5(file_bytes).hexdigest()

def process_and_embed_documents(uploaded_files):
    """
    核心工具：处理上传的 PDF、Word、Excel、TXT、代码 等文档。
    采用智能分流机制：
    1. PDF: 走 MinerU 视觉模型深度解析
    2. 其他: 走 LlamaIndex 极速结构化提取
    """
    if not uploaded_files:
        st.warning("请先上传至少一个文件！")
        return

    files_to_process = []
    
    # ---------------------------------------------------------
    # 1. 无感前置查重
    # ---------------------------------------------------------
    with st.spinner("🔍 正在检查知识库防重..."):
        try:
            connections.connect(host=CFG["llm_server"]["host"], port=CFG["milvus"]["uri"].split(":")[-1])
            collection = Collection(CFG["rag"]["collection_name"])
            for file in uploaded_files:
                file_bytes = file.getvalue()
                file_hash = get_file_hash(file_bytes)
                search_res = collection.query(expr=f'file_hash == "{file_hash}"', output_fields=["file_hash"], limit=1)
                if len(search_res) > 0:
                    st.toast(f"⚠️ 跳过：【{file.name}】已存在知识库中")
                else:
                    file.file_bytes = file_bytes
                    file.file_hash = file_hash
                    files_to_process.append(file)
        except Exception:
            # 如果是第一次建库报错，把所有文件列为新文件
            for file in uploaded_files:
                file.file_bytes = file.getvalue()
                file.file_hash = get_file_hash(file.file_bytes)
                files_to_process.append(file)

    if not files_to_process:
        st.success("✅ 所有上传的文件均已在知识库中。大模型服务未中断，您可以直接关闭本窗口提问！")
        return

    # ---------------------------------------------------------
    # 2. 智能分流解析引擎 (Smart Routing Parser)
    # ---------------------------------------------------------
    with st.status("开始处理新文档...", expanded=True) as status:
        try:
            status.update(label="正在挂起对话大模型，释放硬件资源...")
            HardwareManager.stop_llm_service()
            
            progress_bar = st.progress(0, text="🚀 正在启动解析引擎...")
            total_files = len(files_to_process)
            # 👑 替换为在顶部定义好的“滑动窗口”：
            splitter = node_parser
            
            for i, file in enumerate(files_to_process):
                # 释放上一轮的 BGE 显存，确保解析引擎有充足空间
                EmbeddingService.unload(reason="immediate")

                # 👑 生成当前文件的精确上传时间 (格式: YYYY-MM-DD HH:MM:SS)
                current_upload_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                base_progress = int((i / total_files) * 100)
                file_weight = 100 / total_files
                
                def update_progress(ratio, msg):
                    current_val = min(int(base_progress + (file_weight * ratio)), 100)
                    # 1. 更新内部的进度条
                    progress_bar.progress(current_val, text=f"**[{i+1}/{total_files}] {current_val}%** - {msg}")
                    # 2. 👑 核心优化：同步更新外部的 Status 标题！
                    # 这样即使用户折叠了面板，也能在外面看到当前正在干嘛
                    status.update(label=f"⏳ [{i+1}/{total_files}] {file.name} | {msg}")

                # 将文件写入临时目录
                work_dir = tempfile.mkdtemp()
                file_path = os.path.join(work_dir, file.name)
                with open(file_path, "wb") as f:
                    f.write(file.file_bytes) 
                
                file_ext = file.name.split('.')[-1].lower()
                md_content = ""
                # 用来存放携带富元数据（如页码）的独立文档对象集合
                documents_to_chunk = []

                # ========================================================
                # 👑 核心：智能分流解析逻辑 (Routing Logic)
                # ========================================================
                if file_ext == "pdf":
                    # 通道 A：重量级视觉解析 (MinerU) - 专攻复杂排版和公式
                    update_progress(0.15, f"⚙️ 启动 MinerU 视觉模型分析: {file.name}")
                    try:
                        my_env = os.environ.copy()
                        my_env["MFR_BATCH_SIZE"] = str(CFG["mineru"]["mfr_batch_size"])
                        result = subprocess.run(
                            [CFG["mineru"]["exe_path"], "-p", file_path, "-o", work_dir],
                            check=True, capture_output=True, text=True, encoding="utf-8", errors="replace", env=my_env
                        )
                    except subprocess.CalledProcessError as e:
                        raise Exception(f"MinerU 运行崩溃！\n【错误日志】:\n{e.stderr}")
                    
                    update_progress(0.70, f"📄 MinerU 解析完毕，正在提取底层 JSON 结构化数据...")
                    
                    # 👑 寻找 MinerU 生成的底层 json 文件 (而不是笼统的 md)
                    json_file_path = None
                    for root, dirs, files in os.walk(work_dir):
                        for f in files:
                            # MinerU 的内容结构通常存在以 json 结尾的文件中
                            if f.endswith('.json') and not f.endswith('layout.json'): 
                                json_file_path = os.path.join(root, f)
                                break
                        if json_file_path: break
                            
                    if not json_file_path:
                        raise Exception(f"MinerU 未找到底层的 JSON 结构化文件。")
                        
                    # 👑 核心升级：调用新函数，直接拿到带有精准页码和 hash 的小块 Document，current_upload_time 传给 MinerU 解析函数
                    docs = process_mineru_to_documents(json_file_path, file.name, file.file_hash, current_upload_time)
                    documents_to_chunk.extend(docs)
                    
                    # 为了在前端 UI 的 expander 中还能正常预览，我们把提取出的纯文本拼一下
                    md_content = "\n\n".join([d.text for d in docs])
                        
                else:
                    # 通道 B：轻量级极速解析 (LlamaIndex) - 秒解 Office/纯文本/代码
                    update_progress(0.15, f"⚡ 启动极速结构化提取: {file.name}")
                    try:
                        reader = SimpleDirectoryReader(input_files=[file_path])
                        parsed_docs = reader.load_data() # 这里返回的是自带页码等原生元数据的列表
                        
                        # 👑 核心修复：遍历注入自定义元数据，完美保留原生 page_label
                        for p_doc in parsed_docs:
                            p_doc.metadata.update({
                                "file_name": file.name,
                                "file_hash": file.file_hash,
                                "file_type": file_ext,
                                "upload_time": current_upload_time, # 文件上传时间
                                "parser": "LlamaIndex"
                            })
                            documents_to_chunk.append(p_doc)
                            
                        # 提取所有页面的文本拼合成一个大字符串（仅用于给前端UI预览）
                        md_content = "\n\n".join([doc.text for doc in parsed_docs])
                    except Exception as e:
                        raise Exception(f"极速解析引擎读取失败: {e}")
                        
                    update_progress(0.70, f"📄 极速提取完毕...")
                # ========================================================

                # 兜底：如果提取出来是空的，直接跳过防崩溃
                if not documents_to_chunk:
                    st.warning(f"⚠️ 文件 {file.name} 解析出的内容为空，跳过入库。")
                    shutil.rmtree(work_dir, ignore_errors=True)
                    continue

                update_progress(0.80, f"🧩 正在进行语义切块...")

                # 在前端渲染解析结果 (保留原有预览逻辑)
                with st.expander(f"🔍 查看【{file.name}】底层解析数据", expanded=False):
                    tab1, tab2 = st.tabs(["📊 格式化预览", "📝 源码"])
                    with tab1: st.markdown(md_content, unsafe_allow_html=True)
                    # 如果不是 PDF，则根据文件后缀高亮代码块；如果是 PDF，默认显示 Markdown
                    with tab2: st.code(md_content, language="markdown" if file_ext == "pdf" else file_ext)

                # 向量化入库
                update_progress(0.85, f"🔋 正在唤醒 BGE-M3 向量模型 (GPU)...")
                EmbeddingService.load(device="cuda")
                update_progress(0.90, f"📥 正在将区块向量化存入 Milvus...")
                
                # 👑 优雅入库：直接将装满原生与自定义元数据的文档列表送去切块
                nodes = splitter.get_nodes_from_documents(documents_to_chunk)
                VectorStoreIndex(
                    nodes, 
                    storage_context=StorageContext.from_defaults(vector_store=DatabaseService.get_vector_store()), 
                    embed_model=Settings.embed_model
                )

                # ==========================================================
                # 👑 核心补丁：为 BM25 混合检索保留本地纯文本 Nodes 备份
                # ==========================================================
                NODES_FILE = "local_bm25_nodes.pkl"
                existing_nodes = []
                # 如果本地已经有文件，先读出来
                if os.path.exists(NODES_FILE):
                    with open(NODES_FILE, "rb") as f:
                        existing_nodes = pickle.load(f)
                
                # 把本次新解析的 nodes 追加进去
                existing_nodes.extend(nodes)
                
                # 重新写回硬盘
                with open(NODES_FILE, "wb") as f:
                    pickle.dump(existing_nodes, f)
                # ==========================================================
                
                st.success(f"✅ {file.name} 本地与 Milvus 双端入库成功！")
                update_progress(1.0, f"✅ {file.name} 完美入库！")
                shutil.rmtree(work_dir, ignore_errors=True)
            
            progress_bar.progress(100, text="**[完成] 100%** - 所有新文档处理完毕！")
        
        except Exception as e:
            st.error(f"❌ 处理发生异常: {e}")
            
        finally:
            # ---------------------------------------------------------
            # 3. 完美退场，恢复对话环境
            # ---------------------------------------------------------
            status.update(label="正在释放解析引擎显存，准备重启对话大模型...", state="running")
            EmbeddingService.unload(reason="immediate")
            # 在极其干净的显存环境中，安全点火拉起大语言模型
            HardwareManager.start_llm_service()