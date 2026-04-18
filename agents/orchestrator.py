import re
import asyncio
import streamlit as st  # 引入 st
from core.query_transformer import QueryTransformer
from core.config import PROMPTS

class AgentState:
    """
    状态记忆体 (State Payload)
    包裹所有信息，在节点间流转。
    """
    def __init__(self, user_query: str, has_media: bool):
        self.user_query = user_query
        self.has_media = has_media
        self.intents = []
        self.parameters = {}
        self.sub_queries = []
        self.hyde_passage = ""
        self.rag_nodes = []
        
        # 最终产出的系统提示词模块
        self.system_content_blocks = []

class GraphOrchestrator:
    def __init__(self, router_engine, rag_engine, web_retriever):
        self.router = router_engine
        self.rag = rag_engine
        self.web = web_retriever
        self.transformer = QueryTransformer()

    def _apply_architect_patch(self, state: AgentState, enable_web_search: bool):
        """
        终极意图自愈补丁 (The Ultimate Intent Healer)
        作用：在大模型返回意图后进行硬规则二次矫正，确保高价值链路（RAG/Web/多模态）不会被漏掉。
        """
        # 统一转为小写处理，提高匹配灵敏度
        query = state.user_query.lower()

        # --- 🛡️ 策略 A：语义熔断机制 (Negation Circuit Breaker) ---
        # 识别用户明确拒绝的操作，防止补丁“过度服务”
        # 匹配模式：(否定词) + (任意文字) + (功能词)
        negation_regex = r"(不用|不要|别|无需|停止|关闭|禁止|取消|强制不|不必)(.+?)(上网|联网|搜索|插件|附件|库|图|看)"
        is_hard_blocked = re.search(negation_regex, query)

        # --- 🌐 策略 B：联网搜索补偿矩阵 (Web Search Compensation) ---
        # 如果开关开启且模型未识别出 Web 意图，则进行正则补位
        if enable_web_search and "web_search" not in state.intents and not is_hard_blocked:
            web_patterns = [
                # 1. 时效性特征：股价、天气、实时新闻
                r"(今天|今日|现在|目前|刚刚|近期|最新|实时|动态|走势|股价|汇率|行情|价格|天气|排班|赛事)",
                # 2. 资讯特征：财报、公告、八卦、推文
                r"(新闻|资讯|报道|发布会|财报|公告|消息|传闻|热点|八卦|推文)",
                # 3. 平台特征：明确提到搜索引擎或社交媒体
                r"(上网|全网|联网|搜索|查一查|搜一下|百度|谷歌|google|知乎|百科|维基|wiki|小红书|雪球|微博)",
                # 4. 知识盲区特征：针对定义、评价或“有没有”类问题
                r"(是什么|怎么回事|谁是|介绍一下|科普|评价|怎么样|有没有|哪里有|官网)"
            ]
            # 只要命中任意一个正则组，就强制开启联网
            if any(re.search(p, query) for p in web_patterns):
                state.intents.append("web_search")
                print("🛡️ [补丁] 探测到强时效性/外部需求，已强制追加 web_search")

        # --- 📚 策略 C：本地知识库补偿矩阵 (Local RAG Compensation) ---
        # 针对提及本地资料或文件的指令进行增强
        if "search_knowledge_base" not in state.intents:
            rag_patterns = [
                # 1. 明确的文件后缀或类型
                r"(研报|文档|文件|附件|库里|资料|记录|归档|pdf|docx|txt|excel|表格|ppt|简历|合同|数据)",
                # 2. 空间方位词：内部、本地、我给你的
                r"(内部|私有|本地|内网|我给你的|之前传的|资料|知识库|自有|规章|流程|标准|规范)",
                # 3. 处理动作：总结、归纳、针对...
                r"(提取|总结|归纳|查找|翻一下|看一下|针对上述|基于这些)"
            ]
            if any(re.search(p, query) for p in rag_patterns):
                state.intents.append("search_knowledge_base")
                print("🛡️ [补丁] 探测到本地资料引用，已强制追加 search_knowledge_base")

        # --- 👁️ 策略 D：多模态强制纠偏 (Multimodal Reinforcement) ---
        # 只有在确实存在附件的情况下才触发，防止空载
        if state.has_media:
            # 视觉识别纠偏
            vision_regex = r"(看下|识别|扫描|ocr|提取文字|这张图|照片|截图|画面|细节|背景|颜色|原图|这秒)"
            if re.search(vision_regex, query) and "analyze_image" not in state.intents:
                state.intents.append("analyze_image")

            # 视频/时间线纠偏
            video_regex = r"(视频|这段|播放|动态|动作|过程|完整版|这分钟|镜头|时刻|回放)"
            if re.search(video_regex, query) and "analyze_video" not in state.intents:
                state.intents.append("analyze_video")

            # 音频/听觉纠偏
            audio_regex = r"(听|录音|声音|这段话|语音|背景音|音质|说啥|转录|听写)"
            if re.search(audio_regex, query) and "analyze_audio" not in state.intents:
                state.intents.append("analyze_audio")

        # --- 🧹 策略 E：意图净化与参数对齐 (Sanitization) ---
        # 1. 去重
        state.intents = list(set(state.intents))

        # 2. 互斥逻辑：如果已经确定需要执行硬技能（RAG/Web/视觉），则移除无意义的 "chat"
        if len(state.intents) > 1 and "chat" in state.intents:
            state.intents.remove("chat")

        # 3. 参数对齐：如果补丁强制开启了搜索意图，但 parameters 字典是空的
        # 强制将用户原句设为 search_query，确保后续提取器能正常工作
        if any(i in ["search_knowledge_base", "web_search"] for i in state.intents):
            if not state.parameters.get("search_query"):
                state.parameters["search_query"] = state.user_query if state.user_query.strip() else "综合分析"

    # -----------------------------------------------------
    # 节点 1：路由与提取 (Route & Transform)
    # -----------------------------------------------------
    async def _node_route_and_transform(self, state: AgentState, enable_web_search: bool):
        """
        节点：意图路由与关键词提取
        """
        # --- 步骤 1：调用大模型进行原始意图分析 ---
        route_result = self.router.analyze_intent(
            state.user_query, 
            has_media=state.has_media, 
            enable_web_search=enable_web_search
        )
        state.intents = route_result.get("intents", ["chat"])
        state.parameters = route_result.get("parameters", {})

        # --- 步骤 2：注入“架构师补丁”进行硬规则纠偏 ---
        # 这一步是关键，它能救回那些模型判断错误的意图
        self._apply_architect_patch(state, enable_web_search)

        # --- 步骤 3：根据最终确定的意图，启动提取大脑 ---
        # 只要存在任何形式的检索需求，就必须进入提取环节
        if "search_knowledge_base" in state.intents or "web_search" in state.intents:
            # 这里的 transformer.transform 是异步的，需要 await
            state.sub_queries, state.hyde_passage = await self.transformer.transform(state.user_query)
        else:
            # 纯聊天意图下，关键词即为原句
            state.sub_queries = [state.user_query]
        
        # --- 步骤 4：UI 播报（此时播报的信息是经过补丁修正后的最准意图） ---
        st.toast(f"🧩 最终决策: {state.intents}")
        if "search_knowledge_base" in state.intents or "web_search" in state.intents:
            st.toast(f"🎯 提取关键词: {' | '.join(state.sub_queries)}")
            
        return state

    # -----------------------------------------------------
    # 节点 2：主脑并发调度 (Parallel Retrieve)
    # -----------------------------------------------------
    async def _node_parallel_retrieve(self, state: AgentState):
        """
        职责：外部数据获取 (Data Sourcing)。
        将 RAG 和 Web 检索并行化，极大压缩总 IO 耗时。
        """
        tasks = []
        task_names = [] # 用于结果对齐的标识

        # 1. 挂载本地知识库任务
        if "search_knowledge_base" in state.intents:
            # RAG 是 CPU 密集型计算（重排/向量化），使用 to_thread 避免卡死协程
            tasks.append(asyncio.to_thread(
                self.rag.retrieve_and_format, 
                state.user_query, state.sub_queries, state.hyde_passage
            ))
            task_names.append("rag")

        # 2. 挂载全域联网任务
        if "web_search" in state.intents:
            # 爬虫是原生异步 I/O，直接加入协程池
            tasks.append(self.web.search_and_scrape(state.sub_queries, max_results=3))
            task_names.append("web")

        # 3. 并发执行并处理结果
        if tasks:
            # return_exceptions=True 保证其中一个炸了，另一个还能用
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for idx, res in enumerate(results):
                # 容错处理：如果任务返回的是异常对象
                if isinstance(res, Exception):
                    st.error(f"🚨 检索链路 [{task_names[idx]}] 出错: {res}")
                    continue
                
                # 分发 RAG 结果
                if task_names[idx] == "rag":
                    nodes, context_str = res
                    state.rag_nodes = nodes
                    if context_str:
                        state.system_content_blocks.append(PROMPTS["rag_system_prompt"].format(context_str=context_str))
                        st.toast("📚 本地资料检索完成", icon="✅")
                
                # 分发 Web 结果
                elif task_names[idx] == "web":
                    if res and "未获取到" not in res:
                        state.system_content_blocks.append(PROMPTS["web_search_system_prompt"].format(web_context=res))
                        st.toast("🌐 网络信息抓取完成", icon="✅")

        return state

    # -----------------------------------------------------
    # 节点 3：多模态与兜底提示词拼装 (Prompt Assembly)
    # -----------------------------------------------------
    async def _node_assemble_prompts(self, state: AgentState):
        """
        职责：上下文塑形 (Context Shaping)。
        不涉及任何耗时检索，纯粹基于意图生成“行为指令”模块。
        """
        # --- A. 视觉分析指令 ---
        if "analyze_image" in state.intents:
            state.system_content_blocks.append(PROMPTS["vision_system_prompt"])
            
        # --- B. 音频分析指令 ---
        if "analyze_audio" in state.intents:  
            state.system_content_blocks.append(PROMPTS["audio_system_prompt"])
            
        # --- C. 视频时序分析指令 ---
        if "analyze_video" in state.intents:  
            video_prompt = PROMPTS["video_system_prompt"]
            # 注入 Router 识别出的时间锚点参数
            time_focus = state.parameters.get("time_focus", "")
            if time_focus:
                video_prompt += f"\n\n🚨 指挥官特别指令：请优先分析视频中 【{time_focus}】 附近的画面逻辑。"
            state.system_content_blocks.append(video_prompt)
            
        # --- D. 全局兜底逻辑 ---
        # 判定标准：如果没有检索到任何数据，且没有多模态指令，则启动“暖场聊天”提示词
        if not state.system_content_blocks:
            # 检查是否有显式触发 chat
            if "chat" in state.intents:
                chat_prompt = PROMPTS.get("chat_system_prompt", "你是一个极其专业、具备批判性思维的知识库 Agent 助手。")
                state.system_content_blocks.append(chat_prompt)
            else:
                # 终极防御：如果意图全空（概率极低），给一个通用专家设定
                state.system_content_blocks.append("你是一个严谨的 AI 助手。")

        return state

    # -----------------------------------------------------
    # 运行图结构 (Run Graph)
    # -----------------------------------------------------
    async def run(self, user_query: str, has_media: bool, enable_web_search: bool):
        state = AgentState(user_query, has_media)
        state = await self._node_route_and_transform(state, enable_web_search)
        state = await self._node_parallel_retrieve(state)
        state = await self._node_assemble_prompts(state)
        return state