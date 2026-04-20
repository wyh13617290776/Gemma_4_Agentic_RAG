import os
import sys
import requests
import urllib.parse
import asyncio

import logging
import concurrent.futures
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode

from markdownify import markdownify as md

# 引入项目全局配置
from core.config import CFG
# 引入反思模块
from engines.reflection_engine import ReflectionEngine

# 配置系统日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - 🚀 %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# ==========================================
# 🚨 核心修复：解决 Windows 下 Streamlit 线程调用 Playwright 失败的 Bug
# ==========================================
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

class UltimateWebRetriever:
    """
    高级联网检索器：集成 SearXNG, Serper, Tavily 和 Exa 的多级 RAG 检索系统。
    具备自动代理隔离、多线程并发抓取以及知乎级反爬虫破盾能力。
    """

    def __init__(self):
        """
        初始化检索器，从 CFG (由 config.yaml 和 .env 合并) 加载所有 API Key。
        初始化直连 Session 以彻底规避系统代理导致的 SSL/Proxy 错误。
        """
        api_keys = CFG.get("api_keys", {})
        self.exa_key = api_keys.get("exa_api_key", "")
        self.serper_key = api_keys.get("serper_api_key", "")
        self.tavily_key = api_keys.get("tavily_api_key", "")
        
        # 提取字典
        search_engines = CFG.get("search_engines", {})
        
        # 1. 基础配置
        self.searxng_url = search_engines.get("searxng_url")
        
        # 2. 级联检索上限参数 (带默认保底值)
        self.searxng_max_results = search_engines.get("searxng_max_results")
        self.tavily_max_results = search_engines.get("tavily_max_results")
        self.exa_max_results = search_engines.get("exa_max_results")

        self.MIN_EFFECTIVE_URLS = search_engines.get("min_effective_urls")
        
        # 3. 初始化独立的“反思引擎”
        self.reflector = ReflectionEngine(
            min_urls=search_engines.get("min_effective_urls"),
            min_length=search_engines.get("min_content_length"),
            k_coverage=search_engines.get("keyword_coverage")
        )

        # 动态读取配置
        network_cfg = CFG.get("network", {})
        proxy_url = network_cfg.get("proxy_url", "")
        # 如果 yaml 里没写，给一个最稳妥的默认白名单
        no_proxy = network_cfg.get("no_proxy", "localhost,127.0.0.1,0.0.0.0,::1")

        self.session = requests.Session()
        
        # 2. 只有配置了 proxy_url 时才启用代理
        if proxy_url:
            self.session.proxies = {
                "http": proxy_url,
                "https": proxy_url,
            }
        
        # 3. 核心修复：动态注入白名单环境变量
        os.environ['NO_PROXY'] = no_proxy
        
        logger.info(f"✅ 代理环境已对齐：API 走 {proxy_url}，本地 SearXNG 自动绕过")
        
        logger.info(f"✅ WebRetriever 深度模式启动：自省阈值设定为 {self.MIN_EFFECTIVE_URLS}")
        logger.info(f"✅ 阶梯检索配置就绪：分级上限 ——> SearXNG：{self.searxng_max_results} Tavily：{self.tavily_max_results} Exa API：{self.exa_max_results}")

    def _format_reference(self, title, url, content, index=None):
        """
        统一 Markdown 参考资料格式化工具。
        
        Args:
            title (str): 网页标题
            url (str): 网页链接
            content (str): 提取出的正文内容
            index (int, optional): 来源索引序号
            
        Returns:
            str: 格式化后的 Markdown 字符串，正文限制在 2000 字符内以节省上下文空间
        """
        prefix = f"[{index}] " if index else ""
        return f"### 来源 {prefix}{title}\n🔗 {url}\n\n{content[:2000].strip()}...\n"

    def _tavily_request(self, endpoint: str, payload_ext: dict) -> dict:
        """
        Tavily 通用请求中心。
        
        Args:
            endpoint (str): 对应你截图的功能，如 'search' 或 'extract'
            payload_ext (dict): 具体的业务参数
            
        Returns:
            dict: API 返回的原始数据
        """
        if not self.tavily_key: return {}
        
        # 对应你截图中的 API 地址
        url = f"https://api.tavily.com/{endpoint}"
        
        # 基础 Payload，包含你的 API Key
        payload = {"api_key": self.tavily_key}
        payload.update(payload_ext)

        try:
            # 策略 A：对于外部 API，如果代理报错，尝试禁用代理直连
            response = self.session.post(url, json=payload, timeout=15)
            if response.status_code != 200:
                return {}
            return response.json()
        except Exception as e:
            logger.error(f"❌ Tavily 请求异常，尝试临时脱离代理重试...")
            # 备选方案：临时使用不带代理的 session
            import requests
            try:
                temp_res = requests.post(url, json=payload, timeout=10)
                return temp_res.json() if temp_res.status_code == 200 else {}
            except:
                return {}

    # ==========================================
    # 修改 1：彻底解开异步封印，不再嵌套 asyncio.run
    # ==========================================
    async def _scrape_urls_concurrently(self, urls: list) -> list:
        """
        优雅的原生异步并发抓取：采用 0.8.6 原生 Generator 和 Filter 命名。
        完全剥离同步外壳，完美融入主事件循环。
        """
        results = []
        
        from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
        
        try:
            from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator as MDGen
        except ImportError:
            from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerationStrategy as MDGen
        
        try:
            from crawl4ai.content_filter_strategy import PruningContentFilter
            my_filter = PruningContentFilter(threshold=0.45, min_word_threshold=5)
        except ImportError:
            try:
                from crawl4ai.content_filter_strategy import BM25ContentFilter
                my_filter = BM25ContentFilter(user_query=None, bm25_threshold=1.2)
            except ImportError:
                my_filter = None
        
        proxy_url = CFG.get("network", {}).get("proxy_url", "")
        
        browser_conf = BrowserConfig(
            proxy_config={"server": proxy_url} if proxy_url else "http://127.0.0.1:7897", 
            headless=True,
            verbose=False
        )
        
        md_kwargs = {}
        if my_filter:
            md_kwargs['content_filter'] = my_filter
        md_generator = MDGen(**md_kwargs)
        
        run_conf = CrawlerRunConfig(
            cache_mode=CacheMode.BYPASS, 
            word_count_threshold=20, 
            markdown_generator=md_generator, 
            delay_before_return_html=5.0,    
            magic=True,                      
            remove_overlay_elements=True     
        )

        # 直接使用 async with，不再套 fetch_all() 函数
        async with AsyncWebCrawler(config=browser_conf) as crawler:
            import asyncio
            tasks = [crawler.arun(url=u, config=run_conf) for u in urls]
            crawled_data = await asyncio.gather(*tasks, return_exceptions=True)
            
            for url, res in zip(urls, crawled_data):
                if isinstance(res, Exception):
                    print(f"🚨 Crawl4AI 并发异常: {url} | {res}")
                    results.append({"url": url, "content": "", "status": "error"})
                elif res and hasattr(res, 'markdown') and res.markdown and len(res.markdown) > 50:
                    print(f"✨ Crawl4AI 抓取成功 ({len(res.markdown)} 字符): {url}")
                    results.append({"url": url, "content": res.markdown, "status": "success"})
                else:
                    print(f"⚠️ Crawl4AI 提取内容过短或失败: {url}")
                    results.append({"url": url, "content": "", "status": "empty"})
        return results

    def get_urls(self, query: str, max_results: int = 3) -> list:
        """
        瀑布流获取 URL：优先本地 SearXNG，若失败则切换至商业 Serper。
        
        Args:
            query (str): 搜索关键词
            max_results (int): 期望获取的 URL 数量
            
        Returns:
            list: 包含有效 URL 字符串的列表
        """
        # 优先级 1: 本地自建 SearXNG
        try:
            # 修改后的 get_urls 内部逻辑
            params = {
                "q": query,
                "format": "json",
                "categories": "general",      # 明确搜索类别
                "engines": "google,bing,duckduckgo", # 剔除那些容易触发验证码的引擎
                "language": "en-US,zh-CN",    # 接受中英双语结果
                "time_range": "year",         # 缩小时间范围，增加结果权重
            }
            # 关键：给 SearXNG 请求也加上 User-Agent 头
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            }
            res = self.session.get(self.searxng_url, params=params, headers=headers, timeout=10) # 适当增加超时
            if res.status_code == 200:
                urls = [item['url'] for item in res.json().get('results', [])]
                if urls: return urls[:max_results]
        except Exception as e: 
            logger.warning(f"⚠️ SearXNG 侦察失败: {e}")

        # 优先级 2: 商业 Google Serper
        if self.serper_key:
            try:
                headers = {'X-API-KEY': self.serper_key, 'Content-Type': 'application/json'}
                # 策略 B：给 Serper 增加 verify=False (临时绕过 SSL 检查) 或禁用代理
                res = self.session.post(
                    "https://google.serper.dev/search", 
                    headers=headers, 
                    json={"q": query, "num": max_results}, 
                    timeout=5,
                    verify=False # 如果代理导致握手失败，这一行能救命
                )
                if res.status_code == 200:
                    urls = [item["link"] for item in res.json().get("organic", [])]
                    if urls: return urls
            except Exception as e:
                logger.error(f"⚠️ Serper 检索失败: {e}")
        return []

    def exa_search_and_get_text(self, query: str, max_results: int = 3) -> str:
        """
        语义核武器 Exa：基于 LLM 理解的语义搜索，并直接返回网页清洗后的正文。
        
        Args:
            query (str): 自然语言提问
            max_results (int): 期望返回的结果数
            
        Returns:
            str: 格式化好的 Markdown 全文内容
        """
        if not self.exa_key: return ""
        logger.info(f"🔥 Exa 语义搜索介入，执行降维打击: '{query}'")
        url = "https://api.exa.ai/search"
        payload = {"query": query, "useAutoprompt": True, "numResults": max_results, "contents": {"text": True}}
        headers = {"accept": "application/json", "content-type": "application/json", "x-api-key": self.exa_key}

        try:
            response = self.session.post(url, json=payload, headers=headers, timeout=15)
            response.raise_for_status()
            results = response.json().get("results", [])
            return "\n\n".join([self._format_reference(r.get("title"), r.get("url"), r.get("text"), i+1) for i, r in enumerate(results)])
        except Exception as e:
            logger.error(f"⚠️ Exa 检索彻底失败: {e}")
            return ""
    
    # ==========================================
    # 核心逻辑：带级联反思与自愈的检索引擎
    # ==========================================
    async def search_and_scrape(self, sub_queries: list, max_results: int = 3) -> str:
        """
        V2.0 全链路核心检索路由：具备深度质量审查与增量去重能力的级联检索引擎。
        
        本接口深度集成了 ReflectionEngine（反思引擎），不再盲目堆砌抓取结果，
        而是通过“质量门禁”与“饱和度监测”动态调度不同成本的检索方案。

        执行逻辑：
        1. 弹药整合：将战略中枢提取后的子查询列表拼接为全局检索关键词。
        2. 先锋抓取：使用 self.searxng_max_results 获取网址，并通过 Crawl4AI 异步脱水。
        3. 质量与去重审查 (核心)：
           - 防短/防毒/防跑题：通过 evaluate_quality 拦截反爬提示或低信息密度网页。
           - 增量去重防内卷：通过 is_redundant 拦截高度雷同的新闻通稿。
        4. 精准破盾：针对 Crawl4AI 失败的 URL 调用 Tavily Extract，破盾后同样需经过质量审查。
        5. 饱和度反思 I：通过 is_saturated 评估有效信息数。若匮乏，触发 Tavily Search 商业补偿。
        6. 饱和度反思 II：若信息依然未达阈值，动用 Exa 执行最终的语义降维打击。

        Args:
            sub_queries (list): 战略中枢提取后的核心关键词列表。
            max_results (int): (已废弃，被 yaml 中的阶梯配置接管) 仅为保持接口签名的兼容性。
            
        Returns:
            str: 经过高度提取、去重并组合了多路级联来源的 Markdown 格式参考资料。
        """
        search_query = " ".join(sub_queries)
        logger.info(f"🌟 启动全链路级联检索, 提取词: '{search_query}'")
        
        successful_contents = []

        # -----------------------------------------------------
        # 第一阶段：先锋方案 (使用 searxng_max_results，SearXNG/Serper -> Crawl4AI)
        # -----------------------------------------------------
        urls = await asyncio.to_thread(self.get_urls, search_query, self.searxng_max_results)
        
        if urls:
            scrape_results = await self._scrape_urls_concurrently(urls)
            failed_urls = []

            # 用于专门记录纯文本，方便计算去重
            raw_texts_for_dedup = []

            for res in scrape_results:
                if res["status"] == "success":
                    content = res['content']
                    # 1. 质量检验 (防短、防毒、防跑题)
                    if self.reflector.evaluate_quality(content, sub_queries):
                        # 2. 增量去重检验 (防新闻通稿)
                        if not self.reflector.is_redundant(content, raw_texts_for_dedup):
                            # 检验全部通过！正式编入战报
                            successful_contents.append(self._format_reference("联网搜索", res['url'], content))
                            raw_texts_for_dedup.append(content)
                        else:
                            # 算是有效抓取，但是因为重复被丢弃，不计入失败，直接忽略
                            pass 
                else:
                    failed_urls.append(res['url'])
            
            # 针对失败 URL 的精准破盾 (Tavily Extract)
            if failed_urls and self.tavily_key:
                logger.info(f"🛡️ 探测到 {len(failed_urls)} 个页面抓取失败，启动 Tavily Extract ...")
                for url in failed_urls:
                    t_res = await asyncio.to_thread(self._tavily_request, "extract", {"urls": [url]})
                    
                    # 防护装甲 3：严格校验提取结果的结构
                    if not t_res or not isinstance(t_res, dict):
                        continue
                        
                    res_list = t_res.get('results', [])
                    # 确保 res_list 是列表，且至少有1个元素，且该元素是字典，且包含 raw_content
                    if isinstance(res_list, list) and len(res_list) > 0 and isinstance(res_list[0], dict):
                        extracted_content = res_list[0].get('raw_content')
                        
                        if extracted_content:
                            if self.reflector.evaluate_quality(extracted_content, sub_queries):
                                if not self.reflector.is_redundant(extracted_content, raw_texts_for_dedup):
                                    successful_contents.append(self._format_reference("Tavily 破盾补位", url, extracted_content))
                                    raw_texts_for_dedup.append(extracted_content)

        # -----------------------------------------------------
        # 第二阶段：自省与 Tavily 商业补偿 (使用 tavily_max_results)
        # -----------------------------------------------------
        # 修复 2：使用模块化的饱度度检测，统一管理日志和阈值
        if not self.reflector.is_saturated(len(successful_contents)):
            logger.info("🚀 激活 Tavily 搜索补充...")
            backup_content = await asyncio.to_thread(self._tavily_full_backup, search_query, self.tavily_max_results)
            
            # 防护装甲 4：确保 backup_content 是有实际意义的字符串
            if backup_content and isinstance(backup_content, str):
                if "未获取到" not in backup_content and "不可用" not in backup_content and "未提取到正文" not in backup_content:
                    successful_contents.append(f"\n--- 💡 Tavily 搜索补充结果 ---\n{backup_content}")

        # -----------------------------------------------------
        # 第三阶段：终极自省与 Exa 语义打击 (使用 exa_max_results)
        # -----------------------------------------------------
        if not self.reflector.is_saturated(len(successful_contents)) and self.exa_key:
            logger.info("🚨 启动 Exa API方案...")
            exa_content = await asyncio.to_thread(self.exa_search_and_get_text, search_query, self.exa_max_results)
            
            if exa_content and isinstance(exa_content, str):
                successful_contents.append(f"\n--- 💣 Exa 语义检索结果 ---\n{exa_content}")
                logger.info("✅ Exa 深度语义信息注入完成")

        # -----------------------------------------------------
        # 最终汇总
        # -----------------------------------------------------
        final_count = len(successful_contents)
        if final_count > 0:
            logger.info(f"🏁 全链路级联检索完成，共获取 {final_count} 组有效信息模块")
            return "\n\n".join(successful_contents)
        else:
            logger.info("❌ 级联检索全线溃败：未获取到任何有效内容")
            return "联网检索未获取到有效内容。"

    def _tavily_full_backup(self, query: str, max_results: int):
        """
        最后的堡垒：Tavily 商业全量搜索备份方案。
        
        Returns:
            str: Tavily 返回的带正文的 Markdown 列表
        """
        logger.info("🛟 触发 Tavily 全量搜索备份...")
        # 开启 include_raw_content
        t_data = self._tavily_request("search", {
            "query": query, 
            "max_results": max_results, 
            "include_raw_content": True
        })

        # 防护装甲 1：彻底的类型与存在性校验
        if not t_data or not isinstance(t_data, dict):
            logger.warning("⚠️ Tavily 备份请求失败或返回格式错误")
            return "联网检索模块暂时不可用（API 响应异常）。"
            
        results = t_data.get('results')
        if not results or not isinstance(results, list) or len(results) == 0:
            logger.warning("⚠️ Tavily 请求成功，但未搜索到任何结果")
            return "已连接网络，但未搜索到与关键词相关的公开信息。"
        
        # 防护装甲 2：安全地从字典中取值
        valid_contents = []
        for i in results:
            if isinstance(i, dict):
                title = i.get('title', '参考来源')
                url = i.get('url', '#')
                # 有些页面 Tavily 也抓不到 raw_content，此时退而求其次用 snippet
                content = i.get('raw_content') or i.get('content') or ""
                if content:
                    valid_contents.append(self._format_reference(title, url, content))
                    
        return "\n\n".join(valid_contents) if valid_contents else "未提取到正文内容。"