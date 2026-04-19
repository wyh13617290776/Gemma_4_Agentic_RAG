import logging
import re
from typing import List, Set

logger = logging.getLogger(__name__)

class ReflectionEngine:
    """
    V2.0 专家级反思引擎：具备反爬清洗、语义覆盖评估与增量去重能力。
    """
    def __init__(self, min_urls: int, min_length: int = 400, k_coverage: float = 0.4):
        self.min_urls = min_urls
        self.min_length = min_length
        self.k_coverage = k_coverage
        
        # 🛡️ 黑名单矩阵：拦截假装成功的反爬虫/付费墙/验证码页面
        self.garbage_patterns = [
            r"verify you are human", r"checking your browser", r"enable javascript",
            r"access denied", r"403 forbidden", r"404 not found", r"page not found",
            r"subscribe to read", r"login to continue", r"请输入验证码", r"访问受限",
            r"滑动完成验证", r"安全拦截", r"机器人验证", r"继续阅读需解锁"
        ]

    def evaluate_quality(self, content: str, sub_queries: List[str]) -> bool:
        """
        质量脱水评估：长度 -> 黑名单 -> 语义覆盖 (复合意图修复版)
        """
        if not content:
            return False
            
        content_lower = content.lower()
        
        # 1. 物理脱水：长度校验
        if len(content_lower) < self.min_length:
            logger.debug(f"📍 质量拦截：内容过短 ({len(content_lower)} < {self.min_length})")
            return False

        # 2. 毒性清洗：黑名单特征探测
        for pattern in self.garbage_patterns:
            if re.search(pattern, content_lower):
                logger.info(f"📍 质量拦截：命中无效/拦截页面特征 [{pattern}]")
                return False

        # 3. 语义雷达 (Bug修复：独立子意图校验)
        # 如果没有提供 sub_queries，只要过了长度和黑名单就放行
        if not sub_queries:
            return True

        passed_coverage = False
        max_cov = 0.0
        
        # 遍历每一个提取后的子查询（例如：['安宫牛黄丸 研报', '安宫牛黄丸 淘宝']）
        for sq in sub_queries:
            # 将当前子查询打散为原子词
            words = set(sq.lower().split())
            if not words:
                continue
                
            # 计算当前子查询在这个网页中的命中率
            hits = sum(1 for word in words if word in content_lower)
            coverage = hits / len(words)
            max_cov = max(max_cov, coverage)
            
            # 💡 只要该网页在任意一个子领域达标，即视为优质专科信息，立刻放行
            if coverage >= self.k_coverage:
                passed_coverage = True
                break 
                
        if not passed_coverage:
            logger.info(f"📍 质量拦截：最高细分覆盖度过低 ({max_cov:.2f} < {self.k_coverage})")
            return False
            
        return True

    def is_redundant(self, new_content: str, existing_contents: List[str], threshold: float = 0.8) -> bool:
        """
        增量去重 (Deduplication)：使用轻量级 Jaccard 相似度防止输入雷同的文章 (如新闻通稿)。
        如果新文章与已收录文章相似度 > 80%，则视为冗余。
        """
        if not existing_contents:
            return False
            
        # 提取新文章的字符集合（以每 3 个字符为特征窗口，模拟 n-gram）
        def get_features(text: str) -> Set[str]:
            text = text.replace(" ", "").replace("\n", "")
            return set(text[i:i+3] for i in range(len(text)-2))
            
        features_new = get_features(new_content)
        if not features_new: return False
        
        for existing in existing_contents:
            features_old = get_features(existing)
            if not features_old: continue
            
            # 计算交并比 (Intersection over Union)
            intersection = len(features_new.intersection(features_old))
            union = len(features_new.union(features_old))
            sim_score = intersection / union if union > 0 else 0
            
            if sim_score > threshold:
                logger.info(f"📍 去重拦截：发现高度雷同信息 (相似度 {sim_score:.2f})")
                return True
                
        return False

    def is_saturated(self, current_count: int) -> bool:
        """
        判定当前信息池是否达到“饱和”状态。
        注意：入参改为 count，解耦具体数据结构。
        """
        is_ok = current_count >= self.min_urls
        status = "✅ 饱和" if is_ok else "⚠️ 匮乏"
        logger.info(f"📊 饱和度监测：当前纯净信息 {current_count}/{self.min_urls} ({status})")
        return is_ok