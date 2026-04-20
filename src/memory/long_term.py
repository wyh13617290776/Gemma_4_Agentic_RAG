class LongTermMemory:
    """
    长期记忆：将超出短期窗口的历史以文本摘要形式持久化。

    当前实现：进程内存存储。
    扩展方向：可替换为 SQLite / Redis / Milvus 向量存储。
    """

    def __init__(self):
        self._summary: str = ""

    def store_summary(self, summary: str):
        """追加一段压缩后的摘要到长期记忆（保留历史脉络）"""
        if self._summary:
            self._summary += f"\n\n[后续摘要]\n{summary}"
        else:
            self._summary = summary

    def get_summary(self) -> str:
        return self._summary

    def has_summary(self) -> bool:
        return bool(self._summary.strip())

    def clear(self):
        self._summary = ""
