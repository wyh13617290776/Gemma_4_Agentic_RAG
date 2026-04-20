from core.config import CFG


class ShortTermMemory:
    """
    短期记忆：维护固定长度的滑动对话窗口。
    不持久化，进程重启后清空。

    设计约定：
    - _messages 直接引用外部 state_container["messages"]，保证单一数据源
    - get_window() 返回最近 max_window 轮（即 max_window * 2 条消息）
    """

    def __init__(self, max_window: int = None):
        self.max_window = max_window or CFG["memory"].get("max_window", 3)
        self._messages: list = []

    def add(self, role: str, content, thought: str = ""):
        self._messages.append({"role": role, "content": content, "thought": thought})

    def get_window(self) -> list:
        """返回最近 max_window 轮对话（user + assistant 各算一条）"""
        return self._messages[-(self.max_window * 2):]

    def get_all(self) -> list:
        return self._messages

    def update_last(self, thought: str, content):
        """更新最后一条 assistant 消息（流式输出完成后调用）"""
        if self._messages and self._messages[-1]["role"] == "assistant":
            self._messages[-1].update({"thought": thought, "content": content})

    def clear(self):
        self._messages.clear()

    def __len__(self):
        return len(self._messages)
