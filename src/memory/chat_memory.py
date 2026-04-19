import re
from core.config import CFG

def strip_thinking_from_history(text: str) -> str:
    """思考链剥离器"""
    if not text: return ""
    text = re.sub(r"<\|think\|>.*?</\|think\|>", "", text, flags=re.DOTALL)
    text = text.replace("<|think|>\n", "").replace("<|think|>", "").replace("</|think|>", "")
    text = text.replace("<turn|>", "").replace("<|turn|>", "").replace("<eos>", "")
    return text.strip()

class MemoryManager:
    def __init__(self, session_state, max_window=3):
        self.state = session_state
        self.max_window = max_window
        self.summary_threshold = CFG["memory"].get("summary_threshold", 5)
        
        if "messages" not in self.state:
            self.state.messages = []
        if "current_summary" not in self.state:
            self.state.current_summary = ""

    def get_ui_messages(self):
        return self.state.messages

    def add_user_message(self, content):
        self.state.messages.append({"role": "user", "content": content})

    def add_assistant_message(self, thought, content):
        self.state.messages.append({"role": "assistant", "thought": thought, "content": content})

    def update_last_message(self, thought, content):
        if self.state.messages and self.state.messages[-1]["role"] == "assistant":
            self.state.messages[-1]["thought"] = thought
            self.state.messages[-1]["content"] = content

    def get_llm_payload(self, current_user_content):
        """
        组装 Payload：[System Summary] + [Window History] + [Current Input]
        """
        payload = []
        
        # 1. 提取滑动窗口
        history_window = self.state.messages[-(self.max_window * 2):-1] 

        for msg in history_window:
            role = msg["role"]
            content = msg["content"]
            if role == "user" and isinstance(content, list):
                content = next((item["text"] for item in content if item["type"] == "text"), "")
            elif role == "assistant":
                content = strip_thinking_from_history(content)
            payload.append({"role": role, "content": content})
                
        payload.append({"role": "user", "content": current_user_content})
        return payload

    def need_summarize(self):
        """判断是否需要触发摘要压缩"""
        # 每当对话总轮数超过阈值，且是偶数轮（即一组对话结束）时触发
        total_turns = len(self.state.messages) // 2
        return total_turns >= self.summary_threshold

    def get_summary_prompt(self):
        """获取当前的摘要内容，用于拼接到 System Prompt"""
        if self.state.current_summary:
            return f"\n\n【前情提要（历史对话摘要）】：\n{self.state.current_summary}"
        return ""

    def update_summary(self, new_summary):
        """更新持久化摘要"""
        self.state.current_summary = new_summary