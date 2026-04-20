"""
SessionState：系统运行时的完整状态数据类。

设计原则：
- 纯数据定义，不依赖 Streamlit 或任何 UI 框架
- 可直接在 FastAPI 中用普通 dict 构造
- to_dict() 保证与 st.session_state.update() 的无缝对接
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SessionState:
    # ——— 模型选择 ———
    active_model: str = "gemma-4-local"
    prev_selected_model: str = ""
    last_synced_model: str = ""

    # ——— 功能开关 ———
    enable_web_search: bool = True
    enable_thinking: bool = True
    enable_multimodal: bool = True
    supports_thinking: bool = False
    is_vlm: bool = False

    # ——— 生成参数快照（当前激活模型） ———
    cur_temp: float = 1.0
    cur_top_p: float = 0.95
    cur_top_k: int = 64
    cur_max_tokens: int = 4096

    # ——— 对话与记忆 ———
    messages: list = field(default_factory=list)
    current_summary: str = ""

    # ——— UI 交互状态 ———
    uploader_key: int = 1
    last_audio_hash: Optional[str] = None
    suggested_questions: list = field(default_factory=list)
    trigger_followup: Optional[str] = None
    flash_model_index: int = 0

    # ——— 本地服务状态 ———
    llm_started_this_session: bool = False
    env_check_passed: bool = False
    auto_check_triggered: bool = False
    disable_chat_input: bool = False

    def to_dict(self) -> dict:
        """序列化为普通 dict，用于 st.session_state.update() 或 FastAPI"""
        import dataclasses
        return dataclasses.asdict(self)
