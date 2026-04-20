"""
ContextTracker：跨轮对话的上下文流转追踪器。

职责：
- 模型切换感知（参数同步、能力探针）
- 本地/云端模式判断
- 状态变更摘要（供 UI 层决定是否弹 toast、触发 rerun）

设计原则：
- 完全不包含任何 Streamlit 渲染代码
- 接收并操作 SessionState 对象，与 UI 解耦
"""
import logging
from core.config import ROUTER, get_model_generation_params
from .state import SessionState

logger = logging.getLogger("AgenticRAG")


class ContextTracker:
    def __init__(self, state: SessionState):
        self.state = state

    def sync_model(self, selected_model: str) -> dict:
        """
        当模型切换时，同步更新所有派生状态。

        Returns:
            dict: 变更摘要，供 UI 层决定后续行为：
                  - "model_changed" (bool): 是否发生了模型切换
                  - "is_vlm" (bool): 新模型是否支持视觉
                  - "had_media" (bool): 切换前是否有待发送的附件（由调用方传入）
        """
        model_info = ROUTER.get("models", {}).get(selected_model, {})
        changes = {}

        model_changed = selected_model != self.state.prev_selected_model
        params_need_sync = self.state.last_synced_model != selected_model

        # 1. 同步生成参数
        if params_need_sync:
            m_params = get_model_generation_params(selected_model)
            self.state.cur_temp = float(m_params.get("temperature", 1.0))
            self.state.cur_top_p = float(m_params.get("top_p", 0.95))
            self.state.cur_top_k = int(m_params.get("top_k", 64))
            self.state.cur_max_tokens = int(m_params.get("max_tokens", 4096))

            features = model_info.get("features", {})
            self.state.enable_multimodal = features.get("enable_multimodal", True)
            self.state.last_synced_model = selected_model
            logger.info(f"🔄 模型参数已同步: {selected_model}")

        # 2. 深度思考能力探针
        adapter = str(model_info.get("adapter", "")).lower().strip()
        self.state.supports_thinking = (
            adapter in ["deepseek", "local_think"]
            or bool(model_info.get("dynamic_thinking", False))
        )
        if not self.state.supports_thinking:
            self.state.enable_thinking = False
        elif model_changed:
            self.state.enable_thinking = model_info.get("features", {}).get("enable_thinking", True)

        # 3. 视觉多模态能力探针（VLM Probe）
        self.state.is_vlm = any(
            kw in selected_model.lower() for kw in ["vl", "qvq", "gemma", "vision"]
        )

        # 4. 模型切换时重置本地服务状态
        if model_changed:
            self.state.llm_started_this_session = False
            self.state.prev_selected_model = selected_model
            changes["model_changed"] = True
            changes["is_vlm"] = self.state.is_vlm
            logger.info(f"🔀 模型已切换: {selected_model} (VLM={self.state.is_vlm})")

        self.state.active_model = selected_model
        return changes

    def is_local_mode(self, model_name: str) -> bool:
        """判断指定模型是否为本地私有化部署模式"""
        model_info = ROUTER.get("models", {}).get(model_name, {})
        base_url = model_info.get("base_url", "")
        return (
            "127.0.0.1" in base_url
            or "localhost" in base_url
            or model_info.get("key_env") == "NONE"
        )

    def write_back(self, session_state):
        """
        将 SessionState 的最新值写回 Streamlit session_state。
        在每次 sync_model() 之后调用，保证 UI 状态与追踪器同步。
        """
        import dataclasses
        for k, v in dataclasses.asdict(self.state).items():
            # 只更新已存在的 key，避免覆盖 UI 组件的私有 key
            try:
                session_state[k] = v
            except Exception:
                pass
