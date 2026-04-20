from abc import ABC, abstractmethod


class BaseAgent(ABC):
    """
    智能体抽象基类。
    所有角色智能体（规划者/执行者/批评者）必须继承此类，实现 run() 方法。

    设计约定：
    - run() 接收一个状态字典，返回更新后的状态字典
    - 状态字典在智能体之间流转，由 GraphOrchestrator 统一调度
    - 子类可通过 __init__ 注入所需的引擎（RAGPipeline、LLMGateway 等）
    """

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    @abstractmethod
    async def run(self, state: dict, **kwargs) -> dict:
        """
        执行当前智能体的核心逻辑。

        Args:
            state: 跨节点流转的状态字典，包含 user_query、intents、rag_nodes 等字段
            **kwargs: 可选的运行时参数（如 enable_web_search、session_id 等）

        Returns:
            dict: 更新后的状态字典（原地修改或返回新字典均可）
        """
        ...

    def __repr__(self) -> str:
        return f"<Agent name={self.name!r} description={self.description!r}>"
