import os
import yaml
import json
from dotenv import load_dotenv

# 1. 动态获取项目根目录
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(BASE_DIR, ".env"))

# 2. 精准定位 config 文件夹下的所有配置文件
CONFIG_PATH = os.path.join(BASE_DIR, "config", "config.yaml")
PROMPTS_PATH = os.path.join(BASE_DIR, "config", "prompts.json")
TOOLS_PATH = os.path.join(BASE_DIR, "config", "tools.json")

# 新增：收拢模型路由与任务模板路径
ROUTER_PATH = os.path.join(BASE_DIR, "config", "model_router.yaml")
TASKS_PATH = os.path.join(BASE_DIR, "config", "llm_tasks.yaml")

def load_yaml_safe(path):
    """安全加载 YAML"""
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def load_all_configs():
    """一次性解析所有配置文件，并完成环境治理"""
    config = load_yaml_safe(CONFIG_PATH)
    router = load_yaml_safe(ROUTER_PATH)
    tasks = load_yaml_safe(TASKS_PATH)
    
    # [A] 直接从 .env 环境变量加载密钥
    if "api_keys" not in config: config["api_keys"] = {}
    for key in ["SERPER_API_KEY", "TAVILY_API_KEY", "EXA_API_KEY", "SILICONFLOW_API_KEY", "DASHSCOPE_API_KEY"]:
        env_val = os.getenv(key)
        # 统一转小写，跟 web_retriever.py 里的 get("exa_api_key") 对齐
        if env_val: config["api_keys"][key.lower()] = env_val

    # [B] 代理配置覆盖
    if "network" not in config: config["network"] = {}
    proxy_url = os.getenv("PROXY_URL")
    if proxy_url is not None: config["network"]["proxy_url"] = proxy_url

    # [C] 路径自动化补全逻辑
    path_fields = [
        (['llm_server', 'server_exe_path']), (['rag', 'reranker', 'model_path']),
        (['embedding', 'model_path']), (['mineru', 'exe_path'])
    ]
    for path_map in path_fields:
        curr = config
        for key in path_map[:-1]: curr = curr.get(key, {})
        last_key = path_map[-1]
        raw_path = curr.get(last_key)
        if raw_path and not os.path.isabs(raw_path):
            curr[last_key] = os.path.join(BASE_DIR, raw_path).replace("\\", "/")

    # [D] 读取 JSON
    with open(PROMPTS_PATH, "r", encoding="utf-8") as f:
        prompts = json.load(f)
    with open(TOOLS_PATH, "r", encoding="utf-8") as f:
        tools = json.load(f)
        
    return config, router, tasks, prompts, tools

# 🚀 全局变量暴露
CFG, ROUTER, TASKS, PROMPTS, TOOLS = load_all_configs()

# ==========================================
# 3. 统一配置管理接口
# ==========================================

def get_model_generation_params(model_name):
    """
    核心继承逻辑：获取当前模型的最优生成参数
    优先级：model_router(专属微调) > llm_tasks(系列模板) > config(全局默认)
    """
    # 1. 初始化全局默认值
    final_params = {
        "temperature": CFG.get("llm_generation", {}).get("temperature", 0.7),
        "top_p": CFG.get("llm_generation", {}).get("top_p", 0.95),
        "max_tokens": CFG.get("llm_generation", {}).get("max_tokens", 4096),
        "top_k": CFG.get("llm_generation", {}).get("top_k", 64)
    }

    model_info = ROUTER.get("models", {}).get(model_name, {})
    
    # 2. 尝试从 llm_tasks.yaml 加载系列模板
    model_type = model_info.get("type", "gemma")
    category_cfg = TASKS.get(f"llm_chat_{model_type}", {})
    if category_cfg:
        final_params.update({
            "temperature": category_cfg.get("temperature", final_params["temperature"]),
            "top_p": category_cfg.get("top_p", final_params["top_p"]),
            "max_tokens": category_cfg.get("max_tokens", final_params["max_tokens"])
        })
        if "extra_body" in category_cfg:
            final_params["top_k"] = category_cfg["extra_body"].get("top_k", final_params["top_k"])

    # 3. 尝试从 model_router.yaml 加载专属微调 (最高优先级)
    specific_cfg = model_info.get("generation_cfg", {})
    if specific_cfg:
        final_params.update(specific_cfg)
        
    return final_params

def save_sys_config(current_cfg):
    """
    保存系统配置 (RAG, 记忆)
    此函数会忽略 model_router 中的个性化参数，仅更新全局 config.yaml
    """
    safe_cfg = current_cfg.copy()
    # 1. 保存配置文件前，强行将动态注入的密钥块剔除
    if "api_keys" in safe_cfg:
        del safe_cfg["api_keys"]
    # 2. 写入文件 (此时 CFG["rag"] 和 CFG["memory"] 已经是 UI 最新的值)
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        yaml.dump(safe_cfg, f, sort_keys=False, allow_unicode=True)

def save_model_override(model_name, new_params):
    """
    防止并发冲突导致的文件损坏
    不再直接 dump 全局变量，而是先从硬盘拉取最新状态进行合并
    """
    # 1. 物理读取当前硬盘上的最新文件，确保基准数据准确
    fresh_router = load_yaml_safe(ROUTER_PATH) 
    
    # 2. 严谨校验结构，防止写入空数据
    if "models" not in fresh_router: fresh_router["models"] = {}
    if not model_name: return # 防御性编程：防止 model_name 为空导致产生空白项
    
    if model_name not in fresh_router["models"]: 
        fresh_router["models"][model_name] = {}
    
    # 3. 仅更新生成参数部分
    fresh_router["models"][model_name]["generation_cfg"] = new_params
    
    # 4. 执行写盘操作 (建议增加 flush 确保写入完整)
    with open(ROUTER_PATH, "w", encoding="utf-8") as f:
        yaml.dump(fresh_router, f, sort_keys=False, allow_unicode=True)
        
    # 5. 👑 关键：手动同步更新内存中的全局变量，让所有 Session 立即看到新配置
    global ROUTER
    ROUTER = fresh_router