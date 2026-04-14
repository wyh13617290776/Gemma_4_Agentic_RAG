import os
import yaml
import json
from dotenv import load_dotenv

# 动态获取项目根目录，彻底告别相对路径报错
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 自动加载项目根目录下的 .env 文件
load_dotenv(os.path.join(BASE_DIR, ".env"))

# 精准定位 config 文件夹下的配置文件
CONFIG_PATH = os.path.join(BASE_DIR, "config", "config.yaml")
SECRETS_PATH = os.path.join(BASE_DIR, "config", "secrets.yaml")
PROMPTS_PATH = os.path.join(BASE_DIR, "config", "prompts.json")
TOOLS_PATH = os.path.join(BASE_DIR, "config", "tools.json")

def load_yaml_safe(path):
    """👑 安全加载 YAML 的辅助函数：如果文件不存在，不会报错退出，而是返回空字典"""
    if not os.path.exists(path):
        print(f"⚠️ 提示: 配置文件未找到 -> {path}")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def load_config():
    # 1. 分别安全读取主配置和私密配置
    main_cfg = load_yaml_safe(CONFIG_PATH)
    secrets_cfg = load_yaml_safe(SECRETS_PATH)
    
    # 2. 👑 核心魔法：将 secrets 里的配置覆盖/合并到主配置中
    config = main_cfg.copy()
    config.update(secrets_cfg)

    # ==========================================
    # 2. 环境变量 ( .env ) 强制覆盖逻辑
    # ==========================================
    
    # [A] 覆盖 API Keys
    if "api_keys" not in config: 
        config["api_keys"] = {}
    for key in ["SERPER_API_KEY", "TAVILY_API_KEY", "EXA_API_KEY"]:
        env_val = os.getenv(key)
        if env_val:  # 只有当 .env 中填了值，才覆盖 yaml 里的空值
            config["api_keys"][key.lower()] = env_val

    # [B] 覆盖 网络代理配置
    if "network" not in config: 
        config["network"] = {}
        
    proxy_url = os.getenv("PROXY_URL")
    # 注意：这里用 is not None，是为了允许用户在 .env 中写 PROXY_URL="" 来强制清空代理
    if proxy_url is not None:
        config["network"]["proxy_url"] = proxy_url

    no_proxy = os.getenv("NO_PROXY")
    if no_proxy:
        config["network"]["no_proxy"] = no_proxy

    # [C] 覆盖 运行端口配置 (预留扩展)
    if "server" not in config:
        config["server"] = {}
    port = os.getenv("STREAMLIT_SERVER_PORT")
    if port:
        config["server"]["port"] = port

    # 👑 核心修复：将 server_exe_path 加入自动补全列表
    path_fields = [
        (['llm_server', 'model_path']),
        (['llm_server', 'mmproj_path']),
        (['llm_server', 'server_exe_path']),
        (['rag', 'reranker', 'model_path']),
        (['embedding', 'model_path']),
        (['mineru', 'exe_path'])
    ]

    for path_map in path_fields:
        curr = config
        for key in path_map[:-1]:
            curr = curr.get(key, {})
        
        last_key = path_map[-1]
        raw_path = curr.get(last_key)

        # 自动将相对路径拼接为基于项目根目录的绝对路径
        if raw_path and not os.path.isabs(raw_path):
            curr[last_key] = os.path.join(BASE_DIR, raw_path).replace("\\", "/")

    # 3. 读取现有的 JSON 配置
    with open(PROMPTS_PATH, "r", encoding="utf-8") as f:
        prompts = json.load(f)
    with open(TOOLS_PATH, "r", encoding="utf-8") as f:
        tools = json.load(f)
        
    return config, prompts, tools

def save_config(current_cfg):
    """👑 新增：将内存中的最新配置固化保存到本地 config.yaml 中"""
    # 安全屏障：读取当前的私钥配置，在保存时将私钥字段剥离，防止密钥被意外泄露到主配置文件中
    secrets_cfg = load_yaml_safe(SECRETS_PATH)
    safe_cfg = current_cfg.copy()
    for key in secrets_cfg.keys():
        if key in safe_cfg:
            del safe_cfg[key]
            
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        yaml.dump(safe_cfg, f, sort_keys=False, allow_unicode=True)

# 模块加载时自动解析并暴露出这三个全局变量
CFG, PROMPTS, TOOLS = load_config()