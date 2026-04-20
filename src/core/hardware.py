import os
import sys
import time
import httpx
import socket
import psutil
import subprocess
import gc
import torch
from core.config import CFG, ROUTER

# 新增：尝试加载 NVIDIA 显卡探针库
try:
    import pynvml
    pynvml.nvmlInit()
    HAS_NVML = True
except Exception:
    HAS_NVML = False

def is_port_in_use(port, host='127.0.0.1'):
    """检测指定端口是否被占用"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex((host, port)) == 0

class HardwareManager:
    LOCAL_SERVER = ROUTER["models"]["gemma-4-local"]["server_info"]
    PORT = int(CFG["llm_server"]["port"])
    

    @staticmethod
    def get_llm_cmd():
        server_exe = CFG["llm_server"]["server_exe_path"] 
        
        cmd = [
            server_exe,
            "--model", HardwareManager.LOCAL_SERVER["model_path"],
            "--ctx-size", str(HardwareManager.LOCAL_SERVER["n_ctx"]),
            "--n-gpu-layers", str(HardwareManager.LOCAL_SERVER["n_gpu_layers"]),
            "--threads", str(HardwareManager.LOCAL_SERVER["n_threads"]),
            "--host", CFG["llm_server"]["host"],
            "--port", str(HardwareManager.PORT),
            "--special",  
        ]
        
        mmproj = HardwareManager.LOCAL_SERVER.get("mmproj_path", "")
        if mmproj and os.path.exists(mmproj):
            cmd.extend(["--mmproj", mmproj])
            
        return cmd

    @staticmethod
    def stop_llm_service():
        """停止大模型服务并释放显存 (极致防卡死版)"""
        killed = False
        
        # 1. 优先使用 Windows 原生命令强杀，极其迅速且不卡顿
        if os.name == 'nt':
            try:
                # 屏蔽输出，直接强杀 llama-server.exe 进程树
                subprocess.run(['taskkill', '/F', '/T', '/IM', 'llama-server.exe'], 
                               capture_output=True, check=False)
                killed = True
            except:
                pass
                
        # 2. 兜底方案：按名字寻找，绝对不要使用 proc.connections()！
        for proc in psutil.process_iter(['name']):
            try:
                # 只要名字匹配直接杀，不查端口
                if proc.info['name'] and 'llama-server' in proc.info['name']:
                    proc.kill()
                    killed = True
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
                
        # 3. 智能等待：最多等 3 秒，只要端口一释放立刻跳出，绝不傻等
        if killed:
            max_retries = 3
            for _ in range(max_retries):
                if not is_port_in_use(HardwareManager.PORT):
                    break # 端口已释放，立刻跳出！
                time.sleep(1)
        
        # 4. 顺手清理 PyTorch 显存碎片，保证 MinerU 有干净的环境
        HardwareManager.free_vram()
        
        return killed

    @staticmethod
    def start_llm_service():
        HardwareManager.stop_llm_service() 
        cmd = HardwareManager.get_llm_cmd()
        
        if os.name == 'nt':
            keep_open_cmd = ["cmd.exe", "/k"] + cmd
            subprocess.Popen(keep_open_cmd, creationflags=subprocess.CREATE_NEW_CONSOLE)
        else:
            subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr)
        
        start_time = time.time()
        timeout = 60 
        while time.time() - start_time < timeout:
            if is_port_in_use(HardwareManager.PORT):
                time.sleep(1) 
                return True
            time.sleep(2) 
            
        raise Exception("大模型服务启动超时或崩溃，请查看弹出的黑色控制台报错信息！")

    @staticmethod
    def free_vram():
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    
    # 新增：系统级全局硬件探针
    @staticmethod
    def get_system_metrics():
        """获取 CPU、内存、GPU、显存的实时全局使用率"""
        # 1. 基础资源探测 (非阻塞方式)
        metrics = {
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "ram_percent": psutil.virtual_memory().percent,
            "has_gpu": False,
            "gpu_util": 0,
            "vram_used_gb": 0.0,
            "vram_total_gb": 0.0,
            "vram_percent": 0.0
        }

        # 2. NVIDIA GPU 深度探测 (能抓到 llama.cpp 进程的显存占用)
        if HAS_NVML:
            try:
                # 默认获取第一块显卡 (索引 0)
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                util_info = pynvml.nvmlDeviceGetUtilizationRates(handle)

                metrics["has_gpu"] = True
                metrics["gpu_util"] = util_info.gpu
                metrics["vram_used_gb"] = mem_info.used / (1024**3)
                metrics["vram_total_gb"] = mem_info.total / (1024**3)
                metrics["vram_percent"] = (mem_info.used / mem_info.total) * 100
            except Exception as e:
                pass # 静默处理，防止显卡休眠等异常导致应用崩溃

        return metrics
    
    # =========================================================
    # 新增：云端 API 连通性探活 (从 web_ui.py 抽离)
    # =========================================================
    @staticmethod
    def check_cloud_api_health(protocol: str, base_url: str, key_env: str, model_id: str) -> str:
        api_key = os.getenv(key_env, "")
        if not api_key or api_key == "sk-dummy" or len(api_key) < 10:
            return "🔴 API Key 未配置或无效"
        
        try:
            if protocol == "openai":
                if base_url and "dashscope.aliyuncs.com" in base_url:
                    test_url = f"{base_url.rstrip('/')}/chat/completions"
                    payload = {
                        "model": model_id, 
                        "messages": [{"role": "user", "content": "hi"}], 
                        "max_tokens": 1
                    }
                    res = httpx.post(test_url, headers={"Authorization": f"Bearer {api_key}"}, json=payload, timeout=10.0)
                else:
                    url = f"{base_url.rstrip('/')}/models" if base_url else "https://api.openai.com/v1/models"
                    res = httpx.get(url, headers={"Authorization": f"Bearer {api_key}"}, timeout=5.0)
                
                if res.status_code == 200: return "🟢 在线 (验证通过)"
                elif res.status_code == 401: return "🔴 鉴权失败 (Key 错误)"
                elif res.status_code == 400: return "🟡 参数错误 (HTTP 400)"
                elif res.status_code == 429: return "🟡 触发限流 (频率过高)"
                elif res.status_code >= 500: return f"🔴 云端异常 (HTTP {res.status_code})"
                else: return f"🟡 状态异常 (HTTP {res.status_code})"

        except httpx.ConnectTimeout: return "🔴 连接超时 (请检查网络)"
        except httpx.ConnectError: return "🔴 网络不可达 (请检查代理或域名)"
        except Exception as e: 
            print({str(e)}) 
            return f"🔴 探测异常 ({str(e)[:20]}...)"
            
        return "⚪ 状态未知"

    # =========================================================
    # 新增：本地环境深度安检底层扫描 (从 web_ui.py 抽离)
    # =========================================================
    @staticmethod
    def validate_local_env(cfg: dict, server_info: dict) -> list:
        """扫描本地硬盘，返回缺失的依赖清单列表（为空则代表安检通过）"""
        missing_deps = []
        
        # 1. 检查基础大语言模型权重与多模态权重
        model_path = server_info.get("model_path", "")
        if not model_path or not os.path.exists(model_path):
            missing_deps.append("基础模型权重 (.gguf)")
        mmproj_path = server_info.get("mmproj_path", "")
        if not mmproj_path or not os.path.exists(mmproj_path):
            missing_deps.append("多模态视觉权重 (mmproj-*.gguf)")
        
        # 2. 探测推理引擎
        search_root = os.getcwd() 
        found_exe, found_dlls = False, False
        exe_path_from_cfg = cfg.get("llm_server", {}).get("server_exe_path", "")
        
        if exe_path_from_cfg and os.path.exists(exe_path_from_cfg):
            found_exe = True
            target_dir = os.path.dirname(exe_path_from_cfg)
            if any(f.lower().endswith(('.dll')) for f in os.listdir(target_dir)):
                found_dlls = True
        
        if not found_exe:
            for root, dirs, files in os.walk(search_root):
                if "llama" in root.lower():
                    if "llama-server.exe" in files:
                        found_exe = True
                        if any(f.lower().endswith(('.dll')) for f in files):
                            found_dlls = True
                        break
        
        if not found_exe: missing_deps.append("推理核心引擎 (缺少 llama-server.exe)")
        elif not found_dlls: missing_deps.append("引擎运行依赖 (缺失 *.dll 库文件)")

        return missing_deps