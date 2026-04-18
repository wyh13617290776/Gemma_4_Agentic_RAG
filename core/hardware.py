import os
import sys
import time
import socket
import psutil
import subprocess
import gc
import torch
from core.config import CFG, ROUTER

# 👑 新增：尝试加载 NVIDIA 显卡探针库
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
    
    # 👑 新增：系统级全局硬件探针
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