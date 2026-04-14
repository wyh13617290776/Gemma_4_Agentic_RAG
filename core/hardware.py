import os
import sys
import time
import socket
import psutil
import subprocess
import gc
import torch
from core.config import CFG

def is_port_in_use(port, host='127.0.0.1'):
    """检测指定端口是否被占用"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex((host, port)) == 0

class HardwareManager:
    PORT = int(CFG["llm_server"]["port"])

    @staticmethod
    def get_llm_cmd():
        server_exe = CFG["llm_server"]["server_exe_path"] 
        
        cmd = [
            server_exe,
            "--model", CFG["llm_server"]["model_path"],
            "--ctx-size", str(CFG["llm_server"]["n_ctx"]),
            "--n-gpu-layers", str(CFG["llm_server"]["n_gpu_layers"]),
            "--threads", str(CFG["llm_server"]["n_threads"]),
            "--host", CFG["llm_server"]["host"],
            "--port", str(HardwareManager.PORT),
            "--special",  
        ]
        
        mmproj = CFG["llm_server"].get("mmproj_path", "")
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