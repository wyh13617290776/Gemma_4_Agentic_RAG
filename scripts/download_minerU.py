import os
import sys
import subprocess
import importlib.util

def get_venv_python():
    """获取项目根目录下 venv_gemma 的解释器路径"""
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # 适配 Windows 和 Linux/Mac 的路径差异
    if sys.platform == "win32":
        venv_python = os.path.join(root_dir, "venv_gemma", "Scripts", "python.exe")
    else:
        venv_python = os.path.join(root_dir, "venv_gemma", "bin", "python")
        
    return venv_python

def restart_with_venv():
    """如果当前不在 venv_gemma 中，尝试使用该环境重启脚本"""
    venv_python = get_venv_python()
    current_python = sys.executable

    # 检查解释器路径是否指向 venv_gemma 目录
    if "venv_gemma" not in current_python:
        if os.path.exists(venv_python):
            print(f"🔄 检测到 venv_gemma 环境，正在自动切换解释器...")
            # 使用 venv 里的解释器重新运行当前脚本
            subprocess.call([venv_python] + sys.argv)
            sys.exit(0)
        else:
            print("❌ 错误: 未在根目录下找到 venv_gemma 文件夹。")
            print("💡 请确保您已经运行了 setup.bat/sh 完成了基础环境构建。")
            sys.exit(1)
    else:
        print(f"✅ 已成功运行在虚拟环境中: {os.path.dirname(current_python)}")

def check_dependencies():
    """在虚拟环境中检查核心库"""
    print("\n🔍 环境探针正在检查基础依赖...")
    core_deps = ['torch', 'torchvision']
    missing = [dep for dep in core_deps if importlib.util.find_spec(dep) is None]
    
    if missing:
        print(f"❌ 环境异常: 缺失核心库 {missing}")
        print("💡 请确认 setup 步骤中已正确安装本地 .whl 包。")
        sys.exit(1)
    print("✅ 核心底层依赖检查通过。")

def main():
    TUNA_MIRROR = "https://pypi.tuna.tsinghua.edu.cn/simple"

    print("="*60)
    print("🪐 Gemma 4 Agentic RAG - 模型自动化下载器")
    print("="*60)

    # 1. 自动环境重定向
    restart_with_venv()

    # 2. 检查环境完整性
    check_dependencies()

    print("\n⚠️  网络检查：请确保已关闭全局代理 (VPN) 以加速清华源下载。")
    input("👉 准备就绪，按 [Enter] 键开始下载...")

    # 3. 安装下载器
    print(f"\n⏳ 正在通过清华源安装/更新 modelscope...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "modelscope", 
            "-i", TUNA_MIRROR
        ], stdout=subprocess.DEVNULL)
    except Exception as e:
        print(f"❌ 安装 modelscope 失败: {e}")
        sys.exit(1)

    # 4. 执行下载
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    target_dir = os.path.join(root_dir, "models", "minerU")
    print(f"\n📁 目标路径: {target_dir}")
    print("🚀 正在从 ModelScope 拉取 OpenDataLab/PDF-Extract-Kit-1.0...")
    
    try:
        from modelscope import snapshot_download
        snapshot_download('OpenDataLab/PDF-Extract-Kit-1.0', local_dir=target_dir)
        print("\n🎉 [SUCCESS] 模型下载任务已完美达成！")
    except Exception as e:
        print(f"\n❌ 下载中断: {e}")

if __name__ == "__main__":
    main()