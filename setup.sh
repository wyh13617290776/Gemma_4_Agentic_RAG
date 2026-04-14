#!/bin/bash
set -e 

# 👑 核心绝杀：无论在哪执行，强制将工作目录切换到脚本所在的项目根目录
cd "$(dirname "$0")"

echo "========================================================"
echo "    🪐 Gemma 4 + MinerU 双环境流水线一键初始化"
echo "========================================================"
echo ""

# 1. 交互式配置输入
read -p "请输入 Python 命令 (默认: python3): " USER_PYTHON
PYTHON_CMD=${USER_PYTHON:-python3}

read -p "1. 请输入 Gemma 4 依赖路径 (默认: ./requirements.txt): " USER_REQ_G
REQ_GEMMA=${USER_REQ_G:-requirements.txt}

read -p "2. 请输入 MinerU 依赖路径 (默认: ./tools/mineru_env/requirements.txt): " USER_REQ_M
REQ_MINERU=${USER_REQ_M:-tools/mineru_env/requirements.txt}

PYPI_MIRROR="https://pypi.tuna.tsinghua.edu.cn/simple"

# ==========================================================
# 阶段一：Gemma 4
# ==========================================================
echo "[阶段 1/2] 构建 venv_gemma..."
[ ! -f "venv_gemma/bin/activate" ] && $PYTHON_CMD -m venv venv_gemma
source venv_gemma/bin/activate
pip install --upgrade pip -i $PYPI_MIRROR

DIR_G=$(dirname "$REQ_GEMMA")
if ls "$DIR_G"/torch*.whl 1> /dev/null 2>&1; then
    echo "安装本地 Gemma Torch..."
    pip install "$DIR_G"/torch*.whl "$DIR_G"/torchvision*.whl -i $PYPI_MIRROR
fi
pip install -r "$REQ_GEMMA" -i $PYPI_MIRROR
deactivate

# ==========================================================
# 阶段二：MinerU
# ==========================================================
echo "[阶段 2/2] 构建 venv_mineru..."
[ ! -f "venv_mineru/bin/activate" ] && $PYTHON_CMD -m venv venv_mineru
source venv_mineru/bin/activate
pip install --upgrade pip -i $PYPI_MIRROR

DIR_M=$(dirname "$REQ_MINERU")
if ls "$DIR_M"/torch*.whl 1> /dev/null 2>&1; then
    echo "安装本地 MinerU Torch..."
    pip install "$DIR_M"/torch*.whl "$DIR_M"/torchvision*.whl -i $PYPI_MIRROR
fi
pip install -r "$REQ_MINERU" -i $PYPI_MIRROR
deactivate

# ==========================================================
# 终局：自动拉起业务主线
# ==========================================================
echo "========================================================"
echo " 🚀 全部环境初始化成功！即将自动为您启动系统..."
echo "========================================================"
echo ""

# 双重保险：确保当前路径依然在根目录
cd "$(dirname "$0")"

echo "正在激活主业务环境 (venv_gemma)..."
source venv_gemma/bin/activate

echo "正在拉起 Streamlit Web UI..."
streamlit run web_ui.py