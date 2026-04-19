@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

:: 核心绝杀：无论在哪执行，强制将工作目录切换到脚本所在的项目根目录
cd /d "%~dp0"

echo ========================================================
echo     🪐 Gemma 4 + MinerU 双环境流水线一键初始化
echo ========================================================
echo.

:: 1. 交互式配置输入
set PYTHON_CMD=python
set /p USER_PYTHON="请输入 Python 命令 (直接回车默认使用: python): "
if not "!USER_PYTHON!"=="" set PYTHON_CMD=!USER_PYTHON!

:: 依赖文件路径（统一指向 dependencies/ 目录）
set REQ_GEMMA=dependencies\requirements_gemma.txt
set /p USER_REQ_G="1. 请输入 Gemma 4 依赖路径 (默认: ./dependencies/requirements_gemma.txt): "
if not "!USER_REQ_G!"=="" set REQ_GEMMA=!USER_REQ_G!

set REQ_MINERU=dependencies\requirements_mineru.txt
set /p USER_REQ_M="2. 请输入 MinerU 依赖路径 (默认: ./dependencies/requirements_mineru.txt): "
if not "!USER_REQ_M!"=="" set REQ_MINERU=!USER_REQ_M!

:: 离线 .whl 包统一存放在 dependencies/ 目录
set DEPS_DIR=dependencies

set PYPI_MIRROR=https://pypi.tuna.tsinghua.edu.cn/simple
echo 默认镜像源: !PYPI_MIRROR!
echo.

:: 2. 检查 Python
!PYTHON_CMD! --version >nul 2>nul
if !errorlevel! neq 0 (
    echo [错误] 未检测到命令: !PYTHON_CMD!
    pause & exit /b
)

:: ==========================================================
:: 阶段一：部署 Gemma 4 主业务环境
:: ==========================================================
echo [阶段 1/2] 正在构建 Gemma 4 业务环境 [venv_gemma]...
if not exist "venv_gemma\Scripts\activate" (
    !PYTHON_CMD! -m venv venv_gemma
)
call "venv_gemma\Scripts\activate"
python -m pip install --upgrade pip -i !PYPI_MIRROR! >nul

:: 从 dependencies/ 目录安装 Gemma 专用 torch whl（文件名含 cu121）
if exist "!DEPS_DIR!\torch*cu121*.whl" (
    echo 探测到本地包，正在安装 Gemma 专用 Torch_cu121...
    for %%f in ("!DEPS_DIR!\torch*cu121*.whl" "!DEPS_DIR!\torchvision*cu121*.whl") do (
        if exist "%%f" pip install "%%f" -i !PYPI_MIRROR!
    )
) else if exist "!DEPS_DIR!\torch*.whl" (
    echo 探测到本地包，正在安装 Gemma Torch_cu121...
    for %%f in ("!DEPS_DIR!\torch*.whl" "!DEPS_DIR!\torchvision*.whl") do (
        if exist "%%f" pip install "%%f" -i !PYPI_MIRROR!
    )
)
pip install -r "!REQ_GEMMA!" -i !PYPI_MIRROR!
:: 部署完必须退出，防止环境污染
call deactivate
echo ✅ Gemma 4 环境部署完毕！
echo.

:: ==========================================================
:: 阶段二：部署 MinerU 解析环境
:: ==========================================================
echo [阶段 2/2] 正在构建 MinerU 解析环境 [venv_mineru]...
if not exist "venv_mineru\Scripts\activate" (
    !PYTHON_CMD! -m venv venv_mineru
)
call "venv_mineru\Scripts\activate"
python -m pip install --upgrade pip -i !PYPI_MIRROR! >nul

:: 从 dependencies/ 目录安装 MinerU 专用 torch whl（文件名含 cu126）
if exist "!DEPS_DIR!\torch*cu126*.whl" (
    echo 探测到本地包，正在安装 MinerU 专用 Torch_cu126...
    for %%f in ("!DEPS_DIR!\torch*cu126*.whl" "!DEPS_DIR!\torchvision*cu126*.whl") do (
        if exist "%%f" pip install "%%f" -i !PYPI_MIRROR!
    )
) else if exist "!DEPS_DIR!\torch*.whl" (
    echo 探测到本地包，正在安装 MinerU Torch_cu126...
    for %%f in ("!DEPS_DIR!\torch*.whl" "!DEPS_DIR!\torchvision*.whl") do (
        if exist "%%f" pip install "%%f" -i !PYPI_MIRROR!
    )
)
pip install -r "!REQ_MINERU!" -i !PYPI_MIRROR!
:: 部署完彻底退出 MinerU 环境
call deactivate
echo ✅ MinerU 环境部署完毕！
echo.

:: ==========================================================
:: 终局：自动拉起业务主线（仅激活 venv_gemma）
:: ==========================================================
echo ========================================================
echo  🚀 全部环境初始化成功！即将自动为您启动系统...
echo ========================================================
echo.

:: 双重保险：确保当前路径依然在根目录
cd /d "%~dp0"

echo 正在激活主业务环境 (venv_gemma)...
call "venv_gemma\Scripts\activate"

echo 正在拉起 Streamlit Web UI...
streamlit run src\web_ui.py

pause