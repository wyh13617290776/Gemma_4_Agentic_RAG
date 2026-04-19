## 📦 离线依赖包说明 (Offline Wheel Dependencies)

本目录统一存放以下文件：
- **`requirements_gemma.txt`** — `venv_gemma` 环境的完整 Python 依赖清单
- **`requirements_mineru.txt`** — `venv_mineru` 环境的完整 Python 依赖清单
- **PyTorch 离线 `.whl` 安装包** — GPU 版本 Torch，需手动指定 CUDA 版本下载后放入此目录

---

---

### 🔵 venv_gemma 环境（Gemma 4 主业务环境）

**推荐版本**（对应 CUDA 12.1）：
- `torch-2.5.1+cu121-cp310-cp310-win_amd64.whl`
- `torchvision-0.20.1+cu121-cp310-cp310-win_amd64.whl`

> 前往 [PyTorch 官网](https://download-r2.pytorch.org/whl/cu121) 下载对应 CUDA 版本的 `.whl` 文件，放置在本目录下。

---

### 🟠 venv_mineru 环境（MinerU 解析环境）

**推荐版本**（对应 CUDA 12.6）：
- `torch-2.10.0+cu126-cp310-cp310-win_amd64.whl`
- `torchvision-0.25.0+cu126-cp310-cp310-win_amd64.whl`

> 前往 [PyTorch 官网](https://download-r2.pytorch.org/whl/cu126) 下载对应 CUDA 版本的 `.whl` 文件，放置在本目录下。

---

### ⚙️ 安装逻辑说明

`setup.bat` / `setup.sh` 会自动扫描本目录：
- 文件名含 `gemma` 的 `.whl` → 安装到 `venv_gemma`
- 文件名含 `mineru` 的 `.whl` → 安装到 `venv_mineru`
- 普通 `torch*.whl` / `torchvision*.whl` → 分别按阶段安装

> ⚠️ 请确保两套 `.whl` 文件的 Python 版本（`cp310`）与您本地 Python 版本一致；CUDA 版本需与 `nvidia-smi` 输出一致。
