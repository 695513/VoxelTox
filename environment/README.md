# 环境记录

检查日期：2026-11-11。主要来源为服务器主 Python 环境 `/root/miniconda3/bin/python`；另核对本地 `C:\Anaconda\envs\voxbind\python.exe`，补充预处理库版本。

- 系统：Linux（Ubuntu 22.04）。
- Python：3.12.3。
- PyTorch：2.5.1+cu124，构建使用 CUDA 12.4。
- TorchVision：0.20.1+cu124，已安装，但当前项目代码没有直接使用。
- 检查时 GPU 数量为 0，尚未验证 GPU 训练。

## 文件用途

- `../requirements.txt`：按当前项目代码整理的依赖；主要版本来自服务器主环境，`pyuul` 版本来自本地 `voxbind`。
- `server-packages.txt`：主环境中全部 196 个已安装 Python 分发包的名称和版本。用于对照，不建议将整份清单直接用于安装项目。
- `server-environment.json`：完整库清单、Python 版本及平台信息。
- `server-import-check.json`：核心库的实际导入检查结果。
- `local-voxbind-packages.json`：本地 `voxbind` 中两个预处理依赖的版本及导入检查记录。

## 安装方式

在具有兼容 NVIDIA 驱动的 Linux 环境中，可先创建 Python 3.12.3 环境，再安装对应的 PyTorch 构建：

```bash
conda create -n bvcnn python=3.12.3
conda activate bvcnn
python -m pip install torch==2.5.1+cu124 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
```

上述依赖来自当前环境审计，尚未在全新环境中完成安装及端到端运行验证。

## 预处理依赖

当前主环境中没有安装 `pyuul` 和 `timeout-decorator`：

- `pyuul`：`preprocessing/downstream_voxelize.py`、`preprocessing/pretrain_voxelize.py` 和备用脚本 `legacy/voxel_preprocessor.py` 均使用它生成体素。本地 `voxbind`（Python 3.12.11）中已安装 **0.4.2**，且 `pyuul.VolumeMaker` 实际导入通过，已将此版本固定到 `requirements.txt`。
- `timeout-decorator`：仅备用脚本 `legacy/voxel_preprocessor.py` 使用它，用于设置单分子体素化的 8 秒超时。当前主流程无需安装，因此不列入 `requirements.txt`。服务器主环境和本地 `voxbind` 中均未找到该库；如需运行备用脚本，可另行执行 `python -m pip install timeout-decorator`，版本仍需确认。

`pyuul==0.4.2` 与服务器依赖组合的全新安装及完整预处理尚未验证。

已有体素缓存的训练入口不直接依赖这两个预处理库。ChemBERTa 使用 `DeepChem/ChemBERTa-77M-MLM`，首次运行需能获取该模型，或准备本地模型文件。
