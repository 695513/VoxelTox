# VoxelTox: A Multi-Field 3D Voxel-Grid Framework for Molecular Toxicity Prediction

用于分子性质分类的三维多场体素模型。代码包括体素数据准备、体素重建与跨模态对齐预训练，以及七个下游任务的微调。预训练在随机选定的体素位置计算重建误差，当前实现不按该掩码遮挡编码器输入。

## 项目结构

```text
core.py                  模型、损失函数及数据读取
train.py                 七个下游任务的统一微调入口
pretrain/pretrain.py      从头预训练入口
preprocessing/           预训练和下游数据的体素化
configs/                 当前超参数的参考快照
data/raw/                下游原始 CSV
data/pretrain/           最终预训练 SDF 及原始/中间数据
data/cache/              下游体素缓存；预训练缓存生成后也放在这里
splits/scaffold/         固定骨架划分
weights/                 已提供的预训练及下游权重
results/                 历史日志和新实验输出
environment/             环境依赖来源与检查记录
docs/                    数据来源记录及整理模板
legacy/                  备用脚本、原始代码备份和历史文件
assets/                  项目展示图片的预留位置
```

## 环境安装

依赖主要根据已检查的 Linux 服务器主环境整理，`pyuul==0.4.2` 来自本地 `voxbind` 环境。详情见 [environment/README.md](environment/README.md)。

```bash
conda create -n bvcnn python=3.12.3
conda activate bvcnn
python -m pip install torch==2.5.1+cu124 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
```

当前依赖组合尚未在全新环境中完成安装与完整训练验证。首次使用 ChemBERTa 需要获取 `DeepChem/ChemBERTa-77M-MLM`；下游训练也可以通过 `--chemberta-path` 指定本地模型目录。

## 数据与权重

- 七个下游任务：`bbbp`、`bace`、`sider`、`clintox`、`ames`、`herg`、`dili`。
- 原始 CSV 位于 `data/raw/`，固定划分位于 `splits/scaffold/`；划分索引对应原始 CSV 的行顺序。
- 已有下游缓存位于 `data/cache/BVCNN-<任务名>/train、val、test/`。
- 最终预训练数据：`data/pretrain/pretrain_30w_no_ions.sdf`。
- 下游初始化权重：`weights/pretrain_model_epoch_23.pth`。
- 已有下游权重：`weights/downstream/`，文件对应关系见 [weights/README.md](weights/README.md)。

更多数据说明见 [data/README.md](data/README.md)。已有权重随代码仓库提供；大体积预训练数据和体素缓存通过 `.gitignore` 排除普通 Git 提交，正式发布时需补上独立下载链接。

## 运行

下面命令均从项目根目录运行。默认输入路径根据脚本位置定位到本项目，不依赖原服务器绝对路径。命令行传入的相对路径则相对于当前工作目录。

### 1. 使用已有缓存微调

```bash
python train.py --dataset bace --seed 42
```

将 `bace` 替换成其他支持的任务即可。默认读取本项目的缓存、固定划分和第 23 轮预训练权重。也可以显式指定路径：

```bash
python train.py --dataset bace --seed 42 --cache-dir data/cache/BVCNN --split-dir splits/scaffold --pretrained-path weights/pretrain_model_epoch_23.pth
```

`--cache-dir` 是目录前缀，程序会追加 `-bace`。默认输出到 `results/downstream/bace/seed_42/`；可通过 `--output-dir` 修改。训练期间会计算验证与测试指标，最佳权重按验证集 AUC 选择。重复使用同一输出目录会更新其中的最佳权重文件。

### 2. 从 CSV 重新生成下游缓存（可选）

已有缓存可以直接用于上一步。重新生成时执行：

```bash
python preprocessing/downstream_voxelize.py --dataset bace
```

支持 `--raw-dir`、`--split-dir`、`--cache-dir` 和 `--chunk-size`。该脚本会清空并重建目标任务缓存目录；需要保留原缓存时应指定另一个缓存前缀，例如 `--cache-dir data/cache/regenerated`。

### 3. 生成预训练缓存

```bash
python preprocessing/pretrain_voxelize.py
```

默认读取最终预训练 SDF，生成 `data/cache/pretrain/pretrain_chunk_*.npz`。可通过 `--sdf-path`、`--cache-dir`、`--chunk-size`、`--batch-size`、`--num-workers` 调整输入和处理规模。该脚本会清空并重建指定的预训练缓存目录。

### 4. 从头预训练

```bash
python pretrain/pretrain.py
```

默认使用 `data/cache/pretrain/`，从第 1 轮开始，训练 50 轮，输出到 `results/pretrain/`。可通过 `--cache-root`、`--save-dir`、`--epochs`、`--batch-size`、`--lr`、`--weight-decay`、`--val-chunks` 调整配置。例如：

```bash
python pretrain/pretrain.py --cache-root data/cache/pretrain --save-dir results/pretrain/run_01
```

这会生成新权重。下游训练默认使用已收集的第 23 轮权重；若要使用新训练权重，应通过 `train.py --pretrained-path` 明确指定。

## 配置、评估与发布状态

- 当前训练超参数仍以 `train.py` 和预训练入口中的定义为准；`configs/` 中 JSON 是供阅读和核对的快照，程序不会自动加载它们。
- 当前没有单独的 `evaluate.py`。`train.py` 包含训练期评估，但不提供只加载现有下游权重进行评估的入口。
- 目录整理没有修改模型结构、损失公式、掩码用法和现有实验超参数。原始代码位于 `legacy/original_sources/`。
- 现有权重与模型结构的匹配已检查；目录整理不代表已经复跑完整预训练、下游训练或复核论文 AUC。
- 论文信息、引用信息、代码许可证及数据/权重下载地址，需在正式发布时补充。
