import os
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = str(PROJECT_ROOT / "data" / "raw")
DEFAULT_SPLIT_DIR = str(PROJECT_ROOT / "splits" / "scaffold")
DEFAULT_CACHE_PREFIX = str(PROJECT_ROOT / "data" / "cache" / "BVCNN")
import numpy as np
import torch
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
from tqdm import tqdm
import random
import shutil

# 禁用RDKit日志
RDLogger.DisableLog('rdApp.*')

# 定义体素化核心超时函数
def voxelize_core(voxelizer, batch):
    return voxelizer.mol2vox(batch, num_channels=10)


# 体素化工具类
from pyuul import VolumeMaker


class Voxelizer(torch.nn.Module):
    def __init__(self, grid_dim=64, resolution=0.25, cubes_around=8,
                 device="cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__()
        self.grid_dim = grid_dim
        self.device = device
        self.resolution = resolution
        self.cubes_around = cubes_around
        self.vol_maker = VolumeMaker.Voxels(device=device, sparse=False)

    def mol2vox(self, batch: dict, num_channels=10) -> torch.Tensor:
        batch = self._add_dumb_coords(batch)
        batch["coords"] = batch["coords"].to(self.device)
        batch["radius"] = batch["radius"].to(self.device)
        batch["atoms_channel"] = batch["atoms_channel"].to(self.device)
        batch["electronegativity"] = batch["electronegativity"].to(self.device)
        batch["charge"] = batch["charge"].to(self.device)
        batch["scaffold_weight"] = batch["scaffold_weight"].to(self.device)

        # 生成四类体素特征
        voxel_coords = self.vol_maker(
            batch["coords"], batch["radius"], torch.zeros_like(batch["atoms_channel"]),
            resolution=self.resolution, cubes_around_atoms_dim=self.cubes_around,
            function="gaussian", numberchannels=num_channels
        )
        voxel_electronegativity = self.vol_maker(
            batch["coords"], batch["electronegativity"], batch["atoms_channel"],
            resolution=self.resolution, cubes_around_atoms_dim=self.cubes_around,
            function="gaussian", numberchannels=num_channels
        )
        voxel_charge = self.vol_maker(
            batch["coords"], batch["charge"], batch["atoms_channel"],
            resolution=self.resolution, cubes_around_atoms_dim=self.cubes_around,
            function="gaussian", numberchannels=num_channels
        )
        voxel_scaffold_radius = self.vol_maker(
            batch["coords"], batch["radius"] * batch["scaffold_weight"],
            batch["atoms_channel"], resolution=self.resolution,
            cubes_around_atoms_dim=self.cubes_around, function="gaussian", numberchannels=num_channels
        )

        # 合并为40通道体素
        combined_voxels = torch.cat([voxel_coords, voxel_electronegativity, voxel_charge, voxel_scaffold_radius], dim=1)
        combined_voxels = combined_voxels.to(self.device)

        # 裁剪到固定尺寸
        grid_size = self.grid_dim
        real_atom_count = batch["coords"].shape[1] - 2
        real_coords = batch["coords"][:, :real_atom_count, :]
        mol_centers = real_coords.mean(dim=1)
        D1, D2, D3 = combined_voxels.shape[2], combined_voxels.shape[3], combined_voxels.shape[4]
        offset_d1 = (D1 * self.resolution) / 2
        offset_d2 = (D2 * self.resolution) / 2
        offset_d3 = (D3 * self.resolution) / 2
        offsets = torch.tensor([offset_d1, offset_d2, offset_d3], device=self.device).unsqueeze(0)
        voxel_centers = ((mol_centers + offsets) / self.resolution).round().long()

        B = combined_voxels.shape[0]
        min_indices = torch.tensor([0, 0, 0], device=self.device).unsqueeze(0).repeat(B, 1)
        max_indices = torch.tensor([D1 - 1, D2 - 1, D3 - 1], device=self.device).unsqueeze(0).repeat(B, 1)
        box_min = torch.maximum(voxel_centers - grid_size // 2, min_indices)
        box_max = torch.minimum(voxel_centers + grid_size // 2, max_indices)

        batch_idx = torch.arange(B, device=combined_voxels.device)
        x_idx = torch.arange(grid_size, device=combined_voxels.device).view(1, grid_size)
        y_idx = torch.arange(grid_size, device=combined_voxels.device).view(1, grid_size)
        z_idx = torch.arange(grid_size, device=combined_voxels.device).view(1, grid_size)
        x_abs = box_min[:, 0].view(B, 1) + x_idx
        y_abs = box_min[:, 1].view(B, 1) + y_idx
        z_abs = box_min[:, 2].view(B, 1) + z_idx
        x_abs = torch.clamp(x_abs, min=0, max=D1 - 1)
        y_abs = torch.clamp(y_abs, min=0, max=D2 - 1)
        z_abs = torch.clamp(z_abs, min=0, max=D3 - 1)

        batch_idx_expand = batch_idx.unsqueeze(1).unsqueeze(1).unsqueeze(1)
        x_abs_expand = x_abs.unsqueeze(2).unsqueeze(3)
        y_abs_expand = y_abs.unsqueeze(1).unsqueeze(3)
        z_abs_expand = z_abs.unsqueeze(1).unsqueeze(1)
        cropped_voxels = combined_voxels[batch_idx_expand, :, x_abs_expand, y_abs_expand, z_abs_expand]
        cropped_voxels = cropped_voxels.permute(0, 4, 1, 2, 3)
        return cropped_voxels

    def _add_dumb_coords(self, batch: dict) -> dict:
        coords = batch['coords']
        bsz = batch['coords'].shape[0]
        fixed_min = -25.0
        fixed_max = 25.0
        dumb_neg = torch.tensor([fixed_min, fixed_min, fixed_min], dtype=torch.float32).repeat(bsz, 1, 1)
        dumb_pos = torch.tensor([fixed_max, fixed_max, fixed_max], dtype=torch.float32).repeat(bsz, 1, 1)
        dumb_channel = torch.tensor(-1, dtype=torch.float32).repeat(bsz, 2)
        dumb_radius = torch.tensor(0, dtype=torch.float32).repeat(bsz, 2)
        dumb_electronegativity = torch.tensor(-1, dtype=torch.float32).repeat(bsz, 2)
        dumb_charge = torch.tensor(0.0, dtype=torch.float32).repeat(bsz, 2)
        dumb_scaffold_weight = torch.tensor(0.0, dtype=torch.float32).repeat(bsz, 2)
        return {
            "coords": torch.cat((batch['coords'], dumb_neg, dumb_pos), 1),
            "atoms_channel": torch.cat((batch['atoms_channel'], dumb_channel), 1),
            "radius": torch.cat((batch['radius'], dumb_radius), 1),
            "electronegativity": torch.cat((batch['electronegativity'], dumb_electronegativity), 1),
            "charge": torch.cat((batch['charge'], dumb_charge), 1),
            "scaffold_weight": torch.cat((batch['scaffold_weight'], dumb_scaffold_weight), 1)
        }


def load_scaffold_split(dataset_name, split_path=DEFAULT_SPLIT_DIR):
    """加载支架划分的数据集索引"""
    import pickle
    with open(os.path.join(split_path, f"{dataset_name}.pkl"), 'rb') as f:
        split_data = pickle.load(f)
    return split_data["train_idx"], split_data["valid_idx"], split_data["test_idx"]


def clear_cache_dir(cache_dir):
    """删除缓存目录下的所有文件和子目录（如果存在）"""
    if os.path.exists(cache_dir):
        for item in os.listdir(cache_dir):
            item_path = os.path.join(cache_dir, item)
            if os.path.isfile(item_path) or os.path.islink(item_path):
                os.unlink(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
        print(f"✅ 已清空缓存目录: {cache_dir}")
    else:
        print(f"缓存目录不存在，无需清理: {cache_dir}")


def preprocess_and_cache(
        dataset_name="clintox",
        cache_dir=DEFAULT_CACHE_PREFIX,
        max_samples=300,
        raw_dir=DEFAULT_RAW_DIR,
        split_dir=DEFAULT_SPLIT_DIR,
):
    # 数据集配置
    dataset_config = {
        "bace": {
            "csv_path": str(Path(raw_dir) / "bace.csv"),
            "task_names": ["Class"],
        },
        "bbbp": {
            "csv_path":  str(Path(raw_dir) / "bbbp.csv"),
            "task_names": ["p_np"],
        },
        "clintox": {
            "csv_path": str(Path(raw_dir) / "clintox.csv"),
            "task_names": ["FDA_APPROVED", "CT_TOX"]
        },
        "sider": {
            "csv_path": str(Path(raw_dir) / "sider.csv"),
            "task_names": [
                "Hepatobiliary disorders", "Metabolism and nutrition disorders",
                "Product issues", "Eye disorders", "Investigations",
                "Musculoskeletal and connective tissue disorders",
                "Gastrointestinal disorders", "Social circumstances",
                "Immune system disorders", "Reproductive system and breast disorders",
                "Neoplasms benign, malignant and unspecified (incl cysts and polyps)",
                "General disorders and administration site conditions",
                "Endocrine disorders", "Surgical and medical procedures",
                "Vascular disorders", "Blood and lymphatic system disorders",
                "Skin and subcutaneous tissue disorders",
                "Congenital, familial and genetic disorders",
                "Infections and infestations",
                "Respiratory, thoracic and mediastinal disorders",
                "Psychiatric disorders", "Renal and urinary disorders",
                "Pregnancy, puerperium and perinatal conditions",
                "Ear and labyrinth disorders", "Cardiac disorders",
                "Nervous system disorders",
                "Injury, poisoning and procedural complications"
            ]
        },
        "toxcast": {
            "csv_path": str(Path(raw_dir) / "toxcast.csv"),
            "task_names": "auto_extract"
        },
        "tox21": {
            "csv_path": str(Path(raw_dir) / "tox21.csv"),
            "task_names": [
                "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase",
                "NR-ER", "NR-ER-LBD", "NR-PPAR-gamma", "SR-ARE",
                "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53"]
        },
        "freesolv": {
            "csv_path": str(Path(raw_dir) / "freesolv.csv"),
            "task_names": ["expt"],
        },
        "esol": {
            "csv_path": str(Path(raw_dir) / "esol.csv"),
            "task_names": ["measured log solubility in mols per litre"],
        },
        "lipophilicity": {
            "csv_path": str(Path(raw_dir) / "lipophilicity.csv"),
            "task_names": ["exp"],
        },
        "muv": {
            "csv_path": None,  # MUV为拆分好的csv，主路径在后续单独指定
            "task_names": "auto_extract"  # 自动提取所有任务列（排除smiles/subset）
        },
        "hiv": {
            "csv_path":None,
            "task_names": ["HIV_active"],
        },
        "ames": {
            "csv_path":str(Path(raw_dir) / "ames.csv"),
            "task_names": ["Y"],
        },
        "herg":{
            "csv_path":str(Path(raw_dir) / "herg.csv"),
            "task_names":["Y"]
        },
        "dili": {
            "csv_path": str(Path(raw_dir) / "dili.csv"),
            "task_names": ["Y"]
        },
    }
    config = dataset_config[dataset_name]
    csv_path = config["csv_path"]
    task_names = config["task_names"]
    datasets = ["muv", "hiv"]
    if dataset_name in datasets:

        # ********** 替换原有load_scaffold_split和df加载逻辑 **********
        # MUV为已拆分好的train/val/test.csv，单独加载并合并
        muv_base_path = str(raw_dir)
        train_df = pd.read_csv(os.path.join(muv_base_path, f"{dataset_name.upper()}_train.csv"))
        val_df = pd.read_csv(os.path.join(muv_base_path, f"{dataset_name.upper()}_valid.csv"))
        test_df = pd.read_csv(os.path.join(muv_base_path, f"{dataset_name.upper()}_test.csv"))
        # 为每个子集标记标签，并重置索引（保证idx唯一）
        train_df["subset"] = "train"
        val_df["subset"] = "val"
        test_df["subset"] = "test"
        # 合并为总df，重置索引（后续遍历用这个总df）
        df = pd.concat([train_df, val_df, test_df], ignore_index=True)
        # 提取任务列（排除smiles和subset）
        task_names = [col for col in df.columns if col not in ["smiles", "subset"]]
        num_tasks = len(task_names)
        print(f"处理数据集: {dataset_name}, 任务数: {num_tasks}")
        # 生成所有样本的索引（就是df的行索引）
        all_indices = df.index.tolist()
        # ********** 替换结束 **********
    else:
        # 加载支架划分的索引
        train_indices, val_indices, test_indices = load_scaffold_split(dataset_name, split_path=split_dir)
        df = pd.read_csv(csv_path)

        # 为每个样本标记所属子集
        df["subset"] = "unknown"
        df.loc[train_indices, "subset"] = "train"
        df.loc[val_indices, "subset"] = "val"
        df.loc[test_indices, "subset"] = "test"

        num_tasks = len(task_names) if task_names is not None else len(
            [col for col in df.columns if col not in ["smiles", "mol_id"]])
        print(f"处理数据集: {dataset_name}, 任务数: {num_tasks}")

    # 创建缓存目录（区分子集）
    cache_dir = f"{cache_dir}-{dataset_name}"
    os.makedirs(os.path.join(cache_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(cache_dir, "val"), exist_ok=True)
    os.makedirs(os.path.join(cache_dir, "test"), exist_ok=True)
    print(f"体素数据将缓存至: {cache_dir} (按train/val/test子目录区分)")

    # 清空缓存目录
    clear_cache_dir(cache_dir)
    # 重新创建子目录
    os.makedirs(os.path.join(cache_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(cache_dir, "val"), exist_ok=True)
    os.makedirs(os.path.join(cache_dir, "test"), exist_ok=True)
    print(f"体素数据将缓存至: {cache_dir} (npz压缩格式，每{max_samples}个样本分块保存)")

    # 初始化工具
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    voxelizer = Voxelizer(grid_dim=64, device=device).to(device)

    # 为每个子集初始化分块缓存
    subset_chunks = {
        "train": {"voxels": [], "labels": [], "smiles": [], "indices": [], "index": 0},
        "val": {"voxels": [], "labels": [], "smiles": [], "indices": [], "index": 0},
        "test": {"voxels": [], "labels": [], "smiles": [], "indices": [], "index": 0}
    }

    # 原子特征映射表
    atom_to_channel = {
        'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4,
        'S': 5, 'Cl': 6, 'Br': 7, 'I': 8, 'Other': 9
    }
    atom_radius_dict = {
        'H': 0.37, 'C': 0.77, 'N': 0.75, 'O': 0.73, 'F': 0.71,
        'S': 1.02, 'Cl': 0.99, 'Br': 1.14, 'I': 1.33, 'Other': 0.5
    }
    atom_electronegativity = {
        'H': 2.20, 'C': 2.55, 'N': 3.04, 'O': 3.44, 'F': 3.98,
        'S': 2.58, 'Cl': 3.16, 'Br': 2.96, 'I': 2.66, 'Other': 2.50
    }
    atom_charge_default = {k: 0.0 for k in atom_to_channel}

    # 骨架原子索引获取
    def _get_scaffold_atom_indices(mol):
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        match = mol.GetSubstructMatch(scaffold)
        return set(match) if match else {i for i, a in enumerate(mol.GetAtoms()) if a.GetSymbol() != 'H'}
    if dataset_name not in datasets:
        all_indices = train_indices + val_indices + test_indices
    # ********** 核心修改1：删除原有all_indices = train_indices + val_indices + test_indices **********
    # 直接使用合并后df的全局索引（之前重构的MUV加载逻辑中已生成all_indices = df.index.tolist()）
    for idx in tqdm(all_indices, desc="生成体素数据"):
        # ********** 核心修改2：移除冗余的进度打印（tqdm已自带进度，避免日志杂乱） **********
        # 删掉以下两行原有代码
        # current_sample_num = all_indices.index(idx) + 1
        # print(f"当前处理：第{current_sample_num}个样本 | idx={idx}", flush=True)

        row = df.iloc[idx]
        subset = row["subset"]
        cache_subdir = os.path.join(cache_dir, subset)

        # ********** 核心修改3：保留toxcast/tox21的标签特殊处理，新增MUV的task_names自动提取（仅首次循环执行） **********
        if (dataset_name == "toxcast" or dataset_name == "muv") and task_names == "auto_extract":
            # 排除smiles/subset/mol_id，提取所有任务列（兼容toxcast和MUV）
            task_names = [col for col in df.columns if col.lower() not in ["smiles", "subset", "mol_id"]]
            config["task_names"] = task_names  # 缓存结果，避免循环内重复计算

        # 读取样本 SMILES（修复 herg/dili/ames 列名是 Drug）
        if dataset_name in ["herg", "dili", "ames"]:
            smiles = row['Drug']  # 这三个数据集的SMILES列叫 Drug
        else:
            smiles = row['smiles']  # 其他正常数据集
        try:
            # 提取标签并统一转为数值型（容错处理，避免非数值标签报错）
            labels = row[task_names].values

            # 定义分类任务特殊处理的数据集
            tox_datasets = ["tox21", "toxcast", "muv","ames"]
            # 核心逻辑：tox21/toxcast/MUV → NaN/-1→-1；其他分类→0
            if dataset_name in tox_datasets:
                # tox系列/MUV：NaN填充为-1，-1保留不变（适配常见的毒理分类数据集标注）
                labels = np.nan_to_num(labels, nan=-1.0)
            else:
                # 其他分类任务：NaN填充为0，-1也转为0
                labels = np.nan_to_num(labels, nan=0.0)
                labels[labels == -1.0] = 0

            # 最终转换为float类型，保证后续计算兼容
            labels = labels.astype(float)

        except Exception as e:
            # 增强报错信息，方便定位问题样本
            print(f"样本 {idx} (SMILES: {smiles[:20]}...) 标签处理失败: {e}，跳过", flush=True)
            continue

        # 解析分子
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"样本 {idx} SMILES无效，跳过")
            continue
        mol = Chem.AddHs(mol)

        # 生成3D构象
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        params.useSmallRingTorsions = True
        params.maxIterations = 400
        params.pruneRmsThresh = 0.2
        params.numThreads = 2
        conformer_success = False
        for _ in range(3):
            try:
                if AllChem.EmbedMolecule(mol, params) == 0:
                    try:
                        AllChem.MMFFOptimizeMolecule(mol)
                    except:
                        AllChem.UFFOptimizeMolecule(mol)
                    conformer_success = True
                    break
            except:
                continue
        if not conformer_success:
            print(f"样本 {idx} 3D构象生成失败，直接跳过", flush=True)
            continue

        try:
            test_conf = mol.GetConformer()
            if test_conf.GetNumAtoms() == 0:
                raise ValueError("构象无原子坐标")
        except Exception as e:
            print(f"样本 {idx} 分子构象无效: {e}，直接跳过", flush=True)
            continue

        # 提取原子特征
        coords, atoms_channel, radius, electronegativity, charge, scaffold_weight = [], [], [], [], [], []
        scaffold_indices = _get_scaffold_atom_indices(mol)
        for atom in mol.GetAtoms():
            atom_idx = atom.GetIdx()
            atom_symbol = atom.GetSymbol()
            channel_idx = atom_to_channel.get(atom_symbol, 9)
            pos = mol.GetConformer().GetAtomPosition(atom_idx)

            if not np.isfinite([pos.x, pos.y, pos.z]).all():
                continue
            is_scaffold = 1.0 if atom_idx in scaffold_indices else 0.3
            scaffold_weight.append(is_scaffold)
            coords.append([pos.x, pos.y, pos.z])
            atoms_channel.append(channel_idx)
            radius.append(atom_radius_dict.get(atom_symbol, 0.5))
            electronegativity.append(atom_electronegativity.get(atom_symbol, 2.50))
            charge.append(
                atom.GetFormalCharge() if atom.GetFormalCharge() != 0 else atom_charge_default.get(atom_symbol, 0.0))

        if len(coords) == 0:
            print(f"样本 {idx} 无有效原子，跳过")
            continue

        # 生成体素
        try:
            coords = torch.tensor(coords, dtype=torch.float32).unsqueeze(0)
            atoms_channel = torch.tensor(atoms_channel, dtype=torch.float32).unsqueeze(0)
            radius = torch.tensor(radius, dtype=torch.float32).unsqueeze(0)
            electronegativity = torch.tensor(electronegativity, dtype=torch.float32).unsqueeze(0)
            charge = torch.tensor(charge, dtype=torch.float32).unsqueeze(0)
            scaffold_weight = torch.tensor(scaffold_weight, dtype=torch.float32).unsqueeze(0)


            batch = {
                "coords": coords, "atoms_channel": atoms_channel,
                "radius": radius, "electronegativity": electronegativity,
                "charge": charge, "scaffold_weight": scaffold_weight
            }

            print(f"开始体素化 | idx={idx}", flush=True)
            voxel_tensor = voxelize_core(voxelizer, batch)
            voxel = voxel_tensor.squeeze(0).cpu().numpy()  # 转换为numpy数组
            print(f"体素化完成 | idx={idx}", flush=True)

        except TimeoutError:
            print(f"❌ 样本 idx={idx} 体素化超时（>8秒），跳过", flush=True)
            continue
        except Exception as e:
            print(f"❌ 样本 idx={idx} 体素化失败: {e}，跳过", flush=True)
            continue

        # 缓存当前样本到对应子集的分块中
        current_chunk = subset_chunks[subset]
        current_chunk["voxels"].append(voxel)
        current_chunk["labels"].append(labels)
        current_chunk["smiles"].append(smiles)
        current_chunk["indices"].append(idx)

        # 当分块样本数达到max_samples时，保存分块
        if len(current_chunk["voxels"]) >= max_samples:
            chunk_path = os.path.join(cache_subdir, f"{subset}_chunk_{current_chunk['index']}.npz")
            # 压缩保存（voxels和labels转为numpy数组，其他信息保留列表格式）
            np.savez_compressed(
                chunk_path,
                voxels=np.stack(current_chunk["voxels"], axis=0),
                labels=np.stack(current_chunk["labels"], axis=0),
                smiles=np.array(current_chunk["smiles"], dtype=object),
                indices=np.array(current_chunk["indices"], dtype=int)
            )
            print(f"✅ 已保存 {subset} 分块 {current_chunk['index']}（{len(current_chunk['voxels'])}个样本）")
            # 重置当前分块
            current_chunk["voxels"] = []
            current_chunk["labels"] = []
            current_chunk["smiles"] = []
            current_chunk["indices"] = []
            current_chunk["index"] += 1

    # 处理每个子集剩余的样本
    for subset in ["train", "val", "test"]:
        current_chunk = subset_chunks[subset]
        if len(current_chunk["voxels"]) > 0:
            cache_subdir = os.path.join(cache_dir, subset)
            chunk_path = os.path.join(cache_subdir, f"{subset}_chunk_{current_chunk['index']}.npz")
            np.savez_compressed(
                chunk_path,
                voxels=np.stack(current_chunk["voxels"], axis=0),
                labels=np.stack(current_chunk["labels"], axis=0),
                smiles=np.array(current_chunk["smiles"], dtype=object),
                indices=np.array(current_chunk["indices"], dtype=int)
            )
            print(f"✅ 已保存 {subset} 最后分块 {current_chunk['index']}（{len(current_chunk['voxels'])}个样本）")

    # 统计信息
    print(f"\n处理完成:")
    print(
        f"训练集总有效样本: {len(subset_chunks['train']['voxels']) + subset_chunks['train']['index'] * max_samples}/{len(train_indices)}")
    print(
        f"验证集总有效样本: {len(subset_chunks['val']['voxels']) + subset_chunks['val']['index'] * max_samples}/{len(val_indices)}")
    print(
        f"测试集总有效样本: {len(subset_chunks['test']['voxels']) + subset_chunks['test']['index'] * max_samples}/{len(test_indices)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate downstream voxel caches from CSV and fixed splits.")
    parser.add_argument("--dataset", choices=("bbbp", "bace", "sider", "clintox", "ames", "herg", "dili"), default="dili")
    parser.add_argument("--raw-dir", default=DEFAULT_RAW_DIR)
    parser.add_argument("--split-dir", default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_PREFIX, help="Cache prefix; -<dataset> is appended.")
    parser.add_argument("--chunk-size", type=int, default=150)
    args = parser.parse_args()
    preprocess_and_cache(
        dataset_name=args.dataset,
        cache_dir=args.cache_dir,
        max_samples=args.chunk_size,
        raw_dir=args.raw_dir,
        split_dir=args.split_dir,
    )
