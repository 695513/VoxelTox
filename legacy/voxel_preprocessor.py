import os
import pickle
import torch
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
from tqdm import tqdm
import random
import shutil
# 禁用RDKit日志
RDLogger.DisableLog('rdApp.*')

import timeout_decorator
from timeout_decorator import TimeoutError

# 2. 定义“只包裹体素化核心”的超时函数
@timeout_decorator.timeout(8)  # 缩短到8秒，更快触发跳过
def voxelize_core(voxelizer, batch):
    # 只放最耗时、最易阻塞的mol2vox调用，其他代码不放这里
    return voxelizer.mol2vox(batch, num_channels=10)
    
# --------------------------
# 体素化工具类
# --------------------------
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


def load_scaffold_split(dataset_name, split_path="/root/.cache/Microsoft/DeveloperTools/Voxbind/split/scaffold"):
    """加载支架划分的数据集索引"""
    with open(os.path.join(split_path, f"{dataset_name}.pkl"), 'rb') as f:
        split_data = pickle.load(f)
    return split_data["train_idx"], split_data["valid_idx"], split_data["test_idx"]

# 新增：清空缓存目录函数
def clear_cache_dir(cache_dir):
    """删除缓存目录下的所有文件和子目录（如果存在）"""
    if os.path.exists(cache_dir):
        # 遍历目录下的所有内容
        for item in os.listdir(cache_dir):
            item_path = os.path.join(cache_dir, item)
            # 如果是文件，直接删除
            if os.path.isfile(item_path) or os.path.islink(item_path):
                os.unlink(item_path)
            # 如果是子目录，递归删除
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
        print(f"✅ 已清空缓存目录: {cache_dir}")
    else:
        print(f"缓存目录不存在，无需清理: {cache_dir}")

def preprocess_and_cache(
    dataset_name="clintox", 
    cache_dir="/root/autodl-tmp/voxel_cache", 
    max_samples=300
):
    # 数据集配置
    dataset_config = {
        "bace": {
            "csv_path": "/root/.cache/Microsoft/DeveloperTools/Voxbind/raw/bace.csv",
            "task_names": ["Class"],
        },
        "bbbp": {
            "csv_path": "/root/.cache/Microsoft/DeveloperTools/Voxbind/raw/bbbp.csv",
            "task_names": ["p_np"],
        },
        "clintox": {
            "csv_path": "/root/.cache/Microsoft/DeveloperTools/Voxbind/raw/clintox.csv",
            "task_names": ["FDA_APPROVED", "CT_TOX"]
        },
        "sider": {
            "csv_path": "/root/.cache/Microsoft/DeveloperTools/Voxbind/raw/sider.csv",
            "task_names": [
                "Hepatobiliary disorders",
                "Metabolism and nutrition disorders",
                "Product issues",
                "Eye disorders",
                "Investigations",
                "Musculoskeletal and connective tissue disorders",
                "Gastrointestinal disorders",
                "Social circumstances",
                "Immune system disorders",
                "Reproductive system and breast disorders",
                "Neoplasms benign, malignant and unspecified (incl cysts and polyps)",
                "General disorders and administration site conditions",
                "Endocrine disorders",
                "Surgical and medical procedures",
                "Vascular disorders",
                "Blood and lymphatic system disorders",
                "Skin and subcutaneous tissue disorders",
                "Congenital, familial and genetic disorders",
                "Infections and infestations",
                "Respiratory, thoracic and mediastinal disorders",
                "Psychiatric disorders",
                "Renal and urinary disorders",
                "Pregnancy, puerperium and perinatal conditions",
                "Ear and labyrinth disorders",
                "Cardiac disorders",
                "Nervous system disorders",
                "Injury, poisoning and procedural complications"
            ]
        },
        "toxcast": {
            "csv_path": "/root/.cache/Microsoft/DeveloperTools/Voxbind/raw/toxcast.csv",
            "task_names": "auto_extract"
        },
        "tox21": {
            "csv_path": "/root/.cache/Microsoft/DeveloperTools/Voxbind/raw/tox21.csv",
            "task_names":  [
            "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", 
            "NR-ER", "NR-ER-LBD", "NR-PPAR-gamma", "SR-ARE", 
            "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53"]
        },
        "freesolv": {
            "csv_path": "/root/.cache/Microsoft/DeveloperTools/Voxbind/raw/freesolv.csv",
            "task_names": ["expt"],
        },
        "esol": {
            "csv_path": "/root/.cache/Microsoft/DeveloperTools/Voxbind/raw/esol.csv",
            "task_names": ["measured log solubility in mols per litre"],
        },
        "lipophilicity": {
            "csv_path": "/root/.cache/Microsoft/DeveloperTools/Voxbind/raw/lipophilicity.csv",
            "task_names": ["exp"],
        },
    }
    config = dataset_config[dataset_name]
    csv_path = config["csv_path"]
    task_names = config["task_names"]
    
    # 加载支架划分的索引
    train_indices, val_indices, test_indices = load_scaffold_split(dataset_name)
    df = pd.read_csv(csv_path)
    
    # 为每个样本标记所属子集
    df["subset"] = "unknown"
    df.loc[train_indices, "subset"] = "train"
    df.loc[val_indices, "subset"] = "val"
    df.loc[test_indices, "subset"] = "test"
    
    num_tasks = len(task_names) if task_names is not None else len([col for col in df.columns if col not in ["smiles", "mol_id"]])
    print(f"处理数据集: {dataset_name}, 任务数: {num_tasks}")

    # 创建缓存目录（区分子集）
    cache_dir = f"{cache_dir}-{dataset_name}"
    os.makedirs(os.path.join(cache_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(cache_dir, "val"), exist_ok=True)
    os.makedirs(os.path.join(cache_dir, "test"), exist_ok=True)
    print(f"体素数据将缓存至: {cache_dir} (按train/val/test子目录区分)")

        # 关键新增：运行前清空缓存目录
    clear_cache_dir(cache_dir)
    print(f"体素数据将缓存至: {cache_dir} (按train/val/test子目录区分，每{max_samples}个样本分块保存)")

    # 2. 重新创建子目录（此时创建的子目录不会被删除）
    os.makedirs(os.path.join(cache_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(cache_dir, "val"), exist_ok=True)
    os.makedirs(os.path.join(cache_dir, "test"), exist_ok=True)
    print(f"体素数据将缓存至: {cache_dir} (按train/val/test子目录区分，每{max_samples}个样本分块保存)")    

    # 初始化工具：使用正确的Voxelizer类
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    voxelizer = Voxelizer(grid_dim=64, device=device).to(device)

     # 为每个子集初始化分块缓存（关键修改：按子集分块）
    subset_chunks = {
        "train": {"data": [], "index": 0},  # "data"存储当前分块样本，"index"记录分块编号
        "val": {"data": [], "index": 0},
        "test": {"data": [], "index": 0}
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

    # 获取稳定构象
    def _get_stable_conformers(mol, num_confs=2, keep_top=1):
        scaffold_indices = _get_scaffold_atom_indices(mol)
        conf_ids = AllChem.EmbedMultipleConfs(
            mol, numConfs=num_confs, randomSeed=42,
            useExpTorsionAnglePrefs=True, useBasicKnowledge=True
        )
        if len(conf_ids) < keep_top:
            return [mol]

        valid_confs = []
        original_conf = mol.GetConformer()
        for conf_id in conf_ids:
            AllChem.UFFOptimizeMolecule(mol, confId=conf_id)
            ff = AllChem.UFFGetMoleculeForceField(mol, confId=conf_id)
            energy = ff.CalcEnergy()

            current_conf = mol.GetConformer(conf_id)
            dev = 0.0
            for bond in mol.GetBonds():
                a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                if a1 in scaffold_indices and a2 in scaffold_indices:
                    orig_dist = np.linalg.norm([
                        original_conf.GetAtomPosition(a1).x - original_conf.GetAtomPosition(a2).x,
                        original_conf.GetAtomPosition(a1).y - original_conf.GetAtomPosition(a2).y,
                        original_conf.GetAtomPosition(a1).z - original_conf.GetAtomPosition(a2).z
                    ])
                    curr_dist = np.linalg.norm([
                        current_conf.GetAtomPosition(a1).x - current_conf.GetAtomPosition(a2).x,
                        current_conf.GetAtomPosition(a1).y - current_conf.GetAtomPosition(a2).y,
                        current_conf.GetAtomPosition(a1).z - current_conf.GetAtomPosition(a2).z
                    ])
                    dev += abs(curr_dist - orig_dist)
            if dev < 0.5:
                valid_confs.append((conf_id, energy))

        if len(valid_confs) < keep_top:
            return [mol]
        return [Chem.Mol(mol) for conf_id, _ in sorted(valid_confs, key=lambda x: x[1])[:keep_top]]

    # 扰动构象
    def _perturb_conformer(mol):
        try:
            # 复制分子并尝试获取构象（核心容错点）
            aug_mol = Chem.Mol(mol)
            conf = aug_mol.GetConformer()  # 可能抛出Bad Conformer Id的位置
        except Exception as e:
            # 打印错误日志（可选，但方便调试）
            print(f"构象扰动失败: {e}，使用原分子")
            # 直接返回原分子（确保安全，不使用可能有问题的aug_mol）
            return mol
        scaffold_indices = _get_scaffold_atom_indices(mol)
        side_chain_indices = [i for i in range(aug_mol.GetNumAtoms()) if i not in scaffold_indices]
        if not side_chain_indices:
            return aug_mol

        angle_x = np.radians(random.uniform(-2, 2))
        angle_y = np.radians(random.uniform(-2, 2))
        rot_x = np.array([[1, 0, 0], [0, np.cos(angle_x), -np.sin(angle_x)], [0, np.sin(angle_x), np.cos(angle_x)]])
        rot_y = np.array([[np.cos(angle_y), 0, np.sin(angle_y)], [0, 1, 0], [-np.sin(angle_y), 0, np.cos(angle_y)]])
        noise = torch.normal(mean=0.0, std=0.03, size=(aug_mol.GetNumAtoms(), 3)).numpy()

        for i in range(aug_mol.GetNumAtoms()):
            pos = conf.GetAtomPosition(i)
            coords = np.array([pos.x, pos.y, pos.z])
            if i in side_chain_indices:
                coords = np.dot(coords, rot_x)
                coords = np.dot(coords, rot_y)
            coords += noise[i] * 0.3 if i in scaffold_indices else noise[i]
            conf.SetAtomPosition(i, (coords[0], coords[1], coords[2]))
        return aug_mol


    aug_params = {
        "aug_prob": 0.8,
        "num_confs": 2,
        "keep_top": 1
    }

    # 按支架划分的索引顺序处理
    all_indices = train_indices + val_indices + test_indices
    for idx in tqdm(all_indices, desc="生成体素数据"):
        # if idx in {5, 1390,1333,197}:  # 直接把之前遇到的534也加进去，避免重复卡住
        #     print(f"跳过体素化阻塞样本 | idx={idx}", flush=True)
        #     continue
        current_sample_num = all_indices.index(idx) + 1  # 计算当前是第几个样本（从1开始）
        print(f"当前处理：第{current_sample_num}个样本 | idx={idx}", flush=True)
        row = df.iloc[idx]
        subset = row["subset"]
        cache_subdir = os.path.join(cache_dir, subset)
        cache_file = os.path.join(cache_subdir, f"voxel_{idx}.pkl")

       

        # 读取样本
        smiles = row['smiles']
        # 原标签提取逻辑修改（覆盖所有数据集）
        try:
            # 处理 toxcast 的自动任务名提取
            if dataset_name == "toxcast" and task_names == "auto_extract":
                task_names = [col for col in df.columns if col.lower() != "smiles" and col.lower() != "subset"]
            
            # 提取标签并填充缺失值（关键步骤）
            labels = row[task_names].values
            # 1. 缺失值用0填充（适配BCELoss二分类）
            labels = np.nan_to_num(labels, nan=0.0)
            # 2. 将-1转为0（原-1表示无活性，统一为0；1表示有活性）
            labels[labels == -1.0] = 0.0
            # 3. 确保标签为float32（PyTorch BCELoss要求）
            labels = labels.astype(float)

        except Exception as e:
            print(f"样本 {idx} 标签处理失败: {e}，标签列: {task_names[:5]}...（共{len(task_names)}个），当前值: {row[task_names].values[:5]}...")
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
        params.pruneRmsThresh = 0.2  # 从默认0.1放宽，允许更多构象尝试
        # 关键4：增加初始构象生成数量（多试几次）
        params.numThreads = 2  # 多线程加速尝试
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
        # 关键修改：3D构象失败直接跳过，不尝试2D
        if not conformer_success:
            print(f"样本 {idx} 3D构象生成失败，直接跳过（不尝试2D构象）", flush=True)
            continue  # 直接跳过，不进入后续处理
        # if not conformer_success:
        #     print(f"样本 {idx} 3D构象生成失败，尝试2D构象...")
        #     # 检查2D构象是否生成成功
        #     if not AllChem.Compute2DCoords(mol):
        #         print(f"样本 {idx} 2D构象也生成失败，跳过该样本")
        #         continue  # 直接跳过，避免后续错误
        #     # 2D构象生成成功，继续处理
        #     conf = mol.GetConformer()
        #     for i in range(mol.GetNumAtoms()):
        #         pos = conf.GetAtomPosition(i)
        #         conf.SetAtomPosition(i, (pos.x, pos.y, 0.0))


        # 数据增强（仅对训练集应用）
        final_mol = mol
        if subset == "train" and conformer_success and random.random() < aug_params["aug_prob"]:
            stable_confs = _get_stable_conformers(mol, aug_params["num_confs"], aug_params["keep_top"])
            augmented_confs = [_perturb_conformer(conf) for conf in stable_confs]
            final_mol = random.choice([mol] + augmented_confs)
        else:
            final_mol = mol

        try:
            # 检查final_mol（原始/增强后）是否有有效构象
            test_conf = final_mol.GetConformer()
            # 进一步检查是否有原子坐标（避免空构象）
            if test_conf.GetNumAtoms() == 0:
                raise ValueError("构象无原子坐标")
        except Exception as e:
            print(f"样本 {idx} 最终分子构象无效: {e}，直接跳过", flush=True)
            continue
        # 提取原子特征
        coords, atoms_channel, radius, electronegativity, charge, scaffold_weight = [], [], [], [], [], []
        scaffold_indices = _get_scaffold_atom_indices(final_mol)
        for atom in final_mol.GetAtoms():
            atom_idx = atom.GetIdx()
            atom_symbol = atom.GetSymbol()
            channel_idx = atom_to_channel.get(atom_symbol, 9)
            pos = final_mol.GetConformer().GetAtomPosition(atom_idx)

            if not np.isfinite([pos.x, pos.y, pos.z]).all():
                continue
            is_scaffold = 1.0 if atom_idx in scaffold_indices else 0.3
            scaffold_weight.append(is_scaffold)
            coords.append([pos.x, pos.y, pos.z])
            atoms_channel.append(channel_idx)
            radius.append(atom_radius_dict.get(atom_symbol, 0.5))
            electronegativity.append(atom_electronegativity.get(atom_symbol, 2.50))
            charge.append(atom.GetFormalCharge() if atom.GetFormalCharge() != 0 else atom_charge_default.get(atom_symbol, 0.0))

        if len(coords) == 0:
            print(f"样本 {idx} 无有效原子，跳过")
            continue

        # 生成体素
        try:
            # 1. 先构建batch（这部分不耗时，不用超时）
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

            # 2. 调用超时函数处理核心体素化（关键！）
            print(f"开始体素化 | idx={idx}", flush=True)  # 新增日志，确认走到这步
            voxel_tensor = voxelize_core(voxelizer, batch)  # 这行是核心，超时只盯它

            # 3. 体素化成功后的后续处理（不耗时）
            voxel = voxel_tensor.squeeze(0).cpu()
            print(f"体素化完成 | idx={idx}", flush=True)  # 新增日志，确认成功

        except TimeoutError:
            print(f"❌ 样本 idx={idx} 体素化超时（>8秒），跳过", flush=True)
            continue
        except Exception as e:
            print(f"❌ 样本 idx={idx} 体素化失败: {e}，跳过", flush=True)
            continue

        # 缓存当前样本到对应子集的分块中（关键修改）
        current_chunk = subset_chunks[subset]
        current_chunk["data"].append({
            "voxel": voxel,
            "labels": labels,
            "smiles": smiles,
            "idx": idx,
            "subset": subset
        })

        # 当分块样本数达到max_samples时，保存分块并重置当前分块
        if len(current_chunk["data"]) >= max_samples:
            chunk_path = os.path.join(cache_subdir, f"{subset}_chunk_{current_chunk['index']}.pkl")
            with open(chunk_path, 'wb') as f:
                pickle.dump(current_chunk["data"], f)
            print(f"✅ 已保存 {subset} 分块 {current_chunk['index']}（{len(current_chunk['data'])}个样本）")
            # 重置当前分块，准备下一轮
            current_chunk["data"] = []
            current_chunk["index"] += 1

    # 处理每个子集剩余的样本（不足一个分块的部分）
    for subset in ["train", "val", "test"]:
        current_chunk = subset_chunks[subset]
        if len(current_chunk["data"]) > 0:
            cache_subdir = os.path.join(cache_dir, subset)
            chunk_path = os.path.join(cache_subdir, f"{subset}_chunk_{current_chunk['index']}.pkl")
            with open(chunk_path, 'wb') as f:
                pickle.dump(current_chunk["data"], f)
            print(f"✅ 已保存 {subset} 最后分块 {current_chunk['index']}（{len(current_chunk['data'])}个样本）")

    # 统计信息（修改为按子集总样本数统计）
    print(f"\n处理完成:")
    print(f"训练集总有效样本: {len(subset_chunks['train']['data']) + subset_chunks['train']['index'] * max_samples}/{len(train_indices)}")
    print(f"验证集总有效样本: {len(subset_chunks['val']['data']) + subset_chunks['val']['index'] * max_samples}/{len(val_indices)}")
    print(f"测试集总有效样本: {len(subset_chunks['test']['data']) + subset_chunks['test']['index'] * max_samples}/{len(test_indices)}")

if __name__ == "__main__":
    preprocess_and_cache(
        dataset_name="bbbp", 
        max_samples=150  # 传入分块大小，而非总样本数限制
    )