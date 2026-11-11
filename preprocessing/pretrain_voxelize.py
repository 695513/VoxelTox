import os
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SDF_PATH = str(PROJECT_ROOT / "data" / "pretrain" / "pretrain_30w_no_ions.sdf")
DEFAULT_CACHE_DIR = str(PROJECT_ROOT / "data" / "cache" / "pretrain")
import pickle
import torch
import numpy as np
from rdkit import RDLogger
from tqdm import tqdm
import shutil
from multiprocessing import Pool
import time

# 禁用RDKit日志
RDLogger.DisableLog('rdApp.*')


def voxelize_core(voxelizer, batch):
    """仅封装最耗时的体素化核心调用，支持批量输入"""
    return voxelizer.mol2vox(batch, num_channels=10)


# 体素化工具类（复用）
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

        # 裁剪到固定64×64×64尺寸
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
        cropped_voxels = cropped_voxels.permute(0,4,1,2,3).contiguous()    # 注释后测试
        return cropped_voxels

    def _add_dumb_coords(self, batch: dict) -> dict:
        coords = batch['coords']
        bsz = batch['coords'].shape[0]
        fixed_min = -25.0
        fixed_max = 25.0
        # 关键修改：在self.device上创建临时张量
        dumb_neg = torch.tensor([fixed_min, fixed_min, fixed_min], dtype=torch.float32, device=self.device).repeat(bsz,
                                                                                                                   1, 1)
        dumb_pos = torch.tensor([fixed_max, fixed_max, fixed_max], dtype=torch.float32, device=self.device).repeat(bsz,
                                                                                                                   1, 1)
        dumb_channel = torch.tensor(-1, dtype=torch.float32, device=self.device).repeat(bsz, 2)
        dumb_radius = torch.tensor(0, dtype=torch.float32, device=self.device).repeat(bsz, 2)
        dumb_electronegativity = torch.tensor(-1, dtype=torch.float32, device=self.device).repeat(bsz, 2)
        dumb_charge = torch.tensor(0.0, dtype=torch.float32, device=self.device).repeat(bsz, 2)
        dumb_scaffold_weight = torch.tensor(0.0, dtype=torch.float32, device=self.device).repeat(bsz, 2)
        return {
            "coords": torch.cat((batch['coords'], dumb_neg, dumb_pos), 1),
            "atoms_channel": torch.cat((batch['atoms_channel'], dumb_channel), 1),
            "radius": torch.cat((batch['radius'], dumb_radius), 1),
            "electronegativity": torch.cat((batch['electronegativity'], dumb_electronegativity), 1),
            "charge": torch.cat((batch['charge'], dumb_charge), 1),
            "scaffold_weight": torch.cat((batch['scaffold_weight'], dumb_scaffold_weight), 1)
        }

# 清空缓存目录函数
def clear_cache_dir(cache_dir):
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


# 多进程预处理函数
def preprocess_mol_block(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 禁止子进程访问任何GPU，强制使用CPU
    from rdkit import Chem  # 移到子进程内导入，确保资源隔离
    from rdkit.Chem.Scaffolds import MurckoScaffold
    """单分子块的预处理（解析+原子特征提取），供多进程调用"""
    mol_block, mol_idx, atom_to_channel, atom_radius_dict, atom_electronegativity = args
    # 解析分子块
    mol = Chem.MolFromMolBlock(mol_block)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)  # 加氢

    # 检查3D构象
    if mol.GetNumConformers() == 0:
        return None
    conf = mol.GetConformer()

    # 骨架原子识别
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    scaffold_indices = set(mol.GetSubstructMatch(scaffold)) if scaffold else {i for i, a in enumerate(mol.GetAtoms()) if
                                                                              a.GetSymbol() != 'H'}

    # 提取原子特征
    coords, atoms_channel, radius, electronegativity, charge, scaffold_weight = [], [], [], [], [], []
    for atom in mol.GetAtoms():
        atom_idx = atom.GetIdx()
        atom_symbol = atom.GetSymbol()
        pos = conf.GetAtomPosition(atom_idx)
        # 过滤无效坐标
        if not np.isfinite([pos.x, pos.y, pos.z]).all():
            continue
        # 特征赋值
        is_scaffold = 1.0 if atom_idx in scaffold_indices else 0.3
        scaffold_weight.append(is_scaffold)
        coords.append([pos.x, pos.y, pos.z])
        atoms_channel.append(atom_to_channel.get(atom_symbol, 9))
        radius.append(atom_radius_dict.get(atom_symbol, 0.5))
        electronegativity.append(atom_electronegativity.get(atom_symbol, 2.50))
        charge.append(atom.GetFormalCharge() if atom.GetFormalCharge() != 0 else 0.0)

    # 过滤无有效原子的分子
    if len(coords) == 0:
        return None

    # 返回值全部用numpy数组
    return {
        "coords": np.array(coords, dtype=np.float32),  # 替换torch.tensor
        "atoms_channel": np.array(atoms_channel, dtype=np.float32),
        "radius": np.array(radius, dtype=np.float32),
        "electronegativity": np.array(electronegativity, dtype=np.float32),
        "charge": np.array(charge, dtype=np.float32),
        "scaffold_weight": np.array(scaffold_weight, dtype=np.float32),
        "smiles": Chem.MolToSmiles(mol),
        "mol_idx": mol_idx,
        "scaffold_count": len(scaffold_indices)
    }

def voxel_quantize(voxel_tensor, dtype=torch.uint8):
    # 改为按每个分子独立计算min/max（voxel_tensor.shape: [40,64,64,64]）
    voxel_tensor = voxel_tensor.cpu()
    mol_min = voxel_tensor.view(-1).min().item()  # 单个分子的所有体素最小值
    mol_max = voxel_tensor.view(-1).max().item()
    range_val = mol_max - mol_min
    if range_val < 1e-6:
        scale = 1.0
        zero_point = 0.0
        quantized_data = torch.zeros_like(voxel_tensor, dtype=dtype)
    else:
        scale = range_val / (2**8 - 1)  # 移除1.1倍冗余，避免精度损失
        zero_point = -mol_min / scale
        quantized_data = torch.clamp(
            (voxel_tensor / scale + zero_point).round(),
            0, 255
        ).to(dtype)
    return quantized_data.numpy(), scale, zero_point

# 4. 修正反量化函数
def voxel_dequantize(voxel_data, scale, zero_point):
    """从存储的uint8数据+参数恢复uint8体素"""
    voxel_tensor = torch.from_numpy(voxel_data).float()
    return (voxel_tensor - zero_point) * scale  # 反量化公式

def run_preprocessing(all_mol_blocks, atom_to_channel, atom_radius_dict, atom_electronegativity, num_workers):
    """独立的预处理函数，仅使用CPU，不触碰任何GPU资源"""
    pool_args = [
        (mol_block, mol_idx, atom_to_channel, atom_radius_dict, atom_electronegativity)
        for mol_idx, mol_block in enumerate(all_mol_blocks)
    ]
    valid_mol_features = []
    with Pool(processes=num_workers) as pool:
        for result in tqdm(
                pool.imap_unordered(preprocess_mol_block, pool_args),
                total=len(all_mol_blocks),
                desc="多进程预处理"
        ):
            if result is not None:
                valid_mol_features.append(result)
    return valid_mol_features

# 核心函数：批量体素化+多进程预处理
def preprocess_sdf_and_cache(
        sdf_path=DEFAULT_SDF_PATH,
        cache_dir=DEFAULT_CACHE_DIR,
        max_samples=120,
        batch_size=32,
        num_workers=8
):
    # 1. 读取SDF文件并拆分分子块
    start_time = time.time()
    print(f"📥 正在读取SDF文件: {sdf_path}")
    with open(sdf_path, 'r', encoding='utf-8') as f:
        all_mol_blocks = f.read().split("$$$$\n")[:-1]  # 拆分后去掉最后一个空块
    total_mols = len(all_mol_blocks)
    print(f"✅ SDF文件读取完成，共包含 {total_mols} 个分子块（耗时：{time.time() - start_time:.1f}s）")

    # 2. 初始化缓存目录
    os.makedirs(cache_dir, exist_ok=True)
    clear_cache_dir(cache_dir)
    print(f"📦 体素缓存将保存至: {cache_dir}（每{max_samples}个样本/块，批量体素化大小：{batch_size}）")

    # 3. 原子特征映射表（全局统一）
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

    # 4. 多进程预处理分子块（CPU密集型任务并行）
    print(f"\n🔄 开始多进程预处理分子块（共{total_mols}个）...")
    start_preprocess = time.time()


    # 4. 先执行纯CPU预处理（此时完全不初始化GPU）
    valid_mol_features = run_preprocessing(
        all_mol_blocks, atom_to_channel, atom_radius_dict, atom_electronegativity, num_workers
    )


    # 预处理完成统计
    preprocess_time = time.time() - start_preprocess
    print(
        f"✅ 多进程预处理完成，有效分子数：{len(valid_mol_features)}（耗时：{preprocess_time:.1f}s，成功率：{len(valid_mol_features) / total_mols * 100:.1f}%）"
    )

    # 5. 初始化体素化工具（GPU优先，在预处理完成后再初始化，避免多进程冲突）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    voxelizer = Voxelizer(grid_dim=64, device=device).to(device)
    print(f"⚡ 使用 {device} 进行体素化，批量大小：{batch_size}")

    # 6. 批量体素化+缓存（GPU批量处理，主进程执行）
    print(f"\n 开始批量体素化（每批{batch_size}个分子）...")
    start_voxelize = time.time()
    current_chunk = []  # 存储当前缓存块
    chunk_index = 0  # 块编号
    total_valid_voxel = 0  # 有效体素化分子数

    # 按batch_size拆分有效分子
    for i in range(0, len(valid_mol_features), batch_size):
        batch_features = valid_mol_features[i:i + batch_size]
        B = len(batch_features)
        if B == 0:
            continue

        # 6.1 批量构建模型输入（填充不同长度的原子特征，将numpy转为tensor）
        max_atom_num = max(len(feat["coords"]) for feat in batch_features)  # feat["coords"]是numpy数组
        # 初始化批量张量（直接在GPU上创建，避免后续重复迁移）
        device = voxelizer.device  # 复用体素化器的设备（GPU/CPU）
        coords_batch = torch.zeros((B, max_atom_num, 3), dtype=torch.float32, device=device)
        atoms_channel_batch = torch.zeros((B, max_atom_num), dtype=torch.float32, device=device)
        radius_batch = torch.zeros((B, max_atom_num), dtype=torch.float32, device=device)
        electronegativity_batch = torch.zeros((B, max_atom_num), dtype=torch.float32, device=device)
        charge_batch = torch.zeros((B, max_atom_num), dtype=torch.float32, device=device)
        scaffold_weight_batch = torch.zeros((B, max_atom_num), dtype=torch.float32, device=device)

        # 填充数据（直接在GPU上操作，减少CPU-GPU数据传输）
        for b in range(B):
            feat = batch_features[b]
            atom_num = len(feat["coords"])
            # 仅传输有效原子数据，避免空数据迁移
            coords_batch[b, :atom_num] = torch.from_numpy(feat["coords"]).to(device, non_blocking=True)
            atoms_channel_batch[b, :atom_num] = torch.from_numpy(feat["atoms_channel"]).to(device, non_blocking=True)
            radius_batch[b, :atom_num] = torch.from_numpy(feat["radius"]).to(device, non_blocking=True)
            electronegativity_batch[b, :atom_num] = torch.from_numpy(feat["electronegativity"]).to(device,
                                                                                                   non_blocking=True)
            charge_batch[b, :atom_num] = torch.from_numpy(feat["charge"]).to(device, non_blocking=True)
            scaffold_weight_batch[b, :atom_num] = torch.from_numpy(feat["scaffold_weight"]).to(device,
                                                                                               non_blocking=True)

        # 6.2 批量体素化（GPU并行处理，优化显存释放）
        try:
            batch = {
                "coords": coords_batch,
                "atoms_channel": atoms_channel_batch,
                "radius": radius_batch,
                "electronegativity": electronegativity_batch,
                "charge": charge_batch,
                "scaffold_weight": scaffold_weight_batch
            }
            with torch.no_grad():
                voxel_tensor = voxelize_core(voxelizer, batch)  # [B, 40, 64, 64, 64]

            # 6.3 逐个处理样本并释放资源（核心优化：避免存储完整CPU张量）
            for b in range(B):
                # 单样本从GPU传输到CPU并量化，立即释放GPU内存
                with torch.no_grad():
                    single_voxel = voxel_tensor[b].cpu()  # 仅传输当前样本
                    quantized_data, scale, zero_point = voxel_quantize(single_voxel)
                    del single_voxel  # 释放单样本CPU内存

                total_valid_voxel += 1
                current_chunk.append({
                    "voxel_data": quantized_data,
                    "scale": scale,
                    "zero_point": zero_point,
                    "smiles": batch_features[b]["smiles"],
                    "mol_idx": batch_features[b]["mol_idx"],
                    "scaffold_count": batch_features[b]["scaffold_count"]
                })

                # 当缓存块达到阈值时保存
                if len(current_chunk) >= max_samples:
                    chunk_save_path = os.path.join(cache_dir, f"pretrain_chunk_{chunk_index}.npz")
                    # 优化：使用列表推导式高效构建数组
                    voxel_data = np.array([item["voxel_data"] for item in current_chunk])
                    scales = np.array([item["scale"] for item in current_chunk], dtype=np.float32)
                    zero_points = np.array([item["zero_point"] for item in current_chunk], dtype=np.float32)
                    smiles = np.array([item["smiles"] for item in current_chunk], dtype="U256")
                    mol_idx = np.array([item["mol_idx"] for item in current_chunk], dtype=int)
                    scaffold_count_list = np.array([item["scaffold_count"] for item in current_chunk], dtype=int)

                    # 保存时禁用压缩线程锁，加速写入
                    np.savez_compressed(
                        chunk_save_path,
                        voxels_data=voxel_data,
                        scales=scales,
                        zero_points=zero_points,
                        smiles=smiles,
                        mol_idx=mol_idx,
                        scaffold_count=scaffold_count_list,
                        compression=9
                    )

                    print(
                        f"📦 已保存分块 {chunk_index}：{chunk_save_path}（{len(current_chunk)}个样本，累计：{total_valid_voxel}）",
                        flush=True
                    )
                    current_chunk = []
                    chunk_index += 1
                    # 强制释放当前块的内存
                    del voxel_data, scales, zero_points, smiles, mol_idx, scaffold_count_list

            # 批量释放GPU资源（关键：处理完批次后立即清理）
            del voxel_tensor, coords_batch, atoms_channel_batch, radius_batch, electronegativity_batch, charge_batch, scaffold_weight_batch
            torch.cuda.empty_cache()  # 强制释放未使用的显存

        except Exception as e:
            print(f"❌ 批量体素化失败（批次{i // batch_size}）：{str(e)[:100]}（跳过该批次）")
            # 异常时强制清理资源
            if 'voxel_tensor' in locals():
                del voxel_tensor
            del coords_batch, atoms_channel_batch, radius_batch, electronegativity_batch, charge_batch, scaffold_weight_batch
            torch.cuda.empty_cache()
            continue


    # 6.4 处理最后剩余的样本（同步修正，确保与6.3格式一致）
    if len(current_chunk) > 0:
        chunk_save_path = os.path.join(cache_dir, f"pretrain_chunk_{chunk_index}.npz")
        # 提取所有字段（与6.3保持一致）
        smiles= np.array([item["smiles"] for item in current_chunk], dtype="U256")
        mol_idx = np.array([item["mol_idx"] for item in current_chunk], dtype=int)
        scaffold_count_list = np.array([item["scaffold_count"] for item in current_chunk], dtype=int)
        voxels_data = np.array([item["voxel_data"] for item in current_chunk])
        scales = np.array([item["scale"] for item in current_chunk], dtype=np.float32)
        zero_points = np.array([item["zero_point"] for item in current_chunk], dtype=np.float32)
        # 保存
        np.savez_compressed(
            chunk_save_path,
            voxels_data=voxels_data,
            scales=scales,
            zero_points=zero_points,
            smiles=smiles,
            mol_idx=mol_idx,
            scaffold_count=scaffold_count_list
        )
        print(f"📦 已保存最后分块 {chunk_index}：{chunk_save_path}（{len(current_chunk)}个样本）", flush=True)  # 7. 输出最终统计信息
    voxelize_time = time.time() - start_voxelize
    total_time = time.time() - start_time
    print(f"\n 体素化全部完成！")
    print(f" 统计：")
    print(f"  - 原始SDF分子总数：{total_mols}")
    print(f"  - 预处理后有效分子数：{len(valid_mol_features)}")
    print(f"  - 最终体素化成功数：{total_valid_voxel}")
    print(f"  - 总耗时：{total_time:.1f}s（预处理：{preprocess_time:.1f}s，体素化：{voxelize_time:.1f}s）")
    print(f"  - 生成缓存块数量：{chunk_index + 1}（每块{max_samples}个样本）")
    print(f"  - 缓存目录：{cache_dir}")



# 主程序入口（放在函数外部，确保先定义函数再调用）
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate quantized pretraining voxel caches from SDF.")
    parser.add_argument("--sdf-path", default=DEFAULT_SDF_PATH)
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--chunk-size", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=8)
    args = parser.parse_args()
    import multiprocessing as mp
    # 关键防护：强制使用spawn启动子进程，彻底隔离GPU资源（Linux必加，Windows兼容）
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # 若已设置过（如交互式环境），忽略异常
    mp.freeze_support()  # Windows系统添加，Linux/macOS可忽略
    preprocess_sdf_and_cache(
        sdf_path=args.sdf_path,
        cache_dir=args.cache_dir,
        max_samples=args.chunk_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

"""
# 终端输入
screen -ls
# 会显示类似结果
There is a screen on:
    12345.mol_pretrain   (Detached)
1 Socket in /run/screen/S-root.
screen -r 12345
就可以进去了
"""


