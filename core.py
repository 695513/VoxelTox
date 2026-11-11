# -*- coding: utf-8 -*-
"""Shared model/data components. Dataset-specific numbers live in train.py."""

import os
from sklearn.metrics import r2_score

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, IterableDataset
import numpy as np
from tqdm import tqdm
import gc
from sklearn.metrics import roc_auc_score
from transformers import AutoTokenizer, AutoModel
import random
import glob
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau

CFG = {"loss_default_alpha": 0.65}


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


class BatchLoadedVoxelDataset(IterableDataset):

    def __init__(self, cache_root, subset, indices=None, shuffle=False, seed=None):
        super().__init__()
        self._config = dict(CFG)
        self.cache_dir = os.path.join(cache_root, subset)
        self.subset = subset
        self.indices = set(indices) if indices is not None else None
        self.shuffle = shuffle
        self.seed = seed if seed is not None else random.randint(0, 10000)
        self.chunk_files = self._get_chunk_files()
        if not self.chunk_files:
            raise FileNotFoundError(f"子集 {subset} 无分块文件：{self.cache_dir}")
        print(f"📊 {subset} 分块数量：{len(self.chunk_files)}")
        self.num_tasks = None
        self.task_names = None
        self._init_task_info()

    def _get_chunk_files(self):
        """获取所有npz分块文件，按编号排序"""
        chunk_pattern = os.path.join(self.cache_dir, f"{self.subset}_chunk_*.npz")
        chunk_files = glob.glob(chunk_pattern)
        chunk_files.sort(key=lambda x: int(os.path.basename(x).split("_")[-1].split(".")[0]))
        return chunk_files

    def _init_task_info(self):
        """从第一个分块获取任务数量和名称"""
        if not self.chunk_files:
            return
        first_chunk = self.chunk_files[0]
        with np.load(first_chunk, allow_pickle=True) as data:
            self.num_tasks = data["labels"].shape[1]
            self.task_names = [f"task_{i}" for i in range(self.num_tasks)]

    def get_task_info(self):
        return (self.num_tasks, self.task_names)

    def __len__(self):
        return len(self.indices) if self.indices is not None else 0

    def __iter__(self):
        """Dataset-specific legacy behavior is preserved."""
        worker_info = torch.utils.data.get_worker_info()
        chunk_files = self.chunk_files.copy()
        if self.shuffle:
            global_random = random.Random(self.seed)
            global_random.shuffle(chunk_files)
        if worker_info is not None:
            total_workers = worker_info.num_workers
            worker_id = worker_info.id
            chunk_files = [
                chunk_files[i] for i in range(len(chunk_files)) if i % total_workers == worker_id
            ]
            local_seed = self.seed + worker_id
            local_random = random.Random(local_seed)
        else:
            local_random = random.Random(self.seed)
        for chunk_path in chunk_files:
            try:
                with np.load(chunk_path, allow_pickle=True) as data:
                    voxels_arr = data["voxels"]
                    labels_arr = data["labels"]
                    smiles_arr = data["smiles"]
                    indices_arr = data["indices"]
                if self._config["dataset"] in ("sider",):
                    total_samples = len(voxels_arr)
                    sample_indices = list(range(total_samples))
                if self._config["dataset"] in ("bbbp", "bace", "dili", "ames", "clintox", "herg"):
                    molecule_groups = {}
                if self._config["dataset"] in ("bbbp", "bace", "dili", "ames", "clintox", "herg"):
                    for i in range(len(indices_arr)):
                        mol_idx = indices_arr[i]
                        if self.indices is not None and mol_idx not in self.indices:
                            continue
                        if mol_idx not in molecule_groups:
                            molecule_groups[mol_idx] = []
                        molecule_groups[mol_idx].append(i)
                if self._config["dataset"] in ("bbbp", "bace", "dili", "ames", "clintox", "herg"):
                    selected_indices = []
                if self._config["dataset"] in ("bbbp", "bace", "dili", "ames", "clintox", "herg"):
                    for mol_idx in molecule_groups:
                        candidate_indices = molecule_groups[mol_idx]
                        selected_idx = local_random.choice(candidate_indices)
                        selected_indices.append(selected_idx)
                if self.shuffle:
                    local_random.shuffle(
                        selected_indices
                        if self._config["dataset"]
                        in ("bbbp", "bace", "dili", "ames", "clintox", "herg")
                        else sample_indices
                    )
                if self._config["dataset"] in ("sider",):
                    for i in sample_indices:
                        if self.indices is not None and indices_arr[i] not in self.indices:
                            continue
                        voxel = torch.from_numpy(voxels_arr[i]).float()
                        labels = torch.tensor(labels_arr[i], dtype=torch.float32)
                        smiles = smiles_arr[i]
                        yield (voxel, labels, smiles)
                if self._config["dataset"] in ("bbbp", "bace", "dili", "ames", "clintox", "herg"):
                    for i in selected_indices:
                        voxel = torch.from_numpy(voxels_arr[i]).float()
                        labels = torch.tensor(labels_arr[i], dtype=torch.float32)
                        smiles = smiles_arr[i]
                        yield (voxel, labels, smiles)
                if self._config["dataset"] in ("bbbp", "bace", "dili", "ames", "clintox", "herg"):
                    del (
                        voxels_arr,
                        labels_arr,
                        smiles_arr,
                        indices_arr,
                        molecule_groups,
                        selected_indices,
                    )
                else:
                    del voxels_arr, labels_arr, smiles_arr, indices_arr, sample_indices
            except Exception as e:
                print(f"加载分块 {chunk_path} 失败：{e}")
            finally:
                gc.collect()


def minimal_augment(voxel, angle=None, flip_axis=None, epoch=0):
    """允许传入外部生成的旋转角度和翻转轴，确保多分支同步增强"""
    device = voxel.device
    if CFG["dataset"] in ("bbbp", "bace", "dili", "ames", "clintox", "herg"):
        augment_seed = 42 + epoch + hash(voxel.device)
    if CFG["dataset"] in ("bbbp", "bace", "dili", "ames", "clintox", "herg"):
        torch.manual_seed(augment_seed)
    if angle is None:
        angle = (torch.rand(1, device=device) * 2 - 1) * 0.052
    if torch.rand(1, device=device) < 0.3:
        B, C, D, H, W = voxel.shape
        rot_matrix = torch.tensor(
            [
                [torch.cos(angle), -torch.sin(angle), 0],
                [torch.sin(angle), torch.cos(angle), 0],
                [0, 0, 1],
            ],
            device=device,
            dtype=voxel.dtype,
        )
        trans_col = torch.zeros((3, 1), device=device, dtype=voxel.dtype)
        theta_3x4 = torch.cat([rot_matrix, trans_col], dim=1)
        theta = theta_3x4.unsqueeze(0).repeat(B, 1, 1)
        grid = F.affine_grid(theta, voxel.shape, align_corners=True)
        voxel = F.grid_sample(voxel, grid, mode="nearest", padding_mode="zeros", align_corners=True)
    if flip_axis is None:
        flip_axis = 2 if torch.rand(1) < 0.5 else 3
    if torch.rand(1, device=device) < 0.25:
        voxel = voxel.flip(dims=[flip_axis])
    return voxel


class InfoNCEContrastiveLoss(nn.Module):

    def __init__(self, temperature=0.5, num_neg=5):
        super().__init__()
        self.temp = temperature
        self.num_neg = num_neg

    def forward(self, z1, z2):
        """
        输入：
            z1: 模态1特征（如体素全局特征），[B, D]
            z2: 模态2特征（如ChemBERTa特征），[B, D]
        输出：
            跨模态InfoNCE损失（无标签依赖，避开-1问题）
        """
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)
        B, D = z1.shape
        device = z1.device
        sim_matrix = torch.matmul(z1, z2.T) / self.temp
        pos_sim = sim_matrix.diag()
        mask = torch.eye(B, dtype=torch.bool, device=device)
        sim_matrix_masked = sim_matrix.masked_fill(mask, -float("inf"))
        max_possible_neg = B - 1
        k = min(self.num_neg, max_possible_neg)
        if k <= 0:
            return torch.tensor(0.0, device=device)
        neg_sim_topk, _ = torch.topk(sim_matrix_masked, k=k, dim=1)
        exp_pos = torch.exp(pos_sim)
        exp_neg = torch.exp(neg_sim_topk).sum(dim=1)
        loss = -torch.log(exp_pos / (exp_pos + exp_neg + 1e-08)).mean()
        return loss


class TaskHead(nn.Module):

    def __init__(self, in_dim=512, num_tasks=1, task_type="classification"):
        super().__init__()
        self.num_tasks = num_tasks
        self.task_type = task_type
        self.shared = nn.Sequential(
            nn.Linear(in_dim, 128), nn.GELU(), nn.Dropout(CFG["head_dropout"]), nn.LayerNorm(128)
        )
        if num_tasks == 1:
            self.task_attn = nn.Parameter(torch.tensor(1.0))
        else:
            self.task_attn = nn.Linear(128, num_tasks)
        self.out = nn.Linear(128, num_tasks)

    def forward(self, x):
        shared_feat = self.shared(x)
        if self.num_tasks == 1:
            attn = torch.sigmoid(self.task_attn)
            attn = attn.unsqueeze(0).unsqueeze(0)
        else:
            attn = torch.sigmoid(self.task_attn(shared_feat))
        output = self.out(shared_feat) * attn
        return output


def bce_with_label_smoothing(logits, targets, smoothing=0.05, alpha=CFG["loss_default_alpha"]):
    with torch.no_grad():
        targets = targets * (1 - smoothing) + 0.5 * smoothing
    ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    if alpha >= 0:
        alpha_t = torch.where(targets > 0.5, alpha, 1 - alpha)
        ce_loss = alpha_t * ce_loss
    return ce_loss.mean()


class CrossConvAttention3D(nn.Module):

    def __init__(self, in_channels_list, out_channels):
        super().__init__()
        self.branch_convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv3d(in_ch, out_channels, kernel_size=1, bias=False),
                    nn.BatchNorm3d(out_channels),
                    nn.ReLU(inplace=True),
                )
                for in_ch in in_channels_list
            ]
        )
        self.branch_channel_attn = nn.ModuleList(
            [
                nn.Sequential(
                    nn.AdaptiveAvgPool3d(1),
                    nn.Conv3d(out_channels, out_channels, kernel_size=1, bias=False),
                    nn.Sigmoid(),
                )
                for _ in in_channels_list
            ]
        )
        self.spatial_attn_gate = nn.Sequential(
            nn.Conv3d(out_channels * 6, len(in_channels_list), kernel_size=1, bias=False),
            nn.Softmax(dim=1),
        )

    def forward(self, features_list):
        compressed_feats = [conv(feat) for conv, feat in zip(self.branch_convs, features_list)]
        branch_channel_weights = []
        for i in range(len(compressed_feats)):
            channel_weight = self.branch_channel_attn[i](compressed_feats[i])
            branch_global_weight = channel_weight.mean(dim=1, keepdim=False)
            branch_channel_weights.append(branch_global_weight)
        branch_global_stack = torch.stack(branch_channel_weights, dim=1)
        branch_global_norm = torch.softmax(branch_global_stack, dim=1)
        branch_channel_weights = [branch_global_norm[:, i : i + 1] for i in range(3)]
        spatial_electronic = compressed_feats[0] * compressed_feats[1]
        spatial_scaffold = compressed_feats[0] * compressed_feats[2]
        electronic_scaffold = compressed_feats[1] * compressed_feats[2]
        concat_feats = torch.cat(
            [
                compressed_feats[0],
                compressed_feats[1],
                compressed_feats[2],
                spatial_electronic,
                spatial_scaffold,
                electronic_scaffold,
            ],
            dim=1,
        )
        spatial_attn_weights = self.spatial_attn_gate(concat_feats)
        weighted_feats = []
        for i in range(len(compressed_feats)):
            combined_weight = branch_channel_weights[i] * spatial_attn_weights[:, i : i + 1]
            weighted_feat = compressed_feats[i] * combined_weight
            weighted_feats.append(weighted_feat)
        return weighted_feats[0] + weighted_feats[1] + weighted_feats[2]


class DepthwiseResidualBlock3D(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.main_path = nn.Sequential(
            nn.Conv3d(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=in_channels,
                bias=False,
            ),
            nn.BatchNorm3d(in_channels),
            nn.ReLU(inplace=True),
            nn.Dropout3d(CFG["residual_dropout"]),
            nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm3d(out_channels),
        )
        self.residual_path = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.residual_path = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels),
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.residual_path(x)
        out = self.main_path(x)
        out += residual
        return self.relu(out)


class DualBranchResNet3D(nn.Module):

    def __init__(self, spatial_in_ch=10, electronic_in_ch=20, scaffold_in_ch=10, num_tasks=1):
        super().__init__()
        self.spatial_branch = nn.Sequential(
            nn.Conv3d(spatial_in_ch, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            DepthwiseResidualBlock3D(16, 16, stride=1),
            DepthwiseResidualBlock3D(16, 32, stride=2),
        )
        self.electronic_branch = nn.Sequential(
            nn.Conv3d(electronic_in_ch, 24, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(24),
            nn.ReLU(inplace=True),
            DepthwiseResidualBlock3D(24, 24, stride=1),
            DepthwiseResidualBlock3D(24, 48, stride=2),
        )
        self.scaffold_branch = nn.Sequential(
            nn.Conv3d(scaffold_in_ch, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            DepthwiseResidualBlock3D(16, 16, stride=1),
            DepthwiseResidualBlock3D(16, 32, stride=2),
        )
        self.cross_attn = CrossConvAttention3D(in_channels_list=[32, 48, 32], out_channels=32)
        self.cross_attn_drop = nn.Dropout3d(0.2)
        self.fusion_conv = nn.Sequential(
            DepthwiseResidualBlock3D(32, 112, stride=1), nn.Dropout3d(CFG["fusion_3d_dropout"])
        )
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.proj_3d = nn.Sequential(nn.Linear(112, 128))
        self.proj_chem = nn.Sequential(nn.Linear(384, 128))
        self.contrast_criterion = InfoNCEContrastiveLoss(temperature=0.5)
        self.ln_3d = nn.LayerNorm(112, eps=1e-06)
        self.ln_chem = nn.LayerNorm(384, eps=1e-06)
        self.chem_attention = nn.Sequential(nn.Linear(384, 384), nn.Sigmoid())
        self.ln_atomic = nn.LayerNorm(56, eps=1e-06)
        self.first_drop = nn.Dropout(0.1)
        self.ln_group = nn.LayerNorm(64, eps=1e-06)
        self.second_drop = nn.Dropout(0.1)
        self.scale_proj = nn.Linear(112, 64)
        if CFG["dataset"] in ("sider",):
            nn.init.ones_(self.ln_atomic.weight)
            nn.init.zeros_(self.ln_atomic.bias)
            nn.init.ones_(self.ln_group.weight)
            nn.init.zeros_(self.ln_group.bias)
            nn.init.xavier_uniform_(self.scale_proj.weight)
            nn.init.zeros_(self.scale_proj.bias)

    def forward(self, spatial_x, electronic_x, scaffold_x, chem_features, labels=None):
        spatial_atomic = self.spatial_branch[:4](spatial_x)
        electronic_atomic = self.electronic_branch[:4](electronic_x)
        scaffold_atomic = self.scaffold_branch[:4](scaffold_x)
        B = spatial_atomic.shape[0]
        spatial_atomic_feat = self.global_avg_pool(spatial_atomic).view(B, 16)
        electronic_atomic_feat = self.global_avg_pool(electronic_atomic).view(B, 24)
        scaffold_atomic_feat = self.global_avg_pool(scaffold_atomic).view(B, 16)
        atomic_feat = torch.cat(
            [spatial_atomic_feat, electronic_atomic_feat, scaffold_atomic_feat], dim=1
        )
        atomic_feat = self.ln_atomic(atomic_feat)
        atomic_feat = self.first_drop(atomic_feat)
        spatial_group = self.spatial_branch(spatial_x)
        electronic_group = self.electronic_branch(electronic_x)
        scaffold_group = self.scaffold_branch(scaffold_x)
        group_feat = torch.cat(
            [
                self.global_avg_pool(spatial_group).view(-1, 32),
                self.global_avg_pool(electronic_group).view(-1, 48),
                self.global_avg_pool(scaffold_group).view(-1, 32),
            ],
            dim=1,
        )
        group_feat = self.scale_proj(group_feat)
        group_feat = self.ln_group(group_feat)
        group_feat = self.second_drop(group_feat)
        fused_3d = self.cross_attn([spatial_group, electronic_group, scaffold_group])
        fused_3d = self.cross_attn_drop(fused_3d)
        fused_3d = self.fusion_conv(fused_3d)
        global_feat_112 = self.global_avg_pool(fused_3d).view(-1, 112)
        global_feat_64 = self.scale_proj(global_feat_112)
        global_feat_64 = self.ln_group(global_feat_64)
        chem_weight = self.chem_attention(chem_features)
        chem_feat_weighted = chem_features * chem_weight
        chem_feat_norm = self.ln_chem(chem_feat_weighted)
        proj_3d = self.proj_3d(global_feat_112)
        proj_chem = self.proj_chem(chem_feat_norm)
        return (atomic_feat, group_feat, global_feat_64, chem_feat_norm, proj_3d, proj_chem)


class UniversalMolecularModel(nn.Module):

    def __init__(self, num_tasks=1, task_type="classification", dataset_name=None):
        super().__init__()
        self.feature_extractor = DualBranchResNet3D(num_tasks=num_tasks)
        self.task_type = task_type
        self.task_head = TaskHead(
            in_dim=CFG["fusion_dim"], num_tasks=num_tasks, task_type=task_type
        )
        self.scale_attn = nn.Linear(56 + 64 + 64 + 384, 4)
        self.multi_scale_fusion = nn.Sequential(
            nn.Linear(56 + 64 + 64 + 384, CFG["fusion_dim"]),
            nn.GELU(),
            nn.LayerNorm(CFG["fusion_dim"]),
            nn.Dropout(CFG["fusion_dropout"]),
        )
        self.dataset_name = dataset_name
        self.residual_proj = nn.Linear(56 + 64 + 64 + 384, CFG["fusion_dim"])

    def forward(self, spatial_x, electronic_x, scaffold_x, chem_features, labels=None):
        atomic_feat, group_feat, global_feat, chem_feat_fusion, proj_3d, proj_chem = (
            self.feature_extractor(spatial_x, electronic_x, scaffold_x, chem_features)
        )
        feat_concat = torch.cat([atomic_feat, group_feat, global_feat, chem_feat_fusion], dim=1)
        scale_weights = torch.softmax(self.scale_attn(feat_concat), dim=1)
        weighted_atomic = atomic_feat * scale_weights[:, 0].unsqueeze(1)
        weighted_group = group_feat * scale_weights[:, 1].unsqueeze(1)
        weighted_global = global_feat * scale_weights[:, 2].unsqueeze(1)
        weighted_chem = chem_feat_fusion * scale_weights[:, 3].unsqueeze(1)
        fused_feat = torch.cat(
            [weighted_atomic, weighted_group, weighted_global, weighted_chem], dim=1
        )
        residual = self.residual_proj(feat_concat)
        fused_feat = self.multi_scale_fusion(fused_feat) + residual
        output = self.task_head(fused_feat)
        if labels is not None:
            main_loss = (
                bce_with_label_smoothing(output, labels, smoothing=0.05)
                if CFG["dataset"] in ("bbbp", "bace", "dili", "ames", "clintox", "herg")
                else bce_with_label_smoothing(output, labels, smoothing=0.0, alpha=0.5)
            )
            if self.training:
                contrast_loss = self.feature_extractor.contrast_criterion(proj_3d, proj_chem)
                total_loss = main_loss + CFG["contrast_weight"] * contrast_loss
                return (output, total_loss)
            else:
                return (output, main_loss)
        else:
            return output


class ChemBERTaTeacher:

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(CFG.get("chemberta_path", "DeepChem/ChemBERTa-77M-MLM"))
        self.model = AutoModel.from_pretrained(CFG.get("chemberta_path", "DeepChem/ChemBERTa-77M-MLM"))
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def get_features(self, smiles_list, device=None):
        """提取SMILES的特征向量"""
        if CFG["dataset"] in ("sider",):
            self.model.eval()
        with torch.no_grad():
            inputs = self.tokenizer(
                smiles_list, padding=True, truncation=True, return_tensors="pt"
            ).to(self.device)
            outputs = self.model(**inputs)
            features = outputs.last_hidden_state.mean(dim=1)
            features = F.layer_norm(features, features.shape[1:])
        return features


def collate_fn(batch):
    valid_samples = [
        (item[0], item[1], item[2])
        for item in batch
        if item[0] is not None and item[1] is not None and (torch.numel(item[0]) > 0)
    ]
    if not valid_samples:
        return (None, None, None)
    voxels = torch.stack([s[0] for s in valid_samples])
    labels = torch.stack([s[1] for s in valid_samples])
    smiles = [s[2] for s in valid_samples]
    return (voxels, labels, smiles)


def get_branch_global_weights(model, spatial_x, electronic_x, scaffold_x):
    """
    提取3个分支的全局归一化权重（空间/电子/支架）
    参数：
        model: 训练中的DualBranchResNet3D模型
        spatial_x/electronic_x/scaffold_x: 一个批次的3个分支输入（需在设备上）
    返回：
        weights: 列表，[空间分支权重均值, 电子分支权重均值, 支架分支权重均值]
    """
    model.eval()
    with torch.no_grad():
        spatial_feat = model.spatial_branch(spatial_x)
        electronic_feat = model.electronic_branch(electronic_x)
        scaffold_feat = model.scaffold_branch(scaffold_x)
        features_list = [spatial_feat, electronic_feat, scaffold_feat]
        compressed_feats = [
            conv(feat) for conv, feat in zip(model.cross_attn.branch_convs, features_list)
        ]
        branch_channel_weights = []
        for i in range(len(compressed_feats)):
            channel_weight = model.cross_attn.branch_channel_attn[i](compressed_feats[i])
            branch_global_weight = channel_weight.mean(dim=1, keepdim=False)
            branch_channel_weights.append(branch_global_weight)
        branch_global_stack = torch.stack(branch_channel_weights, dim=1)
        branch_global_norm = torch.softmax(branch_global_stack, dim=1)
        spatial_weight = branch_global_norm[:, 0].mean().item()
        electronic_weight = branch_global_norm[:, 1].mean().item()
        scaffold_weight = branch_global_norm[:, 2].mean().item()
    model.train()
    return [round(spatial_weight, 4), round(electronic_weight, 4), round(scaffold_weight, 4)]


def configure(config):
    """Select settings before constructing models or data loaders."""
    CFG.clear()
    CFG.update(config)
    bce_with_label_smoothing.__defaults__ = (0.05, CFG["loss_default_alpha"])
