import os
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

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
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import CosineAnnealingLR #余弦退火调度器（CosineAnnealingLR），收敛更快且不易陷入局部最优：

class NPZBatchLoadedVoxelDataset(IterableDataset):
    def __init__(self, subset, chunk_files=None,  # chunk_files参数，用于指定使用的缓存块
                 shuffle=False, seed=None, is_pretrain=False, mask_ratio=0.3):
        super().__init__()
        self.chunk_files = chunk_files  # 直接传入分块路径列表，不校验
        self.subset = subset
        self.shuffle = shuffle
        self.seed = seed if seed is not None else random.randint(0, 10000)
        self.is_pretrain = is_pretrain
        self.mask_ratio = mask_ratio
        # 动态掩码率参数（30%起始，60%最高）
        # 统计总样本数
        self.total_samples = 300000
        # 硬编码固定样本数：训练集284000，验证集16000
        if subset == "train":
            self.total_samples = 284000  # 训练集固定28w4
        elif subset == "val":
            self.total_samples = 16000   # 验证集固定1w6
        else:
            self.total_samples = 0  # 其他子集默认0

        self.epoch = 0  # 初始化为0


    def _dequantize_voxel(self, voxel_uint8, scale, zero_point):
        """核心：将uint8体素反量化为float32（恢复原始数值范围）"""
        voxel_float = (voxel_uint8.astype(np.float32) - zero_point) * scale
        # 可选：归一化到0-1（避免数值波动，稳定训练）
        voxel_float = (voxel_float - voxel_float.min()) / (voxel_float.max() - voxel_float.min() + 1e-8)
        return voxel_float
    def __len__(self):
        return self.total_samples

    def __iter__(self):

        worker_info = torch.utils.data.get_worker_info()
        chunk_files = self.chunk_files.copy()

        # 1. 全局分块打乱（复用原逻辑）
        if self.shuffle:
            global_random = random.Random(self.seed)
            global_random.shuffle(chunk_files)

        # 2. 多进程分块分片（复用原逻辑）
        if worker_info is not None:
            total_workers = worker_info.num_workers
            worker_id = worker_info.id
            chunk_files = [chunk_files[i] for i in range(len(chunk_files)) if i % total_workers == worker_id]
            local_seed = self.seed + worker_id
            local_random = random.Random(local_seed)
        else:
            local_random = random.Random(self.seed)

        # 3. 加载npz分块并处理
        for chunk_path in chunk_files:
            try:
                # 加载npz文件（仅读取需要的键）
                npz_data = np.load(chunk_path, mmap_mode='r')  # 内存映射，不加载全量数据
                voxels_uint8 = npz_data["voxels_data"]  # uint8体素（40,64,64,64）
                scales = npz_data["scales"]  # 反量化参数：scale
                zero_points = npz_data["zero_points"]  # 反量化参数：zero_point
                smiles_list = npz_data["smiles"]  # SMILES列表
                mol_indices = npz_data["mol_idx"]  # 分子索引（可选）

                # 若有labels，读取labels（无则注释）
                labels_list = npz_data["labels"] if "labels" in npz_data.files else None


                # 分块内部打乱（样本级）
                sample_indices = list(range(len(voxels_uint8)))
                if self.shuffle:
                    local_random.shuffle(sample_indices)

                # 逐个处理样本
                for idx in sample_indices:
                    # 反量化：uint8 → float32
                    original_voxel = self._dequantize_voxel(
                        voxels_uint8[idx], scales[idx], zero_points[idx]
                    )
                    # 转为torch tensor（匹配模型输入格式：[C,D,H,W]）
                    original_voxel = torch.from_numpy(original_voxel).float()  # [40,64,64,64]

                    # 拆分3个分支（空间10+电子20+支架10，匹配DualBranchResNet3D输入）
                    spatial_voxel = original_voxel[0:10, :, :, :]  # [10,64,64,64]
                    electronic_voxel = original_voxel[10:30, :, :, :]  # [20,64,64,64]
                    scaffold_voxel = original_voxel[30:40, :, :, :]  # [10,64,64,64]

                    # 处理labels（无则设为None）
                    labels = torch.tensor(labels_list[idx], dtype=torch.float32) if labels_list is not None else None
                    smiles = smiles_list[idx]

                    # 预训练模式：返回3分支体素+掩码+labels+SMILES
                    if self.is_pretrain:
                        # 随epoch提高掩码率（最高70%）
                        current_mask_ratio = min(0.3 + self.epoch * 0.015, 0.6)  # 每个epoch增加1%
                        mask = torch.rand_like(original_voxel) < current_mask_ratio
                        masked_voxel = (~mask).float()
                        # 返回3分支体素（模型输入）+ 原始体素（计算损失）+ 掩码 + labels + SMILES
                        yield spatial_voxel, electronic_voxel, scaffold_voxel, original_voxel, masked_voxel, smiles
                    # 下游任务模式：返回3分支体素+labels+SMILES
                    else:
                        yield spatial_voxel, electronic_voxel, scaffold_voxel, labels, smiles



            except Exception as e:
                print(f"加载分块 {chunk_path} 失败：{str(e)[:100]}")
            finally:
                # 及时释放npz数据占用的内存
                npz_data = None
                gc.collect()
        return


def minimal_augment(voxel):
    """仅保留随机小角度旋转（绕z轴±3°），耗时可忽略（训练时使用，体素化缓存时不启用）"""
    if torch.rand(1, device=voxel.device) < 0.3:  # 30%概率增强，设备与voxel一致
        angle = (torch.rand(1, device=voxel.device) * 2 - 1) * 0.052  # 均匀分布±3°，设备一致
        # print("angle 设备：", angle.device)  # 应输出 "cuda:0"（若用GPU）
        # print("angle 类型：", angle.dtype)  # 应输出 "torch.float32"
        # print("torch.cos(angle) 设备：", torch.cos(angle).device)  # 应与 angle 一致
        B, C, D, H, W = voxel.shape  # B=批次大小，C=通道数，D/H/W=3D体素的深度/高度/宽度
        #构造 3×4 仿射矩阵（3行=旋转，4列=旋转+平移）
        # 1. 构造 3×3 旋转矩阵（绕z轴）
        rot_matrix = torch.tensor([
            [torch.cos(angle), -torch.sin(angle), 0],
            [torch.sin(angle), torch.cos(angle), 0],
            [0, 0, 1]
        ], device=voxel.device, dtype=voxel.dtype)  # 保持在GPU上

        # 2. 新增第4列（平移参数，全0代表无平移），形成 3×4 矩阵
        trans_col = torch.zeros((3, 1), device=voxel.device, dtype=voxel.dtype)  # 3行1列，全0
        theta_3x4 = torch.cat([rot_matrix, trans_col], dim=1)  # 拼接为 3×4 矩阵
        # 3. 扩展为批次维度（B×3×4），匹配输入体素的批次大小
        theta = theta_3x4.unsqueeze(0).repeat(B, 1, 1)  # 最终形状：[B, 3, 4]
        grid = F.affine_grid(theta, voxel.shape, align_corners=True)
        voxel = F.grid_sample(voxel, grid, mode='nearest', padding_mode='zeros', align_corners=True)# trilinear（更好） 替换为 nearest（最近邻插值）
    return voxel

# 简化版InfoNCE损失（适配小样本场景）
class InfoNCEContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.5, num_neg=5):
        super().__init__()
        self.temp = temperature  # 温度系数，小样本用较高值（0.5）增强区分度
        self.num_neg = num_neg  # 每样本选择的负样本数量，减少计算量

    def forward(self, z1, z2):
        # 特征归一化，确保相似度计算公平
        z1 = F.normalize(z1, dim=1)  # [B, 64]
        z2 = F.normalize(z2, dim=1)  # [B, 64]

        # 计算相似度矩阵（B×B），对角线为正样本对
        sim_matrix = torch.matmul(z1, z2.T) / self.temp  # 除以温度系数缩放
        B = sim_matrix.size(0)

        # 提取正样本相似度（对角线元素）
        pos_sim = sim_matrix.diag()  # [B]

        # 提取负样本相似度（每样本选前num_neg个非对角线元素）
        neg_sim = []
        for i in range(B):
            mask = ~torch.eye(B, dtype=bool, device=sim_matrix.device)[i]  # 显式指定设备，避免跨设备错误
            non_self_sim = sim_matrix[i, mask]  # 所有非自身样本的相似度

            # 关键优化：按相似度从小到大排序，优先选与正样本差异大的负样本
            # 理由：对比学习需要让模型区分“相似样本”和“不相似样本”，差异大的负样本更能增强判别性
            sorted_neg_sim = torch.sort(non_self_sim, descending=False).values  # 升序排列（相似度从低到高）

            if len(sorted_neg_sim) >= self.num_neg:
                # 选前num_neg个（相似度最低的），而非随机选
                neg_sim.append(sorted_neg_sim[:self.num_neg].mean())
            else:
                neg_sim.append(sorted_neg_sim.mean())  # 样本不足时取全部
        neg_sim = torch.stack(neg_sim)  # [B]

        # 二元交叉熵：让正样本得分 - 负样本得分 趋近于1
        return F.binary_cross_entropy_with_logits(
            pos_sim - neg_sim,  # 正样本应比负样本得分高
            torch.ones_like(pos_sim)  # 目标值为1
        )

# --------------------------
# 3. 3D特征提取模块
# --------------------------
class CrossConvAttention3D(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super().__init__()
        self.branch_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(in_ch, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=True)
            ) for in_ch in in_channels_list
        ])
        self.branch_channel_attn = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool3d(1),
                nn.Conv3d(out_channels, out_channels, kernel_size=1, bias=False),
                nn.Sigmoid()
            ) for _ in in_channels_list
        ])
        self.spatial_attn_gate = nn.Sequential(
            nn.Conv3d(out_channels * 6, len(in_channels_list), kernel_size=1, bias=False),
            nn.Softmax(dim=1)
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
        branch_channel_weights = [branch_global_norm[:, i:i + 1] for i in range(3)]

        spatial_electronic = compressed_feats[0] * compressed_feats[1]
        spatial_scaffold = compressed_feats[0] * compressed_feats[2]
        electronic_scaffold = compressed_feats[1] * compressed_feats[2]
        concat_feats = torch.cat([
            compressed_feats[0], compressed_feats[1], compressed_feats[2],
            spatial_electronic, spatial_scaffold, electronic_scaffold
        ], dim=1)
        spatial_attn_weights = self.spatial_attn_gate(concat_feats)

        weighted_feats = []
        for i in range(len(compressed_feats)):
            combined_weight = branch_channel_weights[i] * spatial_attn_weights[:, i:i + 1]
            weighted_feat = compressed_feats[i] * combined_weight
            weighted_feats.append(weighted_feat)
        return weighted_feats[0] + weighted_feats[1] + weighted_feats[2]


class DepthwiseResidualBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.main_path = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=3, stride=stride,
                      padding=1, groups=in_channels, bias=False),
            nn.BatchNorm3d(in_channels),
            nn.ReLU(inplace=True),
            nn.Dropout3d(0.3),
            nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm3d(out_channels)
        )
        self.residual_path = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.residual_path = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.residual_path(x)
        out = self.main_path(x)
        out += residual
        return self.relu(out)


class DualBranchResNet3D(nn.Module):
    def __init__(self, spatial_in_ch=10, electronic_in_ch=20, scaffold_in_ch=10, num_tasks=1, use_contrast=True):
        super().__init__()
        # 原有3分支、交叉注意力、融合层结构完全保留（不变）
        self.spatial_branch = nn.Sequential(
            nn.Conv3d(spatial_in_ch, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            DepthwiseResidualBlock3D(16, 16, stride=1),
            DepthwiseResidualBlock3D(16, 32, stride=2)
        )
        self.electronic_branch = nn.Sequential(
            nn.Conv3d(electronic_in_ch, 24, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(24),
            nn.ReLU(inplace=True),
            DepthwiseResidualBlock3D(24, 24, stride=1),
            DepthwiseResidualBlock3D(24, 48, stride=2)
        )
        self.scaffold_branch = nn.Sequential(
            nn.Conv3d(scaffold_in_ch, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            DepthwiseResidualBlock3D(16, 16, stride=1),
            DepthwiseResidualBlock3D(16, 32, stride=2)
        )
        self.cross_attn = CrossConvAttention3D(
            in_channels_list=[32, 48, 32],
            out_channels=32
        )
        self.fusion_conv = nn.Sequential(
            DepthwiseResidualBlock3D(32, 112, stride=1),
            nn.Dropout3d(0.2)
        )
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

        self.use_contrast = use_contrast
        self.ln_3d = nn.LayerNorm(112, eps=1e-6)
        self.ln_chem = nn.LayerNorm(384, eps=1e-6)
        self.chem_attention = nn.Sequential(
            nn.Linear(384, 384),
            nn.Sigmoid()
        )

        # 对比学习投影头（下游任务用，预训练时也可复用）
        if self.use_contrast:
            self.proj_3d = nn.Sequential(nn.Linear(112, 64), nn.ReLU())
            self.proj_chem = nn.Sequential(nn.Linear(384, 64), nn.ReLU())
            self.contrast_criterion = InfoNCEContrastiveLoss(temperature=0.5, num_neg=5)

        # 下游任务分类头（预训练时不启用）
        self.fc = nn.Sequential(
            nn.Linear(112 + 384, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_tasks),
        )

    def forward(self, spatial_x, electronic_x, scaffold_x, chem_features, labels=None, return_feat=False):
        # 1. 3D特征提取（核心：返回未经过分类头的原始特征）
        spatial_feat = self.spatial_branch(spatial_x)
        electronic_feat = self.electronic_branch(electronic_x)
        scaffold_feat = self.scaffold_branch(scaffold_x)
        fused_3d = self.cross_attn([spatial_feat, electronic_feat, scaffold_feat])
        fused_3d = self.fusion_conv(fused_3d)
        pooled_3d = self.global_avg_pool(fused_3d).view(fused_3d.size(0), -1)  # [B, 112]（预训练需要的3D原始特征）
        fused_3d_norm = self.ln_3d(pooled_3d)

        # 2. 化学特征处理（返回处理后的化学特征）
        chem_weight = self.chem_attention(chem_features)
        chem_feat_weighted = chem_features * chem_weight
        chem_feat_norm = self.ln_chem(chem_feat_weighted)  # [B, 384]（预训练需要的化学特征）

        # --------------------------
        # 新增：return_feat=True时，返回原始特征（供预训练使用）
        # --------------------------
        if return_feat:
            return pooled_3d, chem_feat_norm  # pooled_3d:112维（体素重建用）；chem_feat_norm:384维（对齐用）

        # 3. 下游任务逻辑（预训练时不触发）
        final_feat = torch.cat([fused_3d_norm, chem_feat_norm], dim=1)
        cls_out = self.fc(final_feat)

        # 下游任务计算损失（预训练时不执行）
        if labels is not None and self.use_contrast:
            proj_3d = self.proj_3d(fused_3d_norm)
            proj_chem = self.proj_chem(chem_feat_norm)
            cls_loss = nn.BCEWithLogitsLoss()(cls_out, labels)
            contrast_loss = self.contrast_criterion(proj_3d, proj_chem)
            total_loss = cls_loss + 0.05 * contrast_loss
            return cls_out, total_loss, cls_loss, contrast_loss

        return cls_out


# --------------------------
# 4. ChemBERTa特征提取
# --------------------------
class ChemBERTaTeacher:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-77M-MLM")
        self.model = AutoModel.from_pretrained("DeepChem/ChemBERTa-77M-MLM")
        self.model.eval()  # 不训练，仅作为特征提取
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def get_features(self, smiles_list, device=None):
        target_device = device if device is not None else self.device  # 优先用传入的设备
        with torch.no_grad():
            inputs = self.tokenizer(smiles_list, padding=True, truncation=True, return_tensors="pt").to(target_device)
            outputs = self.model(** inputs)
            features = outputs.last_hidden_state.mean(dim=1)  # [B, 384]
        return features

def get_branch_global_weights(model, spatial_x, electronic_x, scaffold_x):
    """
    提取3个分支的全局归一化权重（空间/电子/支架）
    参数：
        model: 训练中的DualBranchResNet3D模型
        spatial_x/electronic_x/scaffold_x: 一个批次的3个分支输入（需在设备上）
    返回：
        weights: 列表，[空间分支权重均值, 电子分支权重均值, 支架分支权重均值]
    """
    # 1. 进入评估模式，关闭梯度计算（仅提取权重，不影响训练）
    model.eval()
    with torch.no_grad():
        # 2. 提取3个分支的中间特征（复用模型的分支结构）
        spatial_feat = model.spatial_branch(spatial_x)
        electronic_feat = model.electronic_branch(electronic_x)
        scaffold_feat = model.scaffold_branch(scaffold_x)
        features_list = [spatial_feat, electronic_feat, scaffold_feat]

        # 3. 调用注意力模块的前向计算，到“分支权重归一化”步骤（复用已有逻辑）
        # 3.1 先压缩特征通道（和原逻辑一致）
        compressed_feats = [conv(feat) for conv, feat in zip(model.cross_attn.branch_convs, features_list)]
        # 3.2 计算每个分支的全局权重（和原逻辑一致）
        branch_channel_weights = []
        for i in range(len(compressed_feats)):
            channel_weight = model.cross_attn.branch_channel_attn[i](compressed_feats[i])
            branch_global_weight = channel_weight.mean(dim=1, keepdim=False)  # [B,1,1,1,1]
            branch_channel_weights.append(branch_global_weight)
        # 3.3 提取归一化后的全局权重（核心！这就是最终的分支全局重要性）
        branch_global_stack = torch.stack(branch_channel_weights, dim=1)  # [B,3,1,1,1]
        branch_global_norm = torch.softmax(branch_global_stack, dim=1)  # 归一化后的权重

        # 4. 计算当前批次的权重均值（避免单样本波动，更具参考性）
        spatial_weight = branch_global_norm[:, 0].mean().item()  # 空间分支权重均值
        electronic_weight = branch_global_norm[:, 1].mean().item()  # 电子分支权重均值
        scaffold_weight = branch_global_norm[:, 2].mean().item()  # 支架分支权重均值

    # 恢复模型训练模式（不影响后续训练）
    model.train()
    return [round(spatial_weight, 4), round(electronic_weight, 4), round(scaffold_weight, 4)]


class PretrainModelWithMultiModalAlign(nn.Module):
    def __init__(self):
        super().__init__()
        # 1. 3D编码器（直接复用DualBranchResNet3D的基础版交叉注意力，不替换！）
        self.voxel_encoder = DualBranchResNet3D(
            spatial_in_ch=10, electronic_in_ch=20, scaffold_in_ch=10,
            use_contrast=False  # 关闭编码器自带的对比逻辑，用预训练自定义的对比头
        )
        # 2. 体素重建头（保持不变，适配3D特征112维+化学特征384维）
        self.recon_head = nn.Sequential(
            nn.Linear(112 + 384, 512),  # 3D原始特征(112) + 处理后化学特征(384) 联合重建
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 40 * 8 * 8 * 8),  # 输出40通道（10空间+20电子+10支架）、8×8×8特征图
            nn.Unflatten(1, (40, 8, 8, 8)),
            # 新增：对通道维度做注意力，突出药物关键通道（如电子通道）
            nn.Conv3d(40, 40, kernel_size=1),  # 1×1卷积调整通道权重
            nn.ConvTranspose3d(40, 40, kernel_size=2, stride=2),  # 8→16（上采样到16×16×16）
            nn.BatchNorm3d(40),
            nn.GELU(),
            nn.ConvTranspose3d(40, 40, kernel_size=2, stride=2),  # 16→32（最终匹配输入体素尺寸）
            nn.BatchNorm3d(40),
            nn.GELU(),
            nn.ConvTranspose3d(40, 40, kernel_size=2, stride=2)  # 32→64（3次，最终匹配输入尺寸）
        )

        # 3. 跨模态对齐投影头（保持不变，确保3D和化学特征映射到同一64维空间）
        self.proj_3d = nn.Linear(112, 64)  # 3D原始特征（112维）→ 64维对比空间
        self.proj_chem = nn.Linear(384, 64)  # 处理后化学特征（384维）→ 64维对比空间
        self.contrast_criterion = InfoNCEContrastiveLoss(temperature=0.5, num_neg=5)  # 对比损失

        # 4. 体素重建损失（保持不变）
        self.recon_criterion = nn.MSELoss()

    def forward(self, spatial_x, electronic_x, scaffold_x, chem_features, masked_voxel, original_voxel, epoch=None):
        # --------------------------
        # 1. 获取编码器中间特征（3D原始特征+处理后化学特征）
        # --------------------------
        # 调用voxel_encoder的forward，传入return_feat=True，返回预训练需要的中间特征
        pooled_3d, chem_feat_norm = self.voxel_encoder(
            spatial_x, electronic_x, scaffold_x, chem_features,
            return_feat=True  # 触发编码器返回中间特征，而非下游任务的分类输出
        )  # pooled_3d: [B, 112]（3D分支全局池化后的原始特征）；chem_feat_norm: [B, 384]（注意力加权后的化学特征）

        # 特征归一化（稳定训练，避免梯度爆炸）
        voxel_feat_norm = self.voxel_encoder.ln_3d(pooled_3d)  # 复用编码器自带的3D特征归一化层
        chem_feat_norm = self.voxel_encoder.ln_chem(chem_feat_norm)  # 复用编码器自带的化学特征归一化层

        # --------------------------
        # 2. 任务1：体素重建（掩码区域内，对有效区域加权）
        # --------------------------
        recon_input = torch.cat([voxel_feat_norm, chem_feat_norm], dim=1)
        recon_voxel = self.recon_head(recon_input)  # 重建体素（需确保尺寸与original_voxel一致）

        # 第一步：确定“掩码区域”（仅在掩码区域计算损失，原有逻辑保留）
        mask_region = (1 - masked_voxel)  # 1 - masked_voxel：1表示掩码区域，0表示未掩码区域（与你原有逻辑一致）

        # 第二步：在掩码区域内，拆分“有效非零区域”和“背景零区域”
        # 有效区域：掩码区域内，原始体素非零的部分（分子实际占据的体素）
        valid_region_in_mask = mask_region * (original_voxel != 0.0).float()
        # 背景区域：掩码区域内，原始体素为零的部分（分子外部的空白体素）
        background_region_in_mask = mask_region * (original_voxel == 0.0).float()

        # 第三步：分别计算两个区域的MSE损失
        # 有效区域损失：关注分子结构重建，权重设为15（可根据有效区域占比调整）
        valid_loss = self.recon_criterion(recon_voxel * valid_region_in_mask, original_voxel * valid_region_in_mask)
        # 背景区域损失：仅需保证预测为零，权重设为1（避免模型忽视背景）
        background_loss = self.recon_criterion(recon_voxel * background_region_in_mask,
                                               original_voxel * background_region_in_mask)

        # 第四步：加权求和得到最终重建损失（平衡两个区域的优化优先级）
        # 权重可调整：若有效区域占比极低（<5%），将valid_weight提至25；若背景准确率下降，降低至10
        # 2. 动态权重逻辑：按epoch阶段调整
        if epoch is None:  # 若未传epoch，用默认固定权重
            valid_weight = 25
        elif epoch <= 5:  # 前5个epoch：高权重逼模型学有效区域
            valid_weight = 25
        elif 5 < epoch <= 10:  # 5-10 epoch：中等权重巩固
            valid_weight = 26
        elif epoch == 11:  # 当前轮次：小幅提升
            valid_weight = 30.0  # 30:0.6≈50:1，比原23提升30%
        elif 12 <= epoch <= 13:  # 12-13轮：二次过渡（30→35）
            valid_weight = 35.0
        elif 14 <= epoch <= 15:  # Epoch12及以后：达到目标比例
            valid_weight = 40.0  # 40:0.6≈67:1，你认为的足够比例
        elif epoch == 16:  # 
            valid_weight = 45.0  # 30:0.6≈50:1，比原23提升30%
        elif epoch >= 17:  # 12-13轮：二次过渡（30→35）
            valid_weight = 50.0  #

        background_weight = 0.6
        recon_loss = valid_weight * valid_loss + background_weight * background_loss
        # recon_loss = (valid_weight * valid_loss + background_weight * background_loss) / (
        #             valid_weight + background_weight)

        # --------------------------
        # 3. 任务2：跨模态对齐（让3D特征和化学特征在64维空间语义一致）
        # --------------------------
        proj_3d = self.proj_3d(voxel_feat_norm)  # 3D特征投影到对比空间
        proj_chem = self.proj_chem(chem_feat_norm)  # 化学特征投影到对比空间
        contrast_loss = self.contrast_criterion(proj_3d, proj_chem)  # InfoNCE损失优化对齐效果

        # --------------------------
        # 4. 总损失（加权求和，优先保证体素重建效果，可根据数据调整权重）
        # --------------------------
        total_loss = 0.6 * recon_loss + 0.4 * contrast_loss  # 重建权重更高，确保3D结构学习有效

        # 返回所有关键结果，用于训练监控（如打印损失、可视化重建效果）
        return {
            "total_loss": total_loss,  # 总损失（反向传播用）
            "recon_loss": recon_loss,  # 体素重建损失（监控重建效果）
            "contrast_loss": contrast_loss,  # 跨模态对齐损失（监控对齐效果）
            "recon_voxel": recon_voxel,  # 重建的体素（可视化用）
            "proj_3d": proj_3d,  # 3D投影特征（可选：分析对齐效果）
            "proj_chem": proj_chem  # 化学投影特征（可选：分析对齐效果）
        }


def pretrain_main(cache_root, subset="pretrain", batch_size=16, epochs=10,
                  lr=1e-4, weight_decay=1e-5, save_dir=str(PROJECT_ROOT / "results" / "pretrain"),val_chunk_num=2):
    # --------------------------
    # 1. 初始化设备与随机种子（固定种子确保可复现）
    # --------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.benchmark = True  # 开启benchmark加速大样本训练


    # 2. 划分训练集和验证集的缓存块
    # 2.1 获取所有缓存块路径
    all_chunk_files = glob.glob(os.path.join(cache_root, f"{subset}_chunk_*.npz"))
    all_chunk_files.sort(key=lambda x: int(os.path.basename(x).split("_")[-1].split(".")[0]))  # 按编号排序
    print(f"总缓存块数量：{len(all_chunk_files)}，计划抽取 {val_chunk_num} 个作为验证集")

    # 2.2 随机抽取50个缓存块作为验证集
    random.shuffle(all_chunk_files)  # 随机打乱
    val_chunk_files = all_chunk_files[:val_chunk_num]  # 前50个作为验证集
    train_chunk_files = all_chunk_files[val_chunk_num:]  # 剩余作为训练集
    print(f"训练集缓存块：{len(train_chunk_files)}个，验证集缓存块：{len(val_chunk_files)}个")

    # --------------------------
    # 2. 初始化数据集与数据加载器（多进程加载30w样本）
    # --------------------------
    # 预训练模式：is_pretrain=True，生成掩码
    # 3. 初始化训练集和验证集数据集
    # 训练集
    train_dataset = NPZBatchLoadedVoxelDataset(
        subset="train",
        chunk_files=train_chunk_files,  # 传入训练集缓存块
        shuffle=True,
        seed=42,
        is_pretrain=True,
        mask_ratio=0.3
    )
    # 验证集（不打乱，mask_ratio与训练一致）
    val_dataset = NPZBatchLoadedVoxelDataset(
        subset="val",
        chunk_files=val_chunk_files,  # 传入验证集缓存块
        shuffle=False,  # 验证集不打乱
        seed=42,
        is_pretrain=True,
        mask_ratio=0.3
    )


    # 4. 初始化数据加载器
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2  # 提前预加载，加速训练
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=8,  # 验证集worker可少一些
        pin_memory=True,
        persistent_workers=True
    )

    # --------------------------
    # 3. 初始化模型、优化器、梯度缩放（适配大样本训练）
    # --------------------------
    # 初始化预训练模型
    model = PretrainModelWithMultiModalAlign().to(device)

    print("从头开始预训练：随机初始化体素模型，从第1轮开始。")

    # 多GPU训练（若有多个GPU，开启DataParallel）
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        print(f"⚡ 使用 {torch.cuda.device_count()} 个GPU进行训练")
    from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
    # 从头初始化优化器，使用函数参数指定的初始学习率。
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,  # 首个周期为10轮
        T_mult=2,  # 后续周期依次为20、40轮
        eta_min=1e-5,
        last_epoch=-1,
    )

    # 梯度缩放（混合精度训练，降低显存占用，加速训练）
    scaler = GradScaler()

    # --------------------------
    # 4. 初始化ChemBERTa特征提取器（多模态对齐必需）
    # --------------------------
    chem_teacher = ChemBERTaTeacher()
    chem_teacher.model.to(device)  # 移到GPU，加速特征提取

    # --------------------------
    # 5. 创建模型保存目录
    # --------------------------
    os.makedirs(save_dir, exist_ok=True)
    best_val_loss = float("inf")  #  # 记录最佳损失，保存最优模型

    def validate():
        model.eval()
        total_val_loss = 0.0
        total_val_recon = 0.0
        total_val_contrast = 0.0
        # 新增验证集指标
        total_ss_tot_val = 0.0
        total_ss_res_val = 0.0
        total_zero_correct_val = 0
        total_zero_count_val = 0
        total_cos_sim_val = 0.0
        total_samples = 0

        with torch.no_grad():
            pbar = tqdm(val_dataloader, desc="Validating")
            for batch in pbar:
                spatial_vox, electronic_vox, scaffold_vox, original_vox, masked_vox, smiles = batch
                # 设备转移（保持不变）
                spatial_vox = spatial_vox.to(device, non_blocking=True)
                electronic_vox = electronic_vox.to(device, non_blocking=True)
                scaffold_vox = scaffold_vox.to(device, non_blocking=True)
                original_vox = original_vox.to(device, non_blocking=True)
                masked_vox = masked_vox.to(device, non_blocking=True)

                # 提取化学特征（保持不变）
                chem_feats = chem_teacher.get_features(smiles, device=device)

                # 模型前向（保持不变）
                with autocast(device_type="cuda", dtype=torch.float16):
                    outputs = model(
                        spatial_x=spatial_vox,
                        electronic_x=electronic_vox,
                        scaffold_x=scaffold_vox,
                        chem_features=chem_feats,
                        masked_voxel=masked_vox,
                        original_voxel=original_vox,
                        epoch=epoch
                    )

                # 累计损失（保持不变）
                batch_size = spatial_vox.size(0)
                total_samples += batch_size
                total_val_loss += outputs["total_loss"].item() * batch_size
                total_val_recon += outputs["recon_loss"].item() * batch_size
                total_val_contrast += outputs["contrast_loss"].item() * batch_size

                # --------------------------
                # 新增：验证集指标计算（与训练集一致）
                # --------------------------
                # 1. 有效区域R²
                mask_region = (masked_vox == 0.0)
                masked_recon = outputs["recon_voxel"] * mask_region.float()
                masked_original = original_vox * mask_region.float()
                for i in range(batch_size):
                    sample_original = masked_original[i].view(-1)
                    sample_recon = masked_recon[i].view(-1)
                    mask_region_i = (masked_vox[i] == 0.0).view(-1)

                    # 有效非零区域R²
                    non_zero_mask = (sample_original != 0) & mask_region_i
                    if torch.any(non_zero_mask):
                        valid_original = sample_original[non_zero_mask]
                        valid_recon = sample_recon[non_zero_mask]
                        mean_valid = torch.mean(valid_original)
                        ss_tot = torch.sum((valid_original - mean_valid) ** 2)
                        ss_res = torch.sum((valid_original - valid_recon) ** 2)
                        total_ss_tot_val += ss_tot.item()
                        total_ss_res_val += ss_res.item()

                    # 背景0区域准确率
                    zero_region = (sample_original == 0) & mask_region_i
                    if torch.any(zero_region):
                        zero_recon = sample_recon[zero_region]
                        zero_correct = torch.sum(zero_recon < 0.1).item()
                        zero_count = torch.sum(zero_region).item()
                        total_zero_correct_val += zero_correct
                        total_zero_count_val += zero_count

                # 2. 跨模态余弦相似度
                proj_3d_norm = F.normalize(outputs["proj_3d"], dim=1)
                proj_chem_norm = F.normalize(outputs["proj_chem"], dim=1)
                batch_cos_sim = torch.mean(torch.sum(proj_3d_norm * proj_chem_norm, dim=1)).item()
                total_cos_sim_val += batch_cos_sim * batch_size

                pbar.set_postfix({"val_loss": f"{total_val_loss / total_samples:.4f}"})

                pbar.set_postfix({
                    "val_loss": f"{total_val_loss / total_samples:.4f}",
                    "val_recon": f"{total_val_recon / total_samples:.4f}",
                    "val_contrast": f"{total_val_contrast / total_samples:.4f}"
                })

        # 计算验证集平均指标
        avg_val_loss = total_val_loss / total_samples
        avg_val_recon = total_val_recon / total_samples
        avg_val_contrast = total_val_contrast / total_samples
        # 有效区域R²
        val_valid_r2 = 1 - (total_ss_res_val / (total_ss_tot_val + 1e-8)) if total_ss_tot_val > 0 else -1.0
        # 背景0准确率
        val_zero_accuracy = total_zero_correct_val / (total_zero_count_val + 1e-8) if total_zero_count_val > 0 else 0.0
        # 跨模态余弦相似度
        val_cos_sim = total_cos_sim_val / total_samples

        # 打印验证集完整指标
        print(f"验证集总结：")
        print(f"  总损失：{avg_val_loss:.4f} | 重建损失：{avg_val_recon:.4f} | 对齐损失：{avg_val_contrast:.4f}")
        print(
            f"  有效区域R²：{val_valid_r2:.4f} | 背景0准确率：{val_zero_accuracy:.4f} | 跨模态余弦相似度：{val_cos_sim:.4f}")
        return avg_val_loss,avg_val_recon,avg_val_contrast,val_valid_r2, val_zero_accuracy, val_cos_sim # 返回所有指标

    # --------------------------
    # 6. 预训练循环（核心）
    # --------------------------
    start_epoch = 0  # 从第1轮开始训练，保存的权重编号也从1开始
    for epoch in range(start_epoch,epochs):
        # 关键：更新数据集的当前epoch（从0开始，与训练循环一致）
        train_dataset.epoch = epoch  # 训练集更新epoch
        val_dataset.epoch = epoch    # 验证集同步更新（确保评估时掩码率与训练一致）
        # 打印当前掩码率，方便监控
        current_mask = min(0.3 + epoch * 0.015, 0.6)
        print(f"Epoch {epoch+1}/{epochs}，当前掩码率：{current_mask:.2%}")
        model.train()
        total_loss = 0.0
        total_recon_loss = 0.0
        total_contrast_loss = 0.0
        total_ss_tot = 0.0  # 累计R²计算的总平方和
        total_ss_res = 0.0  # 累计R²计算的残差平方和
        total_zero_correct = 0
        total_zero_count = 0
        total_cos_sim = 0.0  # 累计余弦相似度
        total_samples = 0    # 累计样本数（用于计算平均）
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}")

        total_batches = len(train_dataloader)  # 放在这里

        for batch_idx, batch in enumerate(pbar):
            # 1. 解析batch数据（匹配NPZBatchLoadedVoxelDataset的输出）
            spatial_vox, electronic_vox, scaffold_vox, original_vox, masked_vox,smiles = batch
            # 移到设备
            spatial_vox = spatial_vox.to(device, non_blocking=True)
            electronic_vox = electronic_vox.to(device, non_blocking=True)
            scaffold_vox = scaffold_vox.to(device, non_blocking=True)
            original_vox = original_vox.to(device, non_blocking=True)
            masked_vox = masked_vox.to(device, non_blocking=True)

            # 新增：调用数据增强（仅对3分支体素增强，原始体素和掩码不变
            spatial_vox = minimal_augment(spatial_vox)
            electronic_vox = minimal_augment(electronic_vox)
            scaffold_vox = minimal_augment(scaffold_vox)

            # 2. 提取ChemBERTa特征（多模态对齐必需）
            with torch.no_grad():  # 不训练ChemBERTa，关闭梯度
                chem_feats = chem_teacher.get_features(smiles, device=device)  # [B,384]

            # 3. 混合精度训练（前向传播）
            with autocast(device_type="cuda"):  # 正确：传入 torch.device 对象
                # 模型前向：输入3分支体素+化学特征+掩码+原始体素
                outputs = model(
                    spatial_x=spatial_vox,
                    electronic_x=electronic_vox,
                    scaffold_x=scaffold_vox,
                    chem_features=chem_feats,
                    masked_voxel=masked_vox,
                    original_voxel=original_vox,
                    epoch=epoch
                )
                loss = outputs["total_loss"]  # 总损失（反向传播用）

            # 4. 反向传播与参数更新
            optimizer.zero_grad()  # 清空梯度
            scaler.scale(loss).backward()  # 梯度缩放，避免梯度下溢
            scaler.step(optimizer)  # 更新参数
            scaler.update()  # 更新缩放器

            # 关键：每个batch后更新学习率调度器，传入当前进度（epoch + 已完成batch比例）
            scheduler.step(epoch + batch_idx / total_batches)  # 放在这里！

            # 5. 累计损失（用于日志）
            batch_size = spatial_vox.size(0)
            total_samples += batch_size  # 更新总样本数
            total_loss += loss.item() * batch_size
            total_recon_loss += outputs["recon_loss"].item() * batch_size
            total_contrast_loss += outputs["contrast_loss"].item() * batch_size

            # 累计R²计算所需的平方和（当前batch，按样本逐个计算）
            mask_region = (masked_vox == 0.0)  # 掩码区域：masked_vox为0的位置
            masked_recon = outputs["recon_voxel"] * mask_region.float()  # 仅保留掩码区域的重建结果
            masked_original = original_vox * mask_region.float()  # 仅保留掩码区域的真实值
            for i in range(batch_size):
                # 1. 提取掩码区域的真实值和重建值
                sample_original = masked_original[i].view(-1)
                sample_recon = masked_recon[i].view(-1)
                mask_region = (masked_vox[i] == 0.0).view(-1)  # 掩码区域（需要重建的区域）

                # 2. 计算有效非零区域的R²
                non_zero_mask = (sample_original != 0) & mask_region
                if torch.any(non_zero_mask):
                    valid_original = sample_original[non_zero_mask]
                    valid_recon = sample_recon[non_zero_mask]

                    mean_valid = torch.mean(valid_original)
                    ss_tot = torch.sum((valid_original - mean_valid) ** 2)
                    ss_res = torch.sum((valid_original - valid_recon) ** 2)

                    total_ss_tot += ss_tot.item()
                    total_ss_res += ss_res.item()

                # 3. 计算背景0区域的准确率
                zero_region = (sample_original == 0) & mask_region
                if torch.any(zero_region):
                    zero_recon = sample_recon[zero_region]
                    zero_correct = torch.sum(zero_recon < 0.1).item()  # 归一化后0的阈值设为0.1
                    zero_count = torch.sum(zero_region).item()

                    total_zero_correct += zero_correct
                    total_zero_count += zero_count

            # 累计余弦相似度（当前batch）
            proj_3d_norm = F.normalize(outputs["proj_3d"], dim=1)
            proj_chem_norm = F.normalize(outputs["proj_chem"], dim=1)
            batch_cos_sim = torch.mean(torch.sum(proj_3d_norm * proj_chem_norm, dim=1)).item()
            total_cos_sim += batch_cos_sim * batch_size  # 按样本数加权累计

            # 6. 进度条显示实时损失
            avg_loss = total_loss / total_samples
            pbar.set_postfix({
                "total_loss": f"{avg_loss:.4f}",
                "recon_loss": f"{total_recon_loss/total_samples:.4f}",
                "contrast_loss": f"{total_contrast_loss/total_samples:.4f}"
            })

        # --------------------------
        # 7. 每个epoch结束后的操作
        # --------------------------
        # 计算epoch平均损失
        epoch_avg_loss = total_loss / total_samples
        epoch_avg_recon = total_recon_loss / total_samples
        epoch_avg_contrast = total_contrast_loss / total_samples
        # 计算全epoch的有效区域R²（仅有效区域）
        epoch_valid_r2 = 1 - (total_ss_res / (total_ss_tot + 1e-8)) if total_ss_tot > 0 else -1.0
        # 计算全epoch的背景0预测准确率（汇总所有batch的结果）
        epoch_zero_accuracy = total_zero_correct / (total_zero_count + 1e-8) if total_zero_count > 0 else 0.0
        # 和余弦相似度
        epoch_cos_sim = total_cos_sim / total_samples  # 按样本数加权平均

        print(f"\nEpoch {epoch+1} 总结：")
        print(f"  总损失：{epoch_avg_loss:.4f} | 重建损失：{epoch_avg_recon:.4f} | 对齐损失：{epoch_avg_contrast:.4f}")
        print(f"  有效区域重建R²：{epoch_valid_r2:.4f} | 背景0预测准确率：{epoch_zero_accuracy:.4f}| 跨模态余弦相似度：{epoch_cos_sim:.4f}")


        # 保存每个epoch的模型（可选，用于后续分析）
        torch.save(model.state_dict(), os.path.join(save_dir, f"pretrain_model_epoch_{epoch+1}.pth"))

        # 每个epoch结束后进行验证
        print(f"\n开始验证 Epoch {epoch+1}...")
        val_loss,avg_val_recon,avg_val_contrast,val_r2, val_zero_acc, val_cos_sim = validate()  # 接收所有验证指标

        # 保存最优模型（避免过拟合，保存损失最小的模型）
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # 保存模型权重（多GPU需处理module前缀）
            if torch.cuda.device_count() > 1:
                torch.save(model.module.state_dict(), os.path.join(save_dir, "best_pretrain_model.pth"))
            else:
                torch.save(model.state_dict(), os.path.join(save_dir, "best_pretrain_model.pth"))
            print(f"✅ 保存最优模型至：{os.path.join(save_dir, 'best_pretrain_model.pth')}")



        # 写入训练日志
        current_lr = optimizer.param_groups[0]['lr']  # 获取第一个参数组的学习率（通常所有组一致）
        # 写入日志时补充验证集指标
        with open(os.path.join(save_dir, "train_log.txt"), "a") as f:
            f.write(f"Epoch {epoch + 1}, "
                    f"TotalLoss: {epoch_avg_loss:.4f}, "
                    f"ValTotalLoss: {val_loss:.4f}, "
                    f"ReconLoss: {epoch_avg_recon:.4f}, "
                    f"ValReconLoss: {avg_val_recon:.4f}, "  # 补充验证集重建损失
                    f"ContrastLoss: {epoch_avg_contrast:.4f}, "
                    f"ValContrastLoss: {avg_val_contrast:.4f}, "  # 补充验证集对齐损失
                    f"ReconR2: {epoch_valid_r2:.4f}, "
                    f"ValReconR2: {val_r2:.4f}, "  # 补充验证集R²
                    f"ZeroAccuracy: {epoch_zero_accuracy:.4f}, "
                    f"ValZeroAccuracy: {val_zero_acc:.4f}, "  # 补充验证集背景准确率
                    f"CosSim: {epoch_cos_sim:.4f}, "
                    f"ValCosSim: {val_cos_sim:.4f}, "  # 补充验证集余弦相似度
                    f"LR: {current_lr:.6f}\n")


    # --------------------------
    # 8. 预训练结束
    # --------------------------
    print(f"\n🎉 预训练完成！")
    print(f"📦 最佳模型保存路径：{os.path.join(save_dir, 'best_pretrain_model.pth')}")
    print(f"📊 最佳训练损失：{best_val_loss:.4f}")



# --------------------------
# 调用预训练主函数（配置参数）
# --------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pretrain BVCNN from voxel caches.")
    parser.add_argument("--cache-root", default=str(PROJECT_ROOT / "data" / "cache" / "pretrain"))
    parser.add_argument("--save-dir", default=str(PROJECT_ROOT / "results" / "pretrain"))
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--val-chunks", type=int, default=40)
    args = parser.parse_args()
    pretrain_main(
        cache_root=args.cache_root,
        subset="pretrain",  # 分块文件前缀（与你的npz文件匹配，如pretrain_chunk_0.npz）
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        save_dir=args.save_dir,
        val_chunk_num=args.val_chunks,
    )
"""
3. 后续操作（可选）
查看所有 screen 会话：screen -ls
重新进入会话：screen -r pretrain（pretrain 为你的会话名）
关闭会话：在会话内输入 exit 并回车
"""
