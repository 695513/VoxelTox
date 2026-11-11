# -*- coding: utf-8 -*-
"""Seven dataset configurations and one shared training loop.
Run: python train.py --dataset bace
"""

import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CACHE_PREFIX = str(PROJECT_ROOT / "data" / "cache" / "BVCNN")
DEFAULT_SPLIT_DIR = str(PROJECT_ROOT / "splits" / "scaffold")
DEFAULT_PRETRAINED_PATH = str(PROJECT_ROOT / "weights" / "pretrain_model_epoch_23.pth")

DEFAULT_CONFIG = {
    "dataset": "bbbp",
    "optimizer_groups": {
        "backbone": {"lr": 1e-05, "weight_decay": 0.002},
        "attention": {"lr": 7e-05, "weight_decay": 0.002},
        "projection": {"lr": 0.0001, "weight_decay": 0.002},
        "fusion_conv": {"lr": 5e-05, "weight_decay": 0.002},
        "scale_attn": {"lr": 7e-05, "weight_decay": 0.0001},
        "multi_scale_fusion": {"lr": 7e-05, "weight_decay": 0.0001},
        "residual_proj": {"lr": 7e-05, "weight_decay": 0.0001},
        "task_head": {"lr": 7e-05, "weight_decay": 0.0001},
    },
    "freeze_attention_epoch": 2,
    "freeze_hold_epoch": 7,
    "freeze_full_epoch": 15,
    "head_dropout": 0.4,
    "loss_default_alpha": 0.65,
    "residual_dropout": 0.1,
    "fusion_3d_dropout": 0.3,
    "fusion_dim": 128,
    "fusion_dropout": 0.4,
    "contrast_weight": 0.02,
    "min_lr": 1e-06,
    "lr_decay_epochs": 40,
    "freeze_description_2": "交叉注意力 + 化学注意力",
    "freeze_description_3": "保持现有解冻",
}
DATASET_OVERRIDES = {
    "bbbp": {},
    "sider": {
        "dataset": "sider",
        "optimizer_groups": {
            "backbone_bottom": {"lr": 9e-05, "weight_decay": 0.0002},
            "electronic_top": {"lr": 9e-05, "weight_decay": 0.00015},
            "spatial_scaffold_top": {"lr": 9e-05, "weight_decay": 0.0001},
            "cross_attn": {"lr": 9e-05, "weight_decay": 0.0001},
            "chem_attention": {"lr": 9e-05, "weight_decay": 0.0001},
            "fusion_conv": {"lr": 9e-05, "weight_decay": 0.0001},
            "scale_attn": {"lr": 0.00011, "weight_decay": 5e-05},
            "multi_scale_fusion": {"lr": 0.0001, "weight_decay": 5e-05},
            "task_head": {"lr": 0.00011, "weight_decay": 5e-05},
            "residual_proj": {"lr": 0.00011, "weight_decay": 0},
            "light_adapt": {"lr": 0.00011, "weight_decay": 5e-05},
        },
        "freeze_attention_epoch": 3,
        "freeze_hold_epoch": 6,
        "freeze_full_epoch": 8,
        "loss_default_alpha": 0.5,
        "fusion_dim": 512,
        "fusion_dropout": 0.3,
        "contrast_weight": 0.05,
        "lr_decay_epochs": 60,
    },
    "bace": {
        "dataset": "bace",
        "optimizer_groups": {
            "backbone": {"lr": 1e-05, "weight_decay": 0.001},
            "attention": {"lr": 8e-05, "weight_decay": 0.0001},
            "projection": {"lr": 0.0001, "weight_decay": 0.0001},
            "fusion_conv": {"lr": 8e-05, "weight_decay": 0.0001},
            "scale_attn": {"lr": 0.0001, "weight_decay": 0.0001},
            "multi_scale_fusion": {"lr": 0.0001, "weight_decay": 0.0001},
            "residual_proj": {"lr": 0.0001, "weight_decay": 0.0001},
            "task_head": {"lr": 0.00012, "weight_decay": 0.0001},
        },
        "head_dropout": 0.6,
        "fusion_dropout": 0.6,
        "contrast_weight": 0.05,
        "lr_decay_epochs": 60,
    },
    "dili": {
        "dataset": "dili",
        "optimizer_groups": {
            "backbone": {"lr": 1e-05, "weight_decay": 0.001},
            "attention": {"lr": 6e-05, "weight_decay": 0.0001},
            "projection": {"lr": 0.0001, "weight_decay": 0.0001},
            "fusion_conv": {"lr": 8e-05, "weight_decay": 0.0001},
            "scale_attn": {"lr": 0.0001, "weight_decay": 0.0001},
            "multi_scale_fusion": {"lr": 0.0001, "weight_decay": 0.0001},
            "residual_proj": {"lr": 0.0001, "weight_decay": 0.0001},
            "task_head": {"lr": 8e-05, "weight_decay": 0.0001},
        },
        "head_dropout": 0.5,
        "loss_default_alpha": 0.5,
        "residual_dropout": 0.15,
        "fusion_3d_dropout": 0.35,
        "fusion_dim": 512,
        "fusion_dropout": 0.35,
        "contrast_weight": 0.05,
    },
    "ames": {
        "dataset": "ames",
        "optimizer_groups": {
            "backbone": {"lr": 1e-05, "weight_decay": 0.001},
            "attention": {"lr": 8e-05, "weight_decay": 0.0001},
            "projection": {"lr": 0.0001, "weight_decay": 0.0001},
            "fusion_conv": {"lr": 8e-05, "weight_decay": 0.0001},
            "scale_attn": {"lr": 0.0001, "weight_decay": 0.0001},
            "multi_scale_fusion": {"lr": 0.0001, "weight_decay": 0.0001},
            "residual_proj": {"lr": 0.0001, "weight_decay": 0.0001},
            "task_head": {"lr": 0.00012, "weight_decay": 0.0001},
        },
        "head_dropout": 0.55,
        "residual_dropout": 0.15,
        "fusion_dim": 512,
        "fusion_dropout": 0.5,
        "contrast_weight": 0.05,
        "lr_decay_epochs": 60,
    },
    "clintox": {
        "dataset": "clintox",
        "optimizer_groups": {
            "backbone": {"lr": 5e-06, "weight_decay": 0.001},
            "attention": {"lr": 0.0001, "weight_decay": 0.001},
            "projection": {"lr": 0.0001, "weight_decay": 0.001},
            "fusion_conv": {"lr": 8e-05, "weight_decay": 0.001},
            "scale_attn": {"lr": 0.0001, "weight_decay": 0.001},
            "multi_scale_fusion": {"lr": 0.0001, "weight_decay": 0.001},
            "residual_proj": {"lr": 0.0001, "weight_decay": 0.001},
            "task_head": {"lr": 0.00012, "weight_decay": 0.001},
        },
        "freeze_attention_epoch": 4,
        "freeze_hold_epoch": 10,
        "freeze_full_epoch": 20,
        "loss_default_alpha": 0.5,
        "residual_dropout": 0.15,
        "fusion_dropout": 0.35,
        "min_lr": 8e-07,
        "lr_decay_epochs": 70,
        "freeze_description_2": " 解冻：交叉注意力 + 化学注意力",
        "freeze_description_3": " 保持现有解冻",
    },
    "herg": {
        "dataset": "herg",
        "optimizer_groups": {
            "backbone": {"lr": 1e-05, "weight_decay": 0.001},
            "attention": {"lr": 6e-05, "weight_decay": 0.0001},
            "projection": {"lr": 0.0001, "weight_decay": 0.0001},
            "fusion_conv": {"lr": 8e-05, "weight_decay": 0.0001},
            "scale_attn": {"lr": 0.0001, "weight_decay": 0.0001},
            "multi_scale_fusion": {"lr": 0.0001, "weight_decay": 0.0001},
            "residual_proj": {"lr": 0.0001, "weight_decay": 0.0001},
            "task_head": {"lr": 8e-05, "weight_decay": 0.0001},
        },
        "head_dropout": 0.5,
        "loss_default_alpha": 0.5,
        "residual_dropout": 0.15,
        "fusion_3d_dropout": 0.35,
        "fusion_dropout": 0.35,
        "contrast_weight": 0.05,
    },
}
ENTRY_CONFIGS = {
    "bbbp": {
        "dataset_name": "bbbp",
        "cache_dir": DEFAULT_CACHE_PREFIX,
        "batch_size": 32,
        "epochs": 50,
        "lr": 0.0001,
        "task_type": "classification",
    },
    "sider": {
        "dataset_name": "sider",
        "cache_dir": DEFAULT_CACHE_PREFIX,
        "batch_size": 32,
        "epochs": 50,
        "lr": 0.0001,
        "task_type": "classification",
    },
    "bace": {
        "dataset_name": "bace",
        "cache_dir": DEFAULT_CACHE_PREFIX,
        "batch_size": 32,
        "epochs": 50,
        "lr": 0.0001,
        "task_type": "classification",
    },
    "dili": {
        "dataset_name": "dili",
        "cache_dir": DEFAULT_CACHE_PREFIX,
        "batch_size": 32,
        "epochs": 50,
        "lr": 0.0001,
        "task_type": "classification",
    },
    "ames": {
        "dataset_name": "ames",
        "cache_dir": DEFAULT_CACHE_PREFIX,
        "batch_size": 32,
        "epochs": 50,
        "lr": 0.0001,
        "task_type": "classification",
    },
    "clintox": {
        "dataset_name": "clintox",
        "cache_dir": DEFAULT_CACHE_PREFIX,
        "batch_size": 32,
        "epochs": 50,
        "lr": 0.0001,
        "task_type": "classification",
    },
    "herg": {
        "dataset_name": "herg",
        "cache_dir": DEFAULT_CACHE_PREFIX,
        "batch_size": 32,
        "epochs": 50,
        "lr": 0.0001,
        "task_type": "classification",
    },
}


def get_config(dataset):
    config = dict(DEFAULT_CONFIG)
    config.update(DATASET_OVERRIDES[dataset])
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=tuple(DATASET_OVERRIDES))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-dir", help="Cache prefix; dataset suffix is appended, e.g. voxel-bace")
    parser.add_argument("--split-dir", default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--pretrained-path", default=DEFAULT_PRETRAINED_PATH)
    parser.add_argument("--output-dir", help="Output directory; default: results/downstream/<dataset>/seed_<seed>")
    parser.add_argument("--chemberta-path", default="DeepChem/ChemBERTa-77M-MLM",
                        help="Hugging Face model name or local snapshot directory")
    args = parser.parse_args()
    import core

    config = get_config(args.dataset)
    config["chemberta_path"] = args.chemberta_path
    core.configure(config)
    globals().update({k: v for k, v in vars(core).items() if not k.startswith("_")})
    entry = dict(ENTRY_CONFIGS[args.dataset])
    entry["cache_dir"] = args.cache_dir if args.cache_dir is not None else DEFAULT_CACHE_PREFIX
    train_from_cache(**entry, seed=args.seed, split_dir=args.split_dir,
                     pretrained_path=args.pretrained_path, output_dir=args.output_dir)


def train_from_cache(
    dataset_name="bace",
    cache_dir=DEFAULT_CACHE_PREFIX,
    batch_size=16,
    epochs=35,
    lr=0.0001,
    task_type="classification",
    seed=42,
    split_dir=DEFAULT_SPLIT_DIR,
    pretrained_path=DEFAULT_PRETRAINED_PATH,
    output_dir=None,
):
    output_dir = Path(output_dir) if output_dir is not None else PROJECT_ROOT / "results" / "downstream" / dataset_name / f"seed_{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(seed)
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache_root = f"{cache_dir}-{dataset_name}"
    if not os.path.exists(cache_root):
        raise FileNotFoundError(f"缓存主目录不存在: {cache_root}")
    split_path = os.path.join(split_dir, f"{dataset_name}.pkl")
    if not os.path.exists(split_path):
        raise FileNotFoundError(f"分割文件不存在: {split_path}")
    with open(split_path, "rb") as f:
        split_indices = pickle.load(f)
    train_idx = split_indices["train_idx"]
    val_idx = split_indices["valid_idx"]
    test_idx = split_indices["test_idx"]
    train_dataset = BatchLoadedVoxelDataset(
        cache_root=cache_root, subset="train", indices=train_idx, shuffle=True, seed=seed
    )
    val_dataset = BatchLoadedVoxelDataset(
        cache_root=cache_root, subset="val", indices=val_idx, shuffle=False
    )
    test_dataset = BatchLoadedVoxelDataset(
        cache_root=cache_root, subset="test", indices=test_idx, shuffle=False
    )
    num_tasks, task_names = train_dataset.get_task_info()
    print(f"任务数: {num_tasks}, 任务名: {task_names}")
    print(f"训练集: {len(train_idx)} | 验证集: {len(val_idx)} | 测试集: {len(test_idx)}")
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=collate_fn,
        prefetch_factor=4,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=collate_fn,
        prefetch_factor=2,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=collate_fn,
        prefetch_factor=2,
    )
    model = UniversalMolecularModel(
        num_tasks=num_tasks, task_type=task_type, dataset_name=dataset_name
    ).to(device)
    if os.path.exists(pretrained_path):
        pretrained_weights = torch.load(pretrained_path, map_location=device)
        filtered_weights = {}
        for k, v in pretrained_weights.items():
            cleaned_k = k.replace("voxel_encoder.", "")
            if any(
                (
                    prefix in cleaned_k
                    for prefix in [
                        "spatial_branch",
                        "electronic_branch",
                        "scaffold_branch",
                        "cross_attn",
                        "fusion_conv",
                        "ln_3d",
                        "ln_chem",
                        "chem_attention",
                    ]
                )
            ) and (
                not any(
                    (
                        exclude in cleaned_k
                        for exclude in [
                            "recon_head",
                            "recon_criterion",
                            "voxel_recon",
                            "reconstruct_branch",
                            "mask_generator",
                            "mask_attention",
                            "proj_3d",
                            "proj_chem",
                        ]
                    )
                )
            ):
                filtered_weights[cleaned_k] = v
        model_state = model.feature_extractor.state_dict()
        missing_keys = [k for k in model_state if k not in filtered_weights]
        unexpected_keys = [k for k in filtered_weights if k not in model_state]
        print(
            f"未加载的模型键（共{len(missing_keys)}个）: {(missing_keys if len(missing_keys) < 10 else missing_keys[:10] + ['...'])}"
        )
        print(
            f"意外的权重键（共{len(unexpected_keys)}个）: {(unexpected_keys if len(unexpected_keys) < 10 else unexpected_keys[:10] + ['...'])}"
        )
        model.feature_extractor.load_state_dict(filtered_weights, strict=False)
        print(
            f"✅ 成功加载预训练权重至特征提取器：{pretrained_path}，加载了{len(filtered_weights)}个键"
            if CFG["dataset"] in ("bbbp", "sider", "bace", "dili", "clintox", "herg")
            else f" 成功加载预训练权重至特征提取器：{pretrained_path}，加载了{len(filtered_weights)}个键"
        )
    else:
        print(
            f"⚠️ 未找到预训练权重，将从头训练"
            if CFG["dataset"] in ("bbbp", "sider", "bace", "dili", "clintox", "herg")
            else f" 未找到预训练权重，将从头训练"
        )
    chem_teacher = ChemBERTaTeacher()
    optimizer = (
        optim.Adam
        if CFG["dataset"] in ("bbbp", "bace", "dili", "ames", "clintox", "herg")
        else optim.AdamW
    )(
        (
            [
                {
                    "params": [
                        p
                        for n, p in model.feature_extractor.named_parameters()
                        if any(
                            (
                                prefix in n
                                for prefix in [
                                    "spatial_branch",
                                    "electronic_branch",
                                    "scaffold_branch",
                                ]
                            )
                        )
                        and (
                            not any(
                                (
                                    ending in n
                                    for ending in [".0.weight", ".0.bias", ".1.weight", ".1.bias"]
                                )
                            )
                        )
                    ],
                    "lr": CFG["optimizer_groups"]["backbone"]["lr"],
                    "weight_decay": CFG["optimizer_groups"]["backbone"]["weight_decay"],
                },
                {
                    "params": [
                        p
                        for n, p in model.feature_extractor.named_parameters()
                        if any((prefix in n for prefix in ["cross_attn", "chem_attention"]))
                    ],
                    "lr": CFG["optimizer_groups"]["attention"]["lr"],
                    "weight_decay": CFG["optimizer_groups"]["attention"]["weight_decay"],
                },
                {
                    "params": [
                        p
                        for n, p in model.feature_extractor.named_parameters()
                        if any(
                            (
                                keyword in n
                                for keyword in [
                                    "proj_3d",
                                    "proj_chem",
                                    "ln_atomic",
                                    "ln_group",
                                    "scale_proj",
                                ]
                            )
                        )
                    ],
                    "lr": CFG["optimizer_groups"]["projection"]["lr"],
                    "weight_decay": CFG["optimizer_groups"]["projection"]["weight_decay"],
                },
                {
                    "params": model.feature_extractor.fusion_conv.parameters(),
                    "lr": CFG["optimizer_groups"]["fusion_conv"]["lr"],
                    "weight_decay": CFG["optimizer_groups"]["fusion_conv"]["weight_decay"],
                    "name": "fusion_conv",
                },
                {
                    "params": model.scale_attn.parameters(),
                    "lr": CFG["optimizer_groups"]["scale_attn"]["lr"],
                    "weight_decay": CFG["optimizer_groups"]["scale_attn"]["weight_decay"],
                },
                {
                    "params": model.multi_scale_fusion.parameters(),
                    "lr": CFG["optimizer_groups"]["multi_scale_fusion"]["lr"],
                    "weight_decay": CFG["optimizer_groups"]["multi_scale_fusion"]["weight_decay"],
                },
                {
                    "params": model.residual_proj.parameters(),
                    "lr": CFG["optimizer_groups"]["residual_proj"]["lr"],
                    "weight_decay": CFG["optimizer_groups"]["residual_proj"]["weight_decay"],
                },
                {
                    "params": model.task_head.parameters(),
                    "lr": CFG["optimizer_groups"]["task_head"]["lr"],
                    "weight_decay": CFG["optimizer_groups"]["task_head"]["weight_decay"],
                    "name": "task_head",
                },
            ]
            if CFG["dataset"] in ("bbbp", "bace", "dili", "ames", "clintox", "herg")
            else [
                {
                    "params": [
                        p
                        for n, p in model.feature_extractor.named_parameters()
                        if any(
                            (
                                prefix in n
                                for prefix in [
                                    "spatial_branch.0",
                                    "spatial_branch.1",
                                    "electronic_branch.0",
                                    "electronic_branch.1",
                                    "scaffold_branch.0",
                                    "scaffold_branch.1",
                                ]
                            )
                        )
                    ],
                    "lr": CFG["optimizer_groups"]["backbone_bottom"]["lr"],
                    "weight_decay": CFG["optimizer_groups"]["backbone_bottom"]["weight_decay"],
                    "name": "backbone_bottom",
                },
                {
                    "params": [
                        p
                        for n, p in model.feature_extractor.named_parameters()
                        if "electronic_branch" in n and any((str(layer) in n for layer in [-1, -2]))
                    ],
                    "lr": CFG["optimizer_groups"]["electronic_top"]["lr"],
                    "weight_decay": CFG["optimizer_groups"]["electronic_top"]["weight_decay"],
                    "name": "electronic_top",
                },
                {
                    "params": [
                        p
                        for n, p in model.feature_extractor.named_parameters()
                        if any((prefix in n for prefix in ["spatial_branch", "scaffold_branch"]))
                        and any((str(layer) in n for layer in [-1, -2]))
                    ],
                    "lr": CFG["optimizer_groups"]["spatial_scaffold_top"]["lr"],
                    "weight_decay": CFG["optimizer_groups"]["spatial_scaffold_top"]["weight_decay"],
                    "name": "spatial_scaffold_top",
                },
                {
                    "params": [
                        p
                        for n, p in model.feature_extractor.named_parameters()
                        if "cross_attn" in n
                    ],
                    "lr": CFG["optimizer_groups"]["cross_attn"]["lr"],
                    "weight_decay": CFG["optimizer_groups"]["cross_attn"]["weight_decay"],
                    "name": "cross_attn",
                },
                {
                    "params": [
                        p
                        for n, p in model.feature_extractor.named_parameters()
                        if "chem_attention" in n
                    ],
                    "lr": CFG["optimizer_groups"]["chem_attention"]["lr"],
                    "weight_decay": CFG["optimizer_groups"]["chem_attention"]["weight_decay"],
                    "name": "chem_attention",
                },
                {
                    "params": model.feature_extractor.fusion_conv.parameters(),
                    "lr": CFG["optimizer_groups"]["fusion_conv"]["lr"],
                    "weight_decay": CFG["optimizer_groups"]["fusion_conv"]["weight_decay"],
                    "name": "fusion_conv",
                },
                {
                    "params": model.scale_attn.parameters(),
                    "lr": CFG["optimizer_groups"]["scale_attn"]["lr"],
                    "weight_decay": CFG["optimizer_groups"]["scale_attn"]["weight_decay"],
                    "name": "scale_attn",
                },
                {
                    "params": model.multi_scale_fusion.parameters(),
                    "lr": CFG["optimizer_groups"]["multi_scale_fusion"]["lr"],
                    "weight_decay": CFG["optimizer_groups"]["multi_scale_fusion"]["weight_decay"],
                    "name": "multi_scale_fusion",
                },
                {
                    "params": model.task_head.parameters(),
                    "lr": CFG["optimizer_groups"]["task_head"]["lr"],
                    "weight_decay": CFG["optimizer_groups"]["task_head"]["weight_decay"],
                    "name": "task_head",
                },
                {
                    "params": model.residual_proj.parameters(),
                    "lr": CFG["optimizer_groups"]["residual_proj"]["lr"],
                    "weight_decay": CFG["optimizer_groups"]["residual_proj"]["weight_decay"],
                    "name": "residual_proj",
                },
                {
                    "params": [
                        p
                        for n, p in model.feature_extractor.named_parameters()
                        if any(
                            (keyword in n for keyword in ["ln_atomic", "ln_group", "scale_proj"])
                        )
                    ],
                    "lr": CFG["optimizer_groups"]["light_adapt"]["lr"],
                    "weight_decay": CFG["optimizer_groups"]["light_adapt"]["weight_decay"],
                    "name": "light_adapt",
                },
            ]
        ),
        betas=(0.9, 0.999),
    )
    if CFG["dataset"] in ("sider",):
        initial_target_lrs = {}
        default_target_lrs = {"cross_attn": 0.0001, "chem_attention": 0.0001, "fusion_conv": 0.0001}
        for param_group in optimizer.param_groups:
            pg_name = param_group.get("name", "")
            if pg_name in default_target_lrs.keys():
                initial_target_lrs[pg_name] = param_group["lr"]
        for key in default_target_lrs:
            if key not in initial_target_lrs:
                initial_target_lrs[key] = default_target_lrs[key]
    scaler = torch.amp.GradScaler("cuda")

    def adjust_learning_rate(optimizer, epoch):
        min_lr = CFG["min_lr"]
        for param_group in optimizer.param_groups:
            if "initial_lr" not in param_group:
                param_group["initial_lr"] = param_group["lr"]
            decayed_lr = param_group["initial_lr"] * (1 - epoch / CFG["lr_decay_epochs"])
            param_group["lr"] = max(decayed_lr, min_lr)
        if CFG["dataset"] in ("sider",):
            warmup_epochs = 5
            for param_group in optimizer.param_groups:
                if not param_group["params"]:
                    continue
                if "initial_lr" not in param_group:
                    param_group["initial_lr"] = param_group["lr"]
                init_lr = param_group["initial_lr"]
                if init_lr == 0:
                    continue
                if epoch < warmup_epochs:
                    pg_name = param_group.get("name", "")
                    if pg_name not in ["cross_attn", "chem_attention", "fusion_conv"]:
                        warmup_scale = 1.5
                        param_group["lr"] = init_lr * (epoch + 1) / warmup_epochs * warmup_scale
                    continue

    def set_freeze_mode(model, epoch):
        """Dataset-specific legacy behavior is preserved."""
        for param in model.parameters():
            param.requires_grad = False
        for param in model.scale_attn.parameters():
            param.requires_grad = True
        for param in model.multi_scale_fusion.parameters():
            param.requires_grad = True
        for param in model.task_head.parameters():
            param.requires_grad = True
        for param in model.feature_extractor.ln_atomic.parameters():
            param.requires_grad = True
        for param in model.feature_extractor.ln_group.parameters():
            param.requires_grad = True
        for param in model.feature_extractor.scale_proj.parameters():
            param.requires_grad = True
        for param in model.residual_proj.parameters():
            param.requires_grad = True
        if CFG["dataset"] in ("bbbp", "bace", "dili", "ames", "clintox", "herg"):
            for param in model.feature_extractor.proj_3d.parameters():
                param.requires_grad = True
        if CFG["dataset"] in ("bbbp", "bace", "dili", "ames", "clintox", "herg"):
            for param in model.feature_extractor.proj_chem.parameters():
                param.requires_grad = True
        if CFG["dataset"] in ("bbbp", "bace", "dili", "ames", "clintox", "herg"):
            for param in model.feature_extractor.fusion_conv.parameters():
                param.requires_grad = True
        branches = [
            model.feature_extractor.spatial_branch,
            model.feature_extractor.electronic_branch,
            model.feature_extractor.scaffold_branch,
        ]
        cross_attn = model.feature_extractor.cross_attn
        if CFG["dataset"] in ("sider",):
            fusion_conv = model.feature_extractor.fusion_conv
        chem_attention = model.feature_extractor.chem_attention
        if CFG["dataset"] in ("sider",):
            branch_block1_unfreeze = 3
            branch_block2_unfreeze = 6
            fusion_conv_unfreeze_epoch = 8
        if CFG["dataset"] in ("bbbp",):
            fusion_conv = model.feature_extractor.fusion_conv
        if epoch < (
            CFG["freeze_attention_epoch"]
            if CFG["dataset"] in ("bbbp", "bace", "dili", "ames", "clintox", "herg")
            else branch_block1_unfreeze
        ):
            for branch in branches:
                if CFG["dataset"] in ("sider",):
                    for param in branch[:3].parameters():
                        param.requires_grad = True
                if CFG["dataset"] in ("clintox",):
                    for param in branch[-2:].parameters():
                        param.requires_grad = True
                if CFG["dataset"] in ("bbbp", "bace", "dili", "ames", "herg"):
                    for param in branch[-CFG["freeze_attention_epoch"] :].parameters():
                        param.requires_grad = True
            print(
                " 冻结策略（Epoch 0-2）：顶层模块 + 分支所有残差块（共2层）"
                if CFG["dataset"] in ("bbbp", "bace", "dili", "ames", "herg")
                else (
                    f" 冻结策略（Epoch 0‑{branch_block1_unfreeze - 1}）：顶层模块 + 分支第一层残差块（保泛化）"
                    if CFG["dataset"] in ("sider",)
                    else " 冻结策略（Epoch 0-4）：顶层模块 + 分支残差块"
                )
            )
        else:
            if CFG["dataset"] in ("sider",):
                if epoch < branch_block2_unfreeze:
                    for branch in branches:
                        for param in branch.parameters():
                            param.requires_grad = True
                    print(
                        f" 冻结策略（Epoch {branch_block1_unfreeze}‑{branch_block2_unfreeze - 1}）：+ 分支第二层残差块（释放学习能力）"
                    )
                elif epoch < fusion_conv_unfreeze_epoch:
                    for branch in branches:
                        for param in branch.parameters():
                            param.requires_grad = True
                    print(
                        f" 冻结策略（Epoch {branch_block2_unfreeze}‑{fusion_conv_unfreeze_epoch - 1}）：仅分支全解冻，暂不解冻注意力/融合层（SIDER过拟合防护）"
                    )
                else:
                    for branch in branches:
                        for param in branch.parameters():
                            param.requires_grad = True
                    for param in cross_attn.parameters():
                        param.requires_grad = True
                    for param in chem_attention.parameters():
                        param.requires_grad = True
                    for param in fusion_conv.parameters():
                        param.requires_grad = True
                    if epoch == fusion_conv_unfreeze_epoch:
                        print(
                            f" 冻结策略（Epoch {fusion_conv_unfreeze_epoch}+）：+ 交叉注意力 + 化学注意力 + 融合卷积（8轮同步解冻）"
                        )
                    else:
                        print(
                            f" 冻结策略（Epoch {fusion_conv_unfreeze_epoch}+）：全量微调（融合层+注意力持续适配）"
                        )
            if CFG["dataset"] in ("bbbp", "bace", "dili", "ames", "clintox", "herg"):
                if CFG["freeze_attention_epoch"] <= epoch < CFG["freeze_hold_epoch"]:
                    for branch in branches:
                        if CFG["dataset"] in ("clintox",):
                            for param in branch[-2:].parameters():
                                param.requires_grad = True
                        if CFG["dataset"] in ("bbbp", "bace", "dili", "ames", "herg"):
                            for param in branch[-CFG["freeze_attention_epoch"] :].parameters():
                                param.requires_grad = True
                    for param in cross_attn.parameters():
                        param.requires_grad = True
                    for param in chem_attention.parameters():
                        param.requires_grad = True
                    print(CFG["freeze_description_2"])
                elif CFG["freeze_hold_epoch"] <= epoch < CFG["freeze_full_epoch"]:
                    for branch in branches:
                        if CFG["dataset"] in ("clintox",):
                            for param in branch[-2:].parameters():
                                param.requires_grad = True
                        if CFG["dataset"] in ("bbbp", "bace", "dili", "ames", "herg"):
                            for param in branch[-CFG["freeze_attention_epoch"] :].parameters():
                                param.requires_grad = True
                    for param in cross_attn.parameters():
                        param.requires_grad = True
                    for param in chem_attention.parameters():
                        param.requires_grad = True
                    print(CFG["freeze_description_3"])
                else:
                    for branch in branches:
                        for param in branch.parameters():
                            param.requires_grad = True
                    for param in cross_attn.parameters():
                        param.requires_grad = True
                    for param in chem_attention.parameters():
                        param.requires_grad = True
                    if CFG["dataset"] in ("bbbp",):
                        for param in fusion_conv.parameters():
                            param.requires_grad = True
                    print(" 全量微调")

    best_val_auc = -float("inf")
    best_val_epoch = None
    best_val_test_auc = None
    best_val_path = str(output_dir / f"{dataset_name}_best_val_auc.pth")

    for epoch in range(epochs):
        if train_dataset.shuffle:
            train_dataset.seed = seed + epoch
        set_freeze_mode(model, epoch)
        adjust_learning_rate(optimizer, epoch)
        print(f"当前学习率：")
        for param_group in optimizer.param_groups:
            if param_group.get("name"):
                print(f"  {param_group['name']}: {param_group['lr']:.6f}")
        model.train()
        train_loss = 0.0
        train_correct = [0] * num_tasks if task_type == "classification" else []
        train_valid_total = 0
        all_train_labels = [[] for _ in range(num_tasks)] if task_type == "classification" else []
        all_train_probs = [[] for _ in range(num_tasks)] if task_type == "classification" else []
        print_weights_flag = True
        epoch_contrast_losses = []
        total_main_loss = 0.0
        train_valid_counts = [0] * num_tasks
        total_batches = len(train_loader)
        train_pbar = tqdm(total=total_batches, desc=f"Epoch {epoch + 1}/{epochs}")
        for batch_data in train_loader:
            train_pbar.update(1)
            voxel, labels, smiles_list = batch_data
            if voxel is None:
                del batch_data
                continue
            voxel = voxel.to(device)
            labels = labels.to(device)
            batch_size = labels.size(0)
            train_valid_total += batch_size
            spatial_vox = voxel[:, :10, :, :, :]
            electronic_vox = voxel[:, 10:30, :, :, :]
            scaffold_vox = voxel[:, 30:40, :, :, :]
            if print_weights_flag:
                branch_weights = get_branch_global_weights(
                    model.feature_extractor, spatial_vox, electronic_vox, scaffold_vox
                )
                print(f"\n【Epoch {epoch + 1} 分支权重】")
                print(
                    f"空间: {branch_weights[0]} | 电子: {branch_weights[1]} | 支架: {branch_weights[2]}"
                )
                print_weights_flag = False
            device = spatial_vox.device
            angle = (torch.rand(1, device=device) * 2 - 1) * 0.052
            flip_axis = 2 if torch.rand(1) < 0.5 else 3
            spatial_vox = minimal_augment(
                spatial_vox, angle=angle, flip_axis=flip_axis, epoch=epoch
            )
            electronic_vox = minimal_augment(
                electronic_vox, angle=angle, flip_axis=flip_axis, epoch=epoch
            )
            scaffold_vox = minimal_augment(
                scaffold_vox, angle=angle, flip_axis=flip_axis, epoch=epoch
            )
            chem_features = chem_teacher.get_features(smiles_list, device)
            optimizer.zero_grad()
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                outputs, total_loss = model(
                    spatial_vox, electronic_vox, scaffold_vox, chem_features, labels=labels
                )
                main_loss = (
                    bce_with_label_smoothing(outputs, labels, smoothing=0.05)
                    if CFG["dataset"] in ("bbbp", "bace", "dili", "ames", "clintox", "herg")
                    else bce_with_label_smoothing(outputs, labels, smoothing=0.0, alpha=0.5)
                )
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += total_loss.item() * batch_size
            total_main_loss += main_loss.item() * batch_size
            contrast_loss = (total_loss - main_loss) / CFG["contrast_weight"]
            epoch_contrast_losses.append(contrast_loss.item())
            preds = (torch.sigmoid(outputs) > 0.5).float()
            for t in range(num_tasks):
                valid_mask = labels[:, t] != -1
                train_correct[t] += (preds[valid_mask, t] == labels[valid_mask, t]).sum().item()
                train_valid_counts[t] += valid_mask.sum().item()
                all_train_labels[t].extend(labels[valid_mask, t].detach().cpu().numpy())
                all_train_probs[t].extend(outputs[valid_mask, t].detach().cpu().numpy())
            if CFG["dataset"] in ("bbbp", "bace", "dili", "ames", "clintox", "herg"):
                del preds
            else:
                del voxel, labels, smiles_list, spatial_vox, electronic_vox, scaffold_vox
            if CFG["dataset"] in ("sider",):
                del chem_features, outputs, total_loss, main_loss
                del preds
            gc.collect()
            torch.cuda.empty_cache()
        train_pbar.close()
        train_loss_avg = train_loss / train_valid_total if train_valid_total else 0.0
        avg_main_loss = total_main_loss / train_valid_total if train_valid_total else 0.0
        avg_contrast_loss = np.mean(epoch_contrast_losses) if epoch_contrast_losses else 0.0
        train_accs = [
            train_correct[t] / train_valid_counts[t] if train_valid_counts[t] > 0 else 0.0
            for t in range(num_tasks)
        ]
        train_aucs = [
            (
                roc_auc_score(all_train_labels[t], all_train_probs[t])
                if len(np.unique(all_train_labels[t])) > 1
                else 0.0
            )
            for t in range(num_tasks)
        ]
        mean_train_auc = np.mean([auc for auc in train_aucs if auc > 0])
        mean_train_acc = np.mean([acc for acc in train_accs if acc > 0])
        print("")
        print(
            f"Train: 总损失={train_loss_avg:.4f}, 主损失={avg_main_loss:.4f}, 对比损失={avg_contrast_loss:.4f}"
        )
        print(f"       Mean ACC={mean_train_acc:.4f}, Mean AUC={mean_train_auc:.4f}")
        model.eval()
        val_loss = 0.0
        val_metrics = {}
        with torch.no_grad():
            if CFG["dataset"] in ("bbbp", "bace", "dili", "ames", "clintox", "herg"):
                chem_teacher.model.eval()
            all_val_labels = [[] for _ in range(num_tasks)] if task_type == "classification" else []
            all_val_preds = [[] for _ in range(num_tasks)] if task_type == "classification" else []
            val_valid_total = 0
            for batch_data in val_loader:
                voxel, labels, smiles_list = batch_data
                if voxel is None:
                    continue
                voxel = voxel.to(device)
                labels = labels.to(device)
                batch_size = labels.size(0)
                val_valid_total += batch_size
                spatial_vox = voxel[:, :10, :, :, :]
                electronic_vox = voxel[:, 10:30, :, :, :]
                scaffold_vox = voxel[:, 30:40, :, :, :]
                chem_features = chem_teacher.get_features(smiles_list, device)
                outputs = model(spatial_vox, electronic_vox, scaffold_vox, chem_features)
                loss = (
                    nn.BCEWithLogitsLoss()(outputs, labels)
                    if CFG["dataset"] in ("bbbp", "bace", "dili", "ames", "clintox", "herg")
                    else (
                        nn.BCEWithLogitsLoss()(outputs, labels)
                        if task_type == "classification"
                        else nn.HuberLoss(delta=1.0)(outputs, labels)
                    )
                )
                val_loss += loss.item() * batch_size
                for t in range(num_tasks):
                    all_val_labels[t].extend(labels[:, t].cpu().numpy())
                    all_val_preds[t].extend(outputs[:, t].cpu().numpy())
                del voxel, labels, outputs, loss
                gc.collect()
            val_loss_avg = val_loss / val_valid_total if val_valid_total else 0.0
            val_aucs = [
                (
                    roc_auc_score(all_val_labels[t], all_val_preds[t])
                    if len(np.unique(all_val_labels[t])) > 1
                    else 0.0
                )
                for t in range(num_tasks)
            ]
            mean_val_auc = np.mean([auc for auc in val_aucs if auc > 0])
            val_metrics["mean_auc"] = mean_val_auc
            print(f"Val:    Loss={val_loss_avg:.4f}, Mean AUC={mean_val_auc:.4f}")
        if CFG["dataset"] in ("bbbp", "bace", "dili", "ames", "clintox", "herg"):
            chem_teacher.model.train()
        test_loss = 0.0
        test_metrics = {}
        with torch.no_grad():
            if CFG["dataset"] in ("bbbp", "bace", "dili", "ames", "clintox", "herg"):
                chem_teacher.model.eval()
            all_test_labels = (
                [[] for _ in range(num_tasks)] if task_type == "classification" else []
            )
            all_test_preds = [[] for _ in range(num_tasks)] if task_type == "classification" else []
            test_valid_total = 0
            for batch_data in test_loader:
                voxel, labels, smiles_list = batch_data
                if voxel is None:
                    continue
                voxel = voxel.to(device)
                labels = labels.to(device)
                batch_size = labels.size(0)
                test_valid_total += batch_size
                spatial_vox = voxel[:, :10, :, :, :]
                electronic_vox = voxel[:, 10:30, :, :, :]
                scaffold_vox = voxel[:, 30:40, :, :, :]
                chem_features = chem_teacher.get_features(smiles_list, device)
                outputs = model(spatial_vox, electronic_vox, scaffold_vox, chem_features)
                loss = (
                    nn.BCEWithLogitsLoss()(outputs, labels)
                    if CFG["dataset"] in ("bbbp", "bace", "dili", "ames", "clintox", "herg")
                    else (
                        nn.BCEWithLogitsLoss()(outputs, labels)
                        if task_type == "classification"
                        else nn.HuberLoss(delta=1.0)(outputs, labels)
                    )
                )
                test_loss += loss.item() * batch_size
                for t in range(num_tasks):
                    all_test_labels[t].extend(labels[:, t].cpu().numpy())
                    all_test_preds[t].extend(outputs[:, t].cpu().numpy())
                del voxel, labels, outputs, loss
                gc.collect()
            test_loss_avg = test_loss / test_valid_total if test_valid_total else 0.0
            test_aucs = [
                (
                    roc_auc_score(all_test_labels[t], all_test_preds[t])
                    if len(np.unique(all_test_labels[t])) > 1
                    else 0.0
                )
                for t in range(num_tasks)
            ]
            mean_test_auc = np.mean([auc for auc in test_aucs if auc > 0])
            test_metrics["mean_auc"] = mean_test_auc
            print(f"Test:   Loss={test_loss_avg:.4f}, Mean AUC={mean_test_auc:.4f}")
        current_lrs = [param_group["lr"] for param_group in optimizer.param_groups]
        print(
            f"当前学习率: 底层={current_lrs[0]:.2e}, 顶层={current_lrs[1]:.2e}, 融合层={current_lrs[2]:.2e}, "
            if CFG["dataset"] in ("bbbp", "bace", "dili", "ames", "clintox", "herg")
            else f"当前学习率: 底层={current_lrs[0]:.2e}, 顶层={current_lrs[1]:.2e}, 融合层={current_lrs[5]:.2e}, "
        )
        # Select solely by validation AUC; test AUC is recorded, not used to select.
        if np.isfinite(mean_val_auc) and mean_val_auc > best_val_auc:
            best_val_auc = float(mean_val_auc)
            best_val_epoch = epoch + 1
            best_val_test_auc = float(mean_test_auc)
            # Keep a plain state_dict for compatibility with existing inference code.
            torch.save(model.state_dict(), best_val_path)
            # Store the corresponding scores separately from the model weights.
            import json
            with open(output_dir / f"{dataset_name}_best_val_auc.json", "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "dataset": dataset_name,
                        "selection_metric": "validation_auc",
                        "epoch": best_val_epoch,
                        "validation_auc": best_val_auc,
                        "test_auc": best_val_test_auc if np.isfinite(best_val_test_auc) else None,
                        "weights": best_val_path,
                    },
                    f, ensure_ascii=False, indent=2,
                )
            print(
                f"已更新验证集最佳权重: {best_val_path} | "
                f"Epoch={best_val_epoch}, Val AUC={best_val_auc:.4f}, "
                f"对应 Test AUC={best_val_test_auc:.4f}"
            )
        gc.collect()

    if best_val_epoch is not None:
        print(
            f"训练结束：验证集最佳 Epoch={best_val_epoch}, "
            f"Val AUC={best_val_auc:.4f}, 对应 Test AUC={best_val_test_auc:.4f} | "
            f"权重: {best_val_path}"
        )
    else:
        print("训练结束：没有有效的验证集 AUC，未保存最佳权重。")


if __name__ == "__main__":
    main()
