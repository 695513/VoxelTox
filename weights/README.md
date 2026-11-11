# 权重

预训练初始化权重：`pretrain_model_epoch_23.pth`。

| 下游任务 | 已有文件 |
|---|---|
| AMES | `downstream/ames_0.8276.pth` |
| BACE | `downstream/bace_0.8440.pth` |
| BBBP | `downstream/bbbp_7585.pth` |
| ClinTox | `downstream/clintox_9854.pth` |
| DILI | `downstream/dili_0.9179.pth` |
| hERG | `downstream/herg_8286.pth` |
| SIDER | `downstream/sider_6567.pth` |

这些文件沿用收集时的名称，七份下游权重均已通过对应模型结构的严格加载检查。文件名中的数值不是本次重新评估得到的结果。

新实验权重默认写到 `results/`。正式发布时需补充下载链接、文件校验和、实际训练配置、随机种子及检查点选择依据；不要仅根据文件名推断这些信息。
