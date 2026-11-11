# 数据

- `raw/`：原有下游 CSV。主流程支持 BBBP、BACE、SIDER、ClinTox、AMES、hERG 和 DILI；其他 CSV 作为已有材料保留。
- `../splits/scaffold/`：原有固定划分，使用 `train_idx`、`valid_idx`、`test_idx`，对应原始 CSV 的行顺序。
- `cache/BVCNN-<任务名>/`：收集的下游体素缓存，每个任务包含 `train/`、`val/`、`test/`。
- `pretrain/pretrain_30w_no_ions.sdf`：预训练体素化的默认输入。
- `pretrain/source/`：原有下载分片、筛选分片、合并文件和下载列表。保留其中原有内容，包括最终 SDF 的另一份副本。
- `cache/pretrain/`：预训练缓存的生成位置；整理时没有提供该缓存，需要运行预训练体素化脚本生成。

数据来源与历史记录保留在 `../docs/pretrain_data_sources_and_results.txt`。当前目录不能单凭文件名证明所有原始数据筛选步骤已完整复现；最终 SDF 可以作为预训练复现起点。

发布时需要提供最终 SDF、所选缓存的下载链接和校验和。大文件当前被 `.gitignore` 排除；原始分片和中间数据可作为补充归档提供。
