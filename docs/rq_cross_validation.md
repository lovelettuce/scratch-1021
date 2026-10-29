# Beauty/Toys × RQ tokenizer 交叉验证（四组重训）

本轮 seed=42，完整训练以下四组的 SID tokenizer 和 TIGER：

| 组合 | SID 配置来源 | GR 配置来源 |
| --- | --- | --- |
| Beauty + RQ-kmeans | 当前 `rkmeans_train_flat` 默认值 | 上次 Beauty TIGER 训练归档 |
| Beauty + RQ-VAE | 当前 `rqvae_train_flat` 默认值 | 同上 |
| Toys + RQ-VAE | 当前 `rqvae_train_flat` 默认值 | 同上 |
| Toys + RQ-kmeans | 当前 `rkmeans_train_flat` 默认值 | 同上 |

旧 Beauty + RQ-kmeans tokenizer 仅训练30step，属于验证性运行，不能继续作为本轮完整训练组合。
旧 checkpoint 和分析保留，只用于读取 GR 配置、校验原始数据及固定分组；不会填入本轮四组汇总。

## 参数与调度

SID 参数直接从当前 experiment 默认配置合成，不覆盖预算、每卡batch、worker、精度或数据划分策略；按本轮明确选择，每组GPU数覆盖为1。
当前默认值为：

| SID 方法 | 训练预算 | GPU | 每 GPU batch | 优化器 |
| --- | --- | --- | --- | --- |
| RQ-kmeans | 每层1,000 step，三层共3,000 step；无epoch上限 | 1 | 2,048 | SGD，lr=0.5 |
| RQ-VAE | 联合15,000 step；无epoch上限 | 1 | 2,048 | Adagrad，lr=0.001 |

三层码本各256，输入2048维。RQ-VAE 保留 latent64、beta0.25、BatchNorm+L2 输入归一化及其默认数据加载策略。
`save_last=true` 是为恢复额外保存状态；监控与最终step保存周期沿用默认配置。
运行清单记录实际参数和配置哈希；SID 预算以后改动时，脚本自动读取更新后的默认值。

GR 直接复制旧 Beauty 归档：4+4层、维度128、FFN1024、6头、dropout0.15、Adam lr0.001/weight_decay0.0001、
序列长120、raw batch32、增强上限512、workers8、BF16、梯度累积16、最多150epoch、验证Recall@5选优、patience10。
GR 每组独占一张卡，四层分类头维度固定256；不固定停止于旧运行的epoch67。

SID和GR均使用GPU0/1/2，最多三组并行；每个运行中的实验独占一张卡，第四组排队。
默认分配为Beauty RQ-kmeans→GPU0、Beauty RQ-VAE→GPU1、Toys RQ-VAE→GPU2、Toys RQ-kmeans→GPU0（排队）。
GPU0上的前一组完成后立即启动排队组，不等待GPU1/2的任务全部完成；单组失败也会释放卡并继续其他组。
四组SID全部结束后进入GR阶段，GR使用同样的三卡队列；GPU3排除在本实验资源池外。
可显式指定其他空闲GPU池；单独选择一个实验时仅需指定一张卡。

相对SID默认八卡配置，本轮只将每组GPU数改为1：每卡batch保持2,048，全局batch由16,384变为2,048。
训练步数仍为RQ-kmeans共3,000、RQ-VAE共15,000；梯度累积沿用默认值，不自动改变学习率或进行batch补偿。
该资源差异写入配置、训练记录和报告，不能将本轮结果称为八卡全局batch的完全复现。
GPU 暂时不可用时只进行脚本验证，恢复运行由用户另行发起。

## 运行

在项目根目录执行：

```bash
# 仅检查配置、四组计划和资源分配；不训练、不创建 study。
./scripts/run_rq_cross_validation.sh --dry-run

# SID和GR均最多三组并行，第四组排队；仅使用GPU0/1/2。
./scripts/run_rq_cross_validation.sh --sid-gpus 0,1,2 --gpus 0,1,2

# 单独训练新增 Beauty + RQ-kmeans 的 SID 和导出 SID。
./scripts/run_rq_cross_validation.sh --experiment beauty_rqkmeans --stage tokenizer --sid-gpus 0
```

脚本使用 `/home/yangyujie/data/miniconda3/envs/grid/bin/python`。
统一配置为 `configs/analysis/rq_cross_validation.yaml`；相对路径从项目根解析。
默认四组端口29540/29541/29542/29543，各组命令、GPU、端口、日志、阶段状态和SHA256独立保存。
GPU预检记录已有计算进程、剩余显存，并实际分配和同步张量。用户已明确允许共享GPU0/1/2，当前allow_gpu_sharing=true，因此外部进程存在时继续执行，但仍检查至少12GiB可用显存和CUDA能力；设为false可恢复独占检查。共享状态和已有进程写入manifest，共享时的耗时不能作为独占性能比较。--gpus与--sid-gpus均默认读取统一配置，值为0,1,2；任务数量多于卡数时进入队列。

执行顺序：原始统计/固定分组 → Toys共享向量 → 四组SID按三卡队列训练/严格导出 → 四组GR和分析按三卡队列运行 → 四组汇总。
Toys向量只生成一次；Beauty复用原始2048维float32向量。
新Beauty RQ-kmeans必须使用新导出的SID重新训练GR、重新生成测试编码和两轮分析，不复用旧SID推理产物。

## 恢复与分阶段执行

本次实验矩阵和SID参数已变化，必须新建study；原暂停目录继续保留，不能用新配置对其 `--resume`。
新study的恢复命令为：

```bash
./scripts/run_rq_cross_validation.sh --resume STUDY_DIR --sid-gpus 0,1,2 --gpus 0,1,2
./scripts/run_rq_cross_validation.sh --resume STUDY_DIR --stage tokenizer --sid-gpus 0,1,2
./scripts/run_rq_cross_validation.sh --resume STUDY_DIR --stage recommender --gpus 0,1,2
./scripts/run_rq_cross_validation.sh --resume STUDY_DIR --stage analysis --gpus 0,1,2
./scripts/run_rq_cross_validation.sh --resume STUDY_DIR --experiment beauty_rqkmeans --stage analysis --gpus 0
```

`prepare` 完成原始统计和共享向量；`tokenizer` 包含SID导出。
已提交阶段校验通过后跳过；未完成训练从明确的 `checkpoints/last.ckpt` 恢复。
数据、源码、配置或输入指纹变化时拒绝恢复，不混合结果。
恢复保留优化器、epoch、早停、当前量化层及初始化缓冲；IterableDataset的epoch中间顺序不保证逐位复现。

离线汇总：

```bash
./scripts/run_rq_cross_validation.sh --stage summary --resume STUDY_DIR
```

报告只读取保存产物，不加载模型、不需要GPU或旧参考目录。
四组全部来自本轮训练；若某组未完成，明确报告缺失，不用历史checkpoint补齐。
单组报告重建仍使用 `python -m src.analysis.long_tail report` 和 `python -m src.analysis.second_layer report`。

## 分析与交付

各组先执行16样本验证，再进行全量长尾评估和第二层归因分析。
Beauty有效测试22,363条，Toys有效测试19,412条；每次运行仍从数据扫描校验，不能硬套Beauty数量。
每数据集共享固定原始频次分组，各 tokenizer 分别构建SID编码、前缀支持和真实前缀概率缓存。
推理float32、batch8、beam10/20/50、@5/@10、用户聚类bootstrap2,000次、seed42；沿用原合法性、并列排名和无效槽位口径。

四组报告检查条件/无条件生存率、局部与跨父竞争、分支支持、两个频次基线CE/MRR收益及未见路径。
MRR收益为模型减基线，CE收益为基线减模型，均保存用户聚类95%区间。
扩大beam使用共同第一层存活样本，跨tokenizer还报告双方共同进入第二层的配对结果。
不同数据集不配对，不凭不同token空间的绝对CE判断 tokenizer 优劣。
报告区分重复、部分重复与证据不足；两数据集、两方法、单seed不能推断所有RQ方法普遍成立。

```text
outputs/rq_cross_validation/seed42/<new-study-id>/
  study_manifest.json config.yaml progress/ logs/
  reference/historical_beauty_rqkmeans.json  # 仅历史配置和分组校验
  shared/{beauty,toys}/
  experiments/{beauty_rqkmeans,beauty_rqvae,toys_rqvae,toys_rqkmeans}/
    tokenizer_train/ sid_export/ recommender_train/
    analysis/{smoke_long_tail,long_tail,smoke_second_layer,second_layer}/
    progress/ status.json pipeline.log launch_<stage>.json
  comparison/inputs/<cell>/
  comparison/tables/ figures/ report.md
```

代码和产物分开，outputs、logs、data保持Git忽略。完整命令和恢复校验沿用本轮原子提交机制。

```bash
conda run --no-capture-output -n grid python -m pytest -q tests/experiments tests/analysis tests/test_export_sid_softmax.py
```
