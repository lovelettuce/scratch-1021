# Beauty/Toys × RQ tokenizer 交叉验证（四组重训）

本轮 seed=42，完整训练以下四组的 SID tokenizer 和 TIGER：

| 组合 | SID 配置来源 | GR 配置来源 |
| --- | --- | --- |
| Beauty + RQ-kmeans | 当前 `rkmeans_train_flat` 默认值 | 项目内 `rq_recommender.yaml` |
| Beauty + RQ-VAE | 当前 `rqvae_train_flat` 默认值 | 同上 |
| Toys + RQ-VAE | 当前 `rqvae_train_flat` 默认值 | 同上 |
| Toys + RQ-kmeans | 当前 `rkmeans_train_flat` 默认值 | 同上 |

旧 Beauty + RQ-kmeans tokenizer 仅训练30step，属于验证性运行，不能继续作为本轮完整训练组合。
旧 checkpoint 和分析可保留，但本轮不读取、不校验、不纳入汇总。Beauty/Toys 分组从当前训练数据独立计算并冻结。

## 参数与调度

SID 参数直接从当前 experiment 默认配置合成，不覆盖预算、每卡batch、worker、精度或数据划分策略；按本轮明确选择，每组GPU数覆盖为1。
当前默认值为：

| SID 方法 | 训练预算 | GPU | 每 GPU batch | 优化器 |
| --- | --- | --- | --- | --- |
| RQ-kmeans | 每层1,000 step，三层共3,000 step；无epoch上限 | 1 | 2,048 | SGD，lr=0.5 |
| RQ-VAE | 联合5,000 step；每10个epoch验证；无epoch上限 | 1 | 2,048 | Adagrad，lr=0.001 |

三层码本各256，默认XL输入2048维，base输入768维。RQ-VAE 保留 latent64、beta0.25、BatchNorm+L2 输入归一化及其默认数据加载策略。
`save_last=true` 是为恢复额外保存状态；监控与最终step保存周期沿用默认配置。
运行清单记录实际参数和配置哈希；SID 预算以后改动时，脚本自动读取更新后的默认值。

GR 使用 `configs/analysis/rq_recommender.yaml` 独立配置：4+4层、维度128、FFN1024、6头、dropout0.15、Adam lr0.001/weight_decay0.0001、
序列长120、raw batch32、增强上限512、workers8、BF16、梯度累积16、最多150epoch、验证Recall@5选优、patience10。
GR 每组独占一张卡，四层分类头维度固定256；不固定停止于旧运行的epoch67。

SID和GR均使用GPU4/5/6/7，四组并行，每组使用一张卡。
默认分配为Beauty RQ-kmeans→GPU4、Beauty RQ-VAE→GPU5、Toys RQ-VAE→GPU6、Toys RQ-kmeans→GPU7。
四组SID全部结束后进入GR阶段，GR和分析沿用相同分配；GPU0/1/2/3不在默认资源池内。
可显式指定其他空闲GPU池；单独选择一个实验时仅需指定一张卡。

相对SID默认八卡配置，本轮只将每组GPU数改为1：每卡batch保持2,048，全局batch由16,384变为2,048。
训练步数仍为RQ-kmeans共3,000、RQ-VAE共5,000；梯度累积沿用默认值，不自动改变学习率或进行batch补偿。
该资源差异写入配置、训练记录和报告，不能将本轮结果称为八卡全局batch的完全复现。
GPU 暂时不可用时只进行脚本验证，恢复运行由用户另行发起。

## 运行

在项目根目录执行：

```bash
# 仅检查配置、四组计划和资源分配；不训练、不创建 study。
./scripts/run_rq_cross_validation.sh --dry-run

# SID和GR均四组并行；使用GPU4/5/6/7。
./scripts/run_rq_cross_validation.sh --sid-gpus 4,5,6,7 --gpus 4,5,6,7

# 单独训练新增 Beauty + RQ-kmeans 的 SID 和导出 SID。
./scripts/run_rq_cross_validation.sh --experiment beauty_rqkmeans --stage tokenizer --sid-gpus 4
```

脚本使用 `/home/lettuce/miniconda3/envs/grid/bin/python`。
统一配置为 `configs/analysis/rq_cross_validation.yaml`；相对路径从项目根解析。
默认四组端口29540/29541/29542/29543，各组命令、GPU、端口、日志、阶段状态和SHA256独立保存。
GPU预检记录已有计算进程、剩余显存，并实际分配和同步张量。当前allow_gpu_sharing=true，因此外部进程存在时继续执行，但仍检查至少12GiB可用显存和CUDA能力；设为false可恢复独占检查。共享状态和已有进程写入manifest，共享时的耗时不能作为独占性能比较。--gpus与--sid-gpus均默认读取统一配置，值为4,5,6,7；任务数量多于卡数时进入队列。

执行顺序：原始统计/固定分组 → Toys共享向量 → 四组SID四卡并行训练/严格导出 → 四组GR和分析四卡并行运行 → 四组汇总。
默认XL模式下，Toys向量只生成一次；Beauty复用原始2048维float32向量。base模式的两份共享向量见下方模型选择说明。
新Beauty RQ-kmeans必须使用新导出的SID重新训练GR、重新生成测试编码和两轮分析，不复用旧SID推理产物。

## 恢复与分阶段执行

本次实验矩阵和SID参数已变化，必须新建study；原暂停目录继续保留，不能用新配置对其 `--resume`。
新study的恢复命令为：

```bash
./scripts/run_rq_cross_validation.sh --resume STUDY_DIR --sid-gpus 4,5,6,7 --gpus 4,5,6,7
./scripts/run_rq_cross_validation.sh --resume STUDY_DIR --stage tokenizer --sid-gpus 4,5,6,7
./scripts/run_rq_cross_validation.sh --resume STUDY_DIR --stage recommender --gpus 4,5,6,7
./scripts/run_rq_cross_validation.sh --resume STUDY_DIR --stage analysis --gpus 4,5,6,7
./scripts/run_rq_cross_validation.sh --resume STUDY_DIR --experiment beauty_rqkmeans --stage analysis --gpus 4
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
  recommender_template.yaml              # 本轮GR模板，无历史依赖
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

## 嵌入模型选择与离线缓存

默认 `flan-t5-xl` 保留上面的旧快照、2048维和 Beauty 向量复用方式。
新增 `flan-t5-base` 对应 **google/flan-t5-base**，输出768维，Beauty/Toys分别生成一次新向量，
同数据集的两种tokenizer共享向量。两种选择都使用 T5EncoderModel 和均值聚合，GR架构不变。

```bash
./scripts/run_rq_cross_validation.sh --dry-run
./scripts/run_rq_cross_validation.sh --embedding-model flan-t5-xl --dry-run
./scripts/run_rq_cross_validation.sh --embedding-model flan-t5-base --dry-run
huggingface-cli download google/flan-t5-base
./scripts/run_rq_cross_validation.sh --embedding-model flan-t5-base
```

`embedding_models` 表记录仓库、来源、维度和可复用向量。当前base的 `source: cache`
使用项目中已下载的模型；也可将source设为 `google/flan-t5-base`，从Hugging Face本地缓存解析。
单独下载到项目目录可用 `huggingface-cli download google/flan-t5-base --local-dir cache`。
交叉验证不联网下载，缺少配置、分词器或权重时在创建study前失败。
dry-run只检查本地文件，展示缓存状态与待生成向量的数据集，不读取权重、不创建study。

普通嵌入生成继续支持Hydra参数，不传模型时仍为XL：

```bash
python -m src.inference experiment=sem_embeds_inference_flat data_dir=data/amazon_data/beauty embedding_model=google/flan-t5-base
# 原始T5-base同样可用于普通嵌入生成，但不属于本交叉验证的两个注册选项。
python -m src.inference experiment=sem_embeds_inference_flat data_dir=data/amazon_data/beauty embedding_model=google-t5/t5-base
```

不同嵌入模型必须新建study；恢复时传入原模型选项，模型、维度、资产及向量指纹变化都会拒绝复用。
报告记录嵌入模型及维度。旧study的配置与源码指纹已变化，不能直接恢复为新study。
交叉验证不再要求 `reference`、旧GR归档、checkpoint、SID或历史分析manifest。
GR基于 `tiger_train_flat` 与项目内 `rq_recommender.yaml` 合成；修改任一配置均会改变study指纹。
Beauty/Toys均从当前训练交互重新计算分组。XL的旧Beauty向量复用方式保持不变；当前从头运行请选择base。


## 本机环境与预检

`grid` 使用项目 `requirements.txt` 指定的 `huggingface-hub==0.33.2`，
无需 `/tmp` 中的测试依赖。项目默认的 `transformers==4.47.0` 与Hub 2.x不兼容。
GPU必须对运行进程可见；隔离沙箱中NVML失败不代表宿主机驱动故障。
训练预检同时使用NVML检查空闲显存，并通过PyTorch分配、同步小张量。

下面命令只解析计划，不训练：

```bash
./scripts/run_rq_cross_validation.sh --embedding-model flan-t5-base --dry-run
```

正式运行会先扫描数据、冻结分组并生成两份共享向量，再启动GPU4/5/6/7上的四组SID训练。
四组SID完成后进入GR和分析阶段。显存预检通过不保证所有实际训练批次均不会OOM。
默认XL仍保留原有路径，当前机器未配置XL快照和旧Beauty向量时请显式选择base。

下载命令使用grid兼容的 `huggingface-cli download`。独立 `hf` 2.3.0 CLI要求Hub 2.3.0，不能与本训练环境混装；如需新版CLI，请在另一个环境使用。

RQ-VAE当前采用单GPU探索预算：总计5,000步，每10个epoch验证一次。论文预算为15,000步。
修改配置只影响后续启动；已运行进程仍使用启动时的归档配置，直接用新配置恢复旧study会触发指纹不一致校验。
