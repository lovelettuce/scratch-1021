# 全目录精确排序验证

固定四组已训练模型，以全部合法物品的原始全词表条件概率之积为评分目标。
实际使用 float32 log-softmax 的逐层和，不作合法孩子归一化，不过滤历史物品或训练未见物品。
完整目录枚举是对当前模型评分的精确检索，不是推荐命中率的理论上限。

```bash
PY=/home/lettuce/miniconda3/envs/grid/bin/python
STUDY=outputs/rq_cross_validation/seed42/20261029-211957-196412
$PY -m src.analysis.exact_catalog run --study-dir "$STUDY" --gpus 4,5,6,7
```

本次四组全量使用物理GPU4/5/6/7并行，一组一卡。先在第一张卡依次完成四组前16条历史的整个目录评分，
全部通过后再启动四个独立评分进程。每个进程拥有独立manifest、锁、输入快照和提交目录，完成后统一发布索引。
不传 `--gpus` 时仍支持单卡顺序执行；可用 `CUDA_VISIBLE_DEVICES=4 --device cuda:0` 绑定物理GPU4。
默认历史 batch8、目录 chunk128；显存不足时需创建使用较小 `--candidate-chunk-size` 的新运行。
`--max-samples 16` 创建独立的小规模结果，不能作为全量结论。

也可分阶段运行，prepare 不执行模型推理：

```bash
$PY -m src.analysis.exact_catalog prepare --study-dir "$STUDY"
$PY -m src.analysis.exact_catalog evaluate --run-dir RUN --device cuda:0 --preflight-only
$PY -m src.analysis.exact_catalog evaluate --run-dir RUN --gpus 4,5,6,7 --resume
$PY -m src.analysis.exact_catalog report --run-dir RUN
```

`--cell beauty_rqkmeans|beauty_rqvae|toys_rqvae|toys_rqkmeans` 可限制 prepare/run 范围。
prepare/run 的 `--run-dir` 必须是新目录。evaluate 的 `--resume` 校验来源、源码、快照和提交分片，
发生错误时保留已提交分片。修改输入、评分源码、chunk或样本范围须创建新运行，不得将结果拼接到旧运行。
缺少 CUDA 时明确失败，不自动切换 CPU；显式 `--device cpu` 仅适合小型调试。

输出位于 study 的 `comparison/exact_catalog/runs/<run-id>/`：

- `manifest.json`：来源与源码指纹、运行参数、CUDA实测、各组预检及完成状态。
- `inputs/<cell>/catalog.npy`、`catalog.csv`：按 item_id 对应的完整SID目录；分组与原始beam指标快照也在此目录。
- `preflight/<cell>/`：独立的前16条全目录评分诊断。
- `cells/<cell>/<shard>.npy`：float32 总 log 概率矩阵，每行对应提交标记中的 sample_key，每列对应 item_id。
  四组全量总计约4.02GB，只计算四层之和，不保存完整分类头概率矩阵。
- 同名 `.paired.csv.gz`：每样本每beam一行，四轨指标、目录排名、并列范围、目标逐层log概率、
  与排除目标后的第K名竞争者的分数差、精确Top-K覆盖、原实验队列与数值诊断。
- 同名 `.top50.csv.gz`、`.beam.csv.gz`：精确Top-50与全部原beam候选的独立重评分。
- `progress/scores_<cell>/`：原子提交标记、样本键及校验和，每16个历史batch提交一个分片。
- `workers/<cell>/`：四卡模式下的独立工作目录；父目录提交标记直接引用这些评分文件，不重复复制矩阵。
- `logs/<cell>_gpu<N>.log`：四卡模式的每组日志；manifest记录PID、GPU分配、退出码和实际耗时。
- `report/`：离线统计CSV、PNG/PDF、中文报告和报告产物清单。

四轨分别为原始 `torch.topk`（original）、原候选统一重评分（rescored）、
移除非法候选不补齐（legal）、全目录合法物品（exact）。后三者按分数降序、完整SID字典序处理并列。
独立重评分与目录列的误差须≤1e-4，合法候选的规范评分统一引用目录列，避免分块浮点差异改变比较口径。
目标距截止点≤1e-4的样本单列敏感性比例；它们不应作为确定的机制证据。

逐样本验证 exact−original = (exact−legal)+(legal−rescored)+(rescored−original)。
主比较 exact−legal 同时统计搜索遗漏与原beam的偶然命中，二者可能抵消。
合法目录中的精确Top-K目标若存在于beam候选中，必须也进入该beam的合法Top-K。

`report` 只需要本次运行目录，可复制后在无GPU、无checkpoint、无原始数据的环境重建。
主要比例和配对差值采用用户聚类bootstrap 2,000次、seed42、95%区间；空组／空队列为空值。
连续NLL、排名和分数误差另作描述统计；不以NLL均值代替推荐排序收益。
