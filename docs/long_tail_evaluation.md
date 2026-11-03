# Beauty 长尾评估与 SID 诊断

使用 conda 环境 `grid`，从项目根目录执行。默认配置为 `configs/analysis/beauty_long_tail.yaml`，路径相对于项目根目录解析。

```bash
conda run --no-capture-output -n grid python -m src.analysis.long_tail run \
  --config configs/analysis/beauty_long_tail.yaml
```

如果系统默认 conda 来自其他安装，使用 `/home/lettuce/miniconda3/bin/conda` 替换命令中的 `conda`。也可直接使用环境中的 Python：
`/home/lettuce/miniconda3/envs/grid/bin/python`。

## 分阶段、验证和恢复

```bash
# 只统计原始训练频次并冻结物品分组。
conda run --no-capture-output -n grid python -m src.analysis.long_tail prepare \
  --config configs/analysis/beauty_long_tail.yaml

# 16 个目标的小规模端到端验证；仍扫描测试记录以完整记录覆盖情况。
conda run --no-capture-output -n grid python -m src.analysis.long_tail run \
  --config configs/analysis/beauty_long_tail.yaml --max-samples 16

# 使用原配置恢复一个中断运行。小规模恢复须保留同样的 --max-samples。
conda run --no-capture-output -n grid python -m src.analysis.long_tail run \
  --config configs/analysis/beauty_long_tail.yaml --resume outputs/analysis/beauty_long_tail/runs/RUN_ID

# 仅从已有结果重新汇总、计算置信区间和生成报告，不需要模型或 GPU。
conda run --no-capture-output -n grid python -m src.analysis.long_tail report \
  --run-dir outputs/analysis/beauty_long_tail/runs/RUN_ID
```

输出路径会打印到终端。默认每次新建上海时间戳目录；指定 `run_id` 后如目录已存在则拒绝覆盖。已有运行仅允许显式恢复，配置及输入 SHA256 必须一致。推理 batch size 默认 8；输出分片固定合并 16 个推理 batch（最后一片允许不足），不增加推理显存。每个完成分片有原子提交标记与产物校验和，崩溃留下的未提交文件将在恢复时重算；输入记录按源文件与记录序号确定顺序。

## 评估口径

- 训练流行度：仅原始 `training/sequence_data` 的 item 交互次数，同一用户重复交互重复计数，同时保存交互用户数。不是增强训练 batch 中的标签出现次数。
- 全部 SID 列作为连续原始物品 ID 目录。合法 ID 0 保留，padding/mask 在原始 ID 校验阶段不允许作为交互混入。SID 必须非负且完整编码唯一。
- `f=0` 为 Train-unseen；非零频次物品按数目划分 Head/Mid/Tail=20%/30%/50%。阈值为频次降序第 ceil(0.2N)、ceil(0.5N) 位，同频全部归热门组，允许实际比例偏移及空组。
- 测试每条原始记录的最后一个物品为目标，其余是历史；复用原项目的 SID 映射、截断与 masking/collation。短于两个物品的记录被过滤并记录位置；无效 ID 是错误，不静默跳过。
- 生成保持无约束默认行为。非法候选仍占据推荐位置；使用原 evaluator 的 `torch.topk` 规则分别取 K，目标未生成记零。测试目标决定分组，候选目录不按组过滤。
- 样本平均按请求计权；宏平均先对每个测试目标物品平均，再对出现过的物品等权。主指标样本平均和配对差值给出用户聚类 bootstrap 95% percentile 区间；宏平均输出点估计。空组 CSV 留空、JSON 为 null。
- 真实前缀诊断使用未施加约束的 float32 log_softmax；`raw_nll = legality_nll + conditional_nll`。合法目标并列排名采用平均名次，因此边界 Top-K 诊断不等同于随机打破并列。
- 概率和 log 概率一起保存，极小概率可能下溢至 0，NLL 分析应使用 log 概率。
- 覆盖率：Tail 唯一被推荐物品数 / 全部目录 Tail 物品数。尾部占比：Tail 合法推荐槽位 / 全部合法推荐槽位。非法比例：非法槽位 / 全部推荐槽位。

## 产物复用

`datasets/<fingerprint>/items.csv` 是独立于 SID 编号的原始物品分组；指纹基于训练文件、目录 ID 和分组协议。相同训练数据与目录的不同 tokenizer 自动共用分组。

`runs/<run-id>/` 保存配置、manifest、日志、一次解析后可复用的 `test_inputs/` CPU 输入缓存、每 16 个推理 batch 合并的 teacher-forcing PT 分片、各 beam 的候选 PT 与逐样本 CSV.gz、逐层诊断 CSV.gz、汇总表、PNG/PDF 图和中文报告。`sample_key=source_file:source_record_index` 是跨阶段关联键。离线表由已提交分片重建，报告前会验证校验和与样本对齐。复制整个输出根目录（同时包含 datasets 和 runs）后，也可在新位置离线生成报告。

候选 PT 包含 `generated_sids`、`scores`、`candidate_item_ids`、`candidate_valid`、`target_sids` 和 `sample_keys`。非法候选的 item ID 为 -1，有效性以独立布尔字段表示。teacher-forcing PT 包含 metadata、target_sids、probabilities、log_probabilities 和 sample_keys，概率形状 `[batch, levels, vocabulary]`。

`tables/recommendation_metrics.csv` 是主推荐表；`paired_beam_comparison.csv` 是相对 beam 10 的配对差；`diagnostics_by_*.csv`、`beam_survival.csv`、`first_lost_level.csv` 用于定位 SID 与搜索瓶颈。首次丢失层 0 表示最终未丢失。`tree_structure.csv` 为前缀均匀统计基础，诊断表按测试样本计权。

更换 checkpoint、SID、batch size 或 beam 列表时修改配置并新建运行，保留旧结果。当前 reader 明确支持项目现有四步预处理；若训练预处理改变，须扩展 reader，不能静默绕过。归档绝对路径会由显式 data_dir、semantic_id_path 覆盖，不依赖原 Hydra 工作目录。

## 验证

```bash
conda run --no-capture-output -n grid python -m pytest -q tests/analysis tests/test_export_sid_softmax.py
```

运行耗时包含数据读取、观察回调、诊断与磁盘读写，不是纯模型延迟基准。仅使用一个 checkpoint，bootstrap 不反映重新训练的不确定性。生成算法保留原有概率相乘及浮点行为，未增加约束解码对照。图的标签使用英文，报告正文使用中文，避免缺少中文字体影响图导出。
