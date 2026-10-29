# Beauty 第二层淘汰归因实验

使用现有 conda 环境 `grid`，从项目根目录执行：

```bash
conda run --no-capture-output -n grid python -m src.analysis.second_layer run \
  --config configs/analysis/beauty_second_layer.yaml
```

若系统默认 conda 来自其他安装，请使用 `/home/yangyujie/data/miniconda3/bin/conda`，或直接使用该安装下 `envs/grid/bin/python`。

## 阶段、验证和恢复

```bash
# 校验第一次全量实验、统计训练支持、计算真实前缀频次基线；不使用 GPU。
conda run --no-capture-output -n grid python -m src.analysis.second_layer prepare \
  --config configs/analysis/beauty_second_layer.yaml

# 16 条样本的真实 checkpoint 端到端验证。
conda run --no-capture-output -n grid python -m src.analysis.second_layer run \
  --config configs/analysis/beauty_second_layer.yaml --max-samples 16

# 显式恢复；小规模运行须保留相同的 --max-samples。
conda run --no-capture-output -n grid python -m src.analysis.second_layer run \
  --config configs/analysis/beauty_second_layer.yaml --resume outputs/analysis/beauty_second_layer/runs/RUN_ID

# 不加载 checkpoint 或 GPU，只用本次保存的产物重新生成报告。
conda run --no-capture-output -n grid python -m src.analysis.second_layer report \
  --run-dir outputs/analysis/beauty_second_layer/runs/RUN_ID
```

`evaluate` 只完成重放；`run` 在重放后调用 `report`。运行会复用 `prepare` 已生成的支持与基线缓存。新运行使用上海时间戳目录；指定已存在的 run_id 会失败，不静默覆盖。

恢复须保持配置、第一次实验 manifest、数据/模型指纹和分析源码哈希不变。每个输出分片有原子提交标记与 SHA256；未提交分片重算，已提交分片校验后跳过。推理 batch size 为 8，每 16 个 batch 合并一个输出分片。

## 数据与来源

默认来源是第一次完成的 `20261028-174405-087020` 全量实验，包含 22,363 条测试记录。复用冻结的原始 item 分组、teacher-forcing float32 概率、测试输入、每层生存状态与最终候选。不修改第一次产物。当前重放要求与来源相同的 beam 10/20/50、batch size、seed 和设备，避免批次改变造成不易归因的数值差异。

重放全部四层，以核对每层目标生存状态、最终候选及得分。候选差异只有在数值容差内属于截止并列时才允许并逐行记录；目标路径生存变化始终视为失败。得分容差为 atol=1e-7、rtol=1e-6。

## 第二层归因

主诊断数据每行对应一条测试样本和一个 beam，键为 `sample_key=源文件名:原始记录序号`。

- `entered_second`：第一层目标父节点仍在 beam；`retained_second`：第二层目标前缀仍在 beam。
- `parent_probability`：正确父节点累计概率；`parent_beam_position`：父节点在实际保留 beam 中的位置。
- `target_conditional_probability`：目标第二位在正确父节点下的原始概率。
- `target_cumulative_probability`：原搜索中 float32 乘法形成的两位前缀概率。
- `cutoff_probability`：第二层全部 B×V 扩展的第 B 名概率。
- `log_score_margin`：原生目标累计概率与截止概率的对数差；两者均正时才定义。
- `target_log_score`：正的父概率与条件概率分别取 log 后相加，可帮助辨识乘法下溢。
- `local_rank_best/worst`、`legal_rank_best/worst`、`global_rank_best/worst`：严格更高／至少同高候选数形成的并列排名区间。
- `same_parent_*higher`、`other_parent_*higher`：同父与其他父节点中严格更高的候选数，进一步区分合法／非法。

互斥类别 `cause`：

1. `first_layer_lost`：第一层已丢失；第二层竞争字段为空。
2. `retained`：第二层实际保留。
3. `local_capacity`：至少 B 个同父候选严格更高，单一父节点内部就足以挤掉目标。
4. `cross_parent`：全局至少 B 个候选严格更高，但同父不足 B 个，需要跨父竞争才能解释淘汰。
5. `cutoff_tie`：严格更高不足 B 个，但目标仍因截止点并列选择被排除。

这些是搜索机制分类，不是对 tokenizer、第一层预测或训练采样的因果归因。无前缀约束时，局部容量与全局竞争均包含非法 token；合法集合排名另存。

`numeric_state` 可包括 parent_zero、conditional_zero、product_underflow、cutoff_zero；不存在有限对数差时留空，不加入 epsilon 改写实际剪枝。所有搜索判断基于实际参与 `topk` 的概率，而不是 teacher-forcing 近似分数。

## 训练支持与基线

所有前缀统计只扫描原始训练记录：目录物品数、交互次数、独立交互用户数、潜在目标窗口数、分支占父节点的交互比例、孩子交互量 HHI 与熵。独立用户按前缀去重，不能累加孩子的用户数。

潜在窗口数在原始序列上直接累加：位置 t（从 1 开始）作为末位目标可对应 t−1 个连续窗口，总数为 n(n−1)/2。第一位物品没有目标窗口。该量不包含原训练随机抽样、批次组合或各轮实际曝光，不能称为历史实际标签次数。

两个基线分别使用原始交互支持和潜在窗口支持，在正确父节点合法孩子中加一平滑。零支持父节点退化为均匀分布，不在测试集调参。TIGER 概率在同一合法集合内归一化；CE 使用 log 概率避免下溢；排名取平均并列名次。`raw_ce_gain` / `windows_ce_gain` 等于基线 CE 减模型 CE，正值表示模型更好。

`first_unsupported_raw_level`、`first_unsupported_window_level` 表示目标路径首次零支持的前缀层；0 表示全路径有支持。它们与 `first_lost_level` 的交叉表仅描述支持缺失和搜索丢失的对应关系。

## 报表与统计口径

输出位于 `outputs/analysis/beauty_second_layer/`：

- `datasets/<fingerprint>/prefix_support.csv`：可复用的完整前缀支持。
- `datasets/<fingerprint>/teacher/<source-selection-fingerprint>/`：真实前缀基线与准备阶段的提交标记。
- `runs/<run-id>/diagnostics/beam_*/`：逐样本诊断 CSV.gz，包含本次报告所需支持、基线和最终源推荐结果。
- `runs/<run-id>/traces/beam_*/`：父前缀/得分、第二层保留前缀/得分及 sample_keys 的 PT 分片。不保存完整条件概率矩阵。
- `tables/attribution_summary.csv`：全部组内样本、各自第一层存活样本和三个 beam 共同存活样本三个队列口径。
- `tables/baseline_comparison.csv`：上述队列的 CE、MRR、Top-K 和 CE 收益。
- `tables/paired_common_cohort.csv`：共同进入第二层队列上，相对 beam 10 的存活率和淘汰机制差值。
- `tables/beam_*_by_*.csv`：训练支持、潜在窗口、分支数量及其联合分桶。
- `tables/competition_scores.csv`、`rank_margin_retention.csv`：分数竞争和局部／全局排名描述。
- `tables/unseen_support_vs_loss.csv`：Train-Unseen 零支持层与实际丢失层交叉表。
- `tables/summary.json`、`figures/*.png/pdf` 和 `report.md`：结构化汇总、图表与中文解读。

全部样本口径的 `retained_second` 均值为 S2；进入第二层口径的均值为 S2/S1。跨 beam 的进入样本集合不同，必须同时看共同进入队列。

主要比例、CE/CE 收益和配对差值使用按用户聚类的 2,000 次 bootstrap，seed=42，95% percentile 区间。细分桶、MRR/Top-K 和交叉表为描述性点估计，不提供未经验证的因果结论；空组为空值，不记零。

复制整个输出根目录的 datasets 和 runs 后，可以在新位置离线生成报告，不需要原始数据、第一次实验目录、checkpoint 或 GPU。结果记录源码版本、脏状态、源码哈希、输入指纹和环境版本；旧运行产物保留。

## 测试

```bash
conda run --no-capture-output -n grid python -m pytest -q tests/analysis tests/test_export_sid_softmax.py
```

先通过测试与 16 条真实样本验证，再运行全量。验收须满足：样本集合一致、所有进入第二层样本有完整诊断、归因互斥穷尽、源生存状态复现、离线指标 CSV/JSON 重算一致。
