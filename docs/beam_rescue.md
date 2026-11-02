# 第二层救回目标的离线追踪

在项目根目录、现有 grid 环境中运行：

```bash
python -m src.analysis.beam_rescue \
  --study-dir outputs/rq_cross_validation/seed42/20261029-211957-196412
```

入口默认要求本轮四组完整缓存：Beauty 每组22,363条，Toys 每组19,412条。
仅访问 `comparison/inputs` 中冻结的 manifest 和 CSV，无需 GPU、模型权重、原始数据或候选 PT。
复制该 inputs 目录至另一个 study 目录后仍可运行，原 manifest 中的历史模型路径仅作元数据。

默认输出 `STUDY/comparison/beam_rescue/`。目录非空时拒绝覆盖；重建使用新的 `--output-dir`。
可用 `--bootstrap-replicates` 和 `--seed` 调整统计参数，正式结果采用2,000次和42。
失败运行写入 failed manifest，不会将部分结果标记为完成。

主比较10→50，辅助比较10→20、20→50。分别分析四个模型与 Overall/Head/Mid/Tail/Train-unseen。
共同首层存活要求比较双方均通过第一层；另计算三 beam 共同首层队列。
第二层救回与反向丢失均独立统计，不假设大 beam 包含小 beam。

输出包括逐样本配对 CSV.gz、五份统计表、Overall/Tail 的救回去向和收益损失 PNG/PDF、中文报告和带输入/源码哈希的 manifest。
`destinations` 中 queue 分母为救回/反向丢失队列，all_samples 分母为全部组内样本；
反向丢失的最终去向读取小 beam。`hit_changes` 中 path_change=all 是总体命中转移，其余行是新增/丢失命中的互斥细分。
`metrics` 保存队列指标均值及全部样本的配对差值，差值方向均为大 beam 减小 beam。

主比例与指标差值采用用户聚类 bootstrap 95%区间；条件分母每次重算，零分母抽样被排除并记录有效次数。
实现对相同用户贡献向量合并计数，再按其用户数作多项抽样，与逐用户有放回抽样等价。
空队列的条件比例为无样本，不记为0；状态转移表仅作描述性统计。
比例使用0～1，指标差值使用原指标单位；报告中的 pp 为百分点。

校验包括缓存哈希、唯一键和身份对齐、样本覆盖、生存单调性、首次丢失层、命中与生存一致性、
原汇总指标复算、命中增减守恒，以及完成后输入哈希未变化。
保留原先的并列排名与非法候选槽位口径，不推算精确排名，不作 tokenizer 或校准的因果归因。

```bash
python -m pytest -q tests/analysis/test_beam_rescue.py
```
