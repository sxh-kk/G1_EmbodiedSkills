# M2 工程诊断汇总工具

将 `analysis/summarize_m2_engineering.py` 与 `tests/test_m2_engineering_summary.py` 放入项目的对应目录。工具只用 CPU、NumPy 和项目 `m2_analysis.py`，没有仿真、控制器或 GPU 调用；`analysis/` 位于物理冻结的 `src/`、`scripts/` 之外。

从项目根目录运行，所有输入均显式指定，不扫描结果目录：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 python analysis/summarize_m2_engineering.py \
  --pipelines outputs/m2/development-10env-body006-v2/pipeline.json \
  --output outputs/m2/development-10env-body006-v2/engineering-summary.json
```

示例仅适用于该 pipeline **全部完成后**。工具先拒绝运行中的 pipeline，再检查逐分片物理审计、官方 CPU FK、manifest/audit 哈希、全部实际审计证据及运行期 `source_changes=[]`。已完成实验之后的源码变化单独保留，不改写旧物理审计。`--project` 默认由工具所在位置推导，也可显式指定；`--reference-public` 可逐项对照既有公开 JSON。输出文件必须不存在。

腰部响应先扣除每个分支自己的 anchor 腰角，再扣对应默认格；+.5 秒的方向门为 `sign(b)*D_b > max(3*R3中心化配对噪声, 1e-4 rad)`。噪声只在相同冻结代码、物理输入、协议、family 和 anchor 内匹配，缺失不补零。报告保留实际幅度、复测参照的幅度一致性、四场景和五锚点覆盖、分析分支与全物理槽位成本。

**逐组 `engineering_blockers` 是保守诊断，不能当作要求所有网格和方向全通过的正式放行门。** 正式放行使用 `docs/M2-工程验收与正式采集协议.md` 中事先固定的规则，包括固定实例 `929000/low/480` 的双向响应门。物理审计通过也不等同工程通过或具备交互模型证据。

本工具永不自动写 `engineering_passed=true`，只会写 `false` 或 `null`；不会生成正式采集 acceptance 文件。该字段和逐组 blocker 不能替代固定正式规则的独立判定。

运行四项纯 CPU 契约测试：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 python -m unittest discover \
  -s tests -p test_m2_engineering_summary.py -v
```

测试覆盖：读取分片前拒绝未完成流水线、拒绝缺失/非空源变化检查、发现证据修改、正确扣除各分支锚点并计算直接 R3 噪声。测试数据为临时生成的合成数值，不是机器人实验结果。独立暂存工具尚未合入项目时，可用 `G1_PROJECT_ROOT` 指向包含 `src/` 的项目根；正式集成后无需此变量。
