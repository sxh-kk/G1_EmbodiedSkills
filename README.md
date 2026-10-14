# G1 EmbodiedSkills

将 EmbodiedSkills 的观察、五候选、后果预测、选择和执行流程接入 Unitree G1。复用已有 Isaac Lab 与冻结 SONIC，先做右侧两指夹爪的桌面方块抓放。

当前阶段见 [LOG.md](LOG.md)，各阶段的实测结论放在 [docs](docs/)。规划策略作为首个候选后端；预训练 VLA 适配留待后续。

## 本地配置

仓库只保存新增源码、配置、报告与小体积指标；上游代码、模型、生成的机器人资产和实验数据通过本地路径引用。

```bash
export SONIC_ROOT=/path/to/SONIC-runtime
export EMBODIEDSKILLS_ROOT=/path/to/EmbodiedSkills
export ISAAC_PYTHON=/path/to/sonic_isaaclab/bin/isaac-python
export PYTHONPATH="$PWD/src:$SONIC_ROOT:$EMBODIEDSKILLS_ROOT/src:${PYTHONPATH:-}"
```

依赖现有 SONIC / Isaac Lab 环境，不自动更新依赖。仿真需使用其 Isaac 启动包装器，离线准备可使用同一环境的普通 Python。

## 当前可运行内容

五阶段迁移完成。新采集 200 锚点/1000 条物理分支，完成真实三个后果模型和比较头训练，并在相同 30 个新种子上运行完整闭环：直接执行 27/30、随机选择 21/30、EmbodiedSkills 选择器 29/30。全部 90 回合及 10,800 次选择通过独立物理、编码和执行记录审计。

本轮是单任务仿真工程验收，不作统计优越性或新方法贡献结论。离线测试选择器 28/30、直接 29/30；离线分支选择与在线反复选择应分别解释。完整结果、失败分析、延迟和资源见 [阶段 5 报告](docs/05-在线闭环与配对评测结论.md)。

![真实选择器抓放过程](docs/figures/05-selector-seed900000.jpg)

完整采集、训练及在线入口见 [复现指南](docs/复现指南.md)。阶段报告逐项区分工程检查、离线分支结果和在线任务成功率。

## 执行与学习路径

```mermaid
flowchart LR
  A[头部 RGB-D 与本体反馈] --> B[视觉定位与参考纠偏]
  B --> C[五个原生参考候选]
  A --> D[冻结视觉与语言编码]
  C --> E[三个后果模型]
  D --> E
  E --> F[原 EmbodiedSkills 比较头]
  F --> G[选中一个候选]
  G --> H[冻结 SONIC 身体控制与独立夹爪]
  H --> I[物理仿真推进 0.2 秒]
  I --> A
```

`simulation.py` 管理浮动 G1、接触、相机与重建；`candidates.py` 和 `native_reference.py` 保证候选编码与实际 SONIC 预览对应；`collection.py` 采集共同起点的五分支；`learning.py` 调用原三成员及比较头训练；`online.py` 记录每次真实预测、选择及执行。离线未来图像与成功标签只参与训练和评测。

```bash
python scripts/build_asset.py --output outputs/assets
python scripts/prepare_reference.py --output outputs/references --duration 20
python scripts/prepare_task.py --output outputs/task-pilot-v1
bash "$ISAAC_PYTHON" scripts/run_sim.py \
  --reference outputs/task-pilot-v1/native \
  --asset outputs/assets/g1_parallel_gripper.urdf \
  --output outputs/direct-demo --mode direct --seconds 24 \
  --cameras --servo --record-video
```

## 来源

- [EmbodiedSkills](https://github.com/ZJU4EmbodiedAI/EmbodiedSkills)：后果模型、比较头、训练与运行接口；当前参考版本 `e3aa582614a9601f1c826950b9673cab81da2cb0`。
- [SONIC](https://github.com/NVlabs/GR00T-WholeBodyControl)：冻结的 G1 身体跟踪控制器。
- [Isaac Lab](https://github.com/isaac-sim/IsaacLab)：物理场景与传感器。
- [CCFA-Skills](https://github.com/mikubaka88/CCFA-Skills)：阶段规划及实验记录规范。

上游代码、资产和权重遵循各自许可。本仓库不重新分发上游模型与机器人网格。
