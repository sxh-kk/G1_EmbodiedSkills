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

阶段 1 完成 6 次站立、腕部运动与开合验收；阶段 2 固定场景抓放 10/10 通过。现有入口可重新生成夹爪、轨迹并运行视觉反馈抓放。五候选采集与学习选择器仍在实施，详见各阶段报告。

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
