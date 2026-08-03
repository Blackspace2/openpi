# Mobile ALOHA 数据微调 π0.5 与离线评估设计

## 1. 目标

本项目首先打通一条可复现的 Mobile ALOHA 微调链路：

1. 将原始 Mobile ALOHA HDF5 episode 转换为 OpenPI 可读取的 LeRobot 数据集。
2. 使用 `qpos[14]` 作为 state，使用双臂动作与底盘动作拼接得到的 16 维 action。
3. 从官方 `pi05_base` checkpoint 微调一个由自然语言 prompt 区分任务的单一 VLA 模型。
4. 将全部 episode 全局随机划分为 70% 训练集和 30% 验证集。
5. 在没有真机和仿真的条件下，对验证 episode 做 teacher-forced 离线推理，生成预测动作与真值动作的指标、图表和视频。

首阶段只使用 Mobile ALOHA 数据。固定式 ALOHA 数据参与共训练作为后续独立对比实验，不进入首阶段实现和验收范围。

## 2. 已确认的数据语义

现有样本路径为：

```text
dataset/mobile_aloha/aloha_mobile_cabinet/episode_0.hdf5
```

样本和本地 Mobile ALOHA、ACT++ 源码共同确认了以下结构：

| 字段 | 形状 | 本设计中的用途 |
| --- | --- | --- |
| `/observations/qpos` | `[T, 14]` | `observation.state` |
| `/observations/qvel` | `[T, 14]` | 保留在原始数据中，本阶段不输入模型 |
| `/action` | `[T, 14]` | 左臂 6 + 左夹爪 1 + 右臂 6 + 右夹爪 1 |
| `/base_action` | `[T, 2]` | 底盘线速度 + 角速度 |
| 三路相机 | `[T, JPEG bytes]` | `cam_high`、`cam_left_wrist`、`cam_right_wrist` |

物理动作定义为：

```text
action16 = concat(
    left_arm_joint_target[6],
    left_gripper_target[1],
    right_arm_joint_target[6],
    right_gripper_target[1],
    base_linear_velocity[1],
    base_angular_velocity[1],
)
```

因此：

```text
state_dim = 14
physical_action_dim = 16
model_action_dim = 32
action_horizon = 50
```

32 维是 π0.5 网络接口宽度，不是机器人实际动作维度。数据先在真实的 14D state、16D action 空间完成机器人变换和归一化，再右侧补零到 32 维。输出时必须完成反归一化和反变换，最后只返回前 16 个物理动作维度。

## 3. Prompt 与多任务语义

六个任务共同训练一个 π0.5 VLA，不增加任务分类器、任务专用 action head、数值任务 ID embedding 或每任务 checkpoint。

每个 episode 在转换时写入自然语言任务描述。LeRobot 内部可以用 `task_index` 存储任务表索引，但 OpenPI 数据管线必须通过 `PromptFromLeRobotTask` 将其还原为自然语言 prompt，再交给 π0.5 tokenizer。模型实际接收的是图像、state 和 prompt。

任务目录到 prompt 的映射由一个独立 JSON 文件提供，例如：

```json
{
  "aloha_mobile_cabinet": "open the cabinet and put away the pot"
}
```

转换器不把六个任务名称硬编码进 Python。发现没有映射的任务目录时直接报错，避免使用目录名或空字符串作为意外 prompt。

## 4. 数据转换架构

新增 Mobile ALOHA 专用转换器，不修改原始 HDF5，也不把现有通用 ALOHA 转换器的 `is_mobile=True` 当作完整的移动底盘支持。

转换器负责：

1. 扫描 `raw_root/*/episode_*.hdf5`。
2. 从映射 JSON 取得每个任务的自然语言 prompt。
3. 解码三路 JPEG 相机。
4. 写入 `observation.state = qpos[14]`。
5. 写入 `action = concat(action[14], base_action[2])`。
6. 保留源任务目录、源文件名和原始帧号等追踪信息。
7. 建立 episode 级全局 70/30 划分 manifest。
8. 输出数据审计摘要和每个 prompt 在 train/val 中的 episode 数量。

建议的 LeRobot 字段为：

```text
observation.images.cam_high
observation.images.cam_left_wrist
observation.images.cam_right_wrist
observation.state        float32[14]
action                   float32[16]
task                     natural-language prompt
```

转换过程默认保留原始动作值：

- 不裁剪夹爪到 `[0, 1]`；样本中少量超界值继续作为监督标签。
- 不对 `/base_action` 做隐式平滑。
- 不自动偏移 action 时间戳。
- 使用 `observation[t] -> action[t:t+50]` 作为训练时间关系。

ACT++ 的底盘 5 帧平滑和 `start_ts - 1` 逻辑可作为后续显式消融实验，但不应悄悄进入首个基线。

## 5. Episode 划分与防止数据泄漏

划分单位必须是完整 episode，不能按帧随机拆分。

全部任务的 episode 合并后使用固定随机种子 `42` 全局打乱，再按 70%/30% 分为 train 和 val，不做任务分层。划分结果写入版本化 manifest，至少包括：

```text
converted_episode_id
source_task_directory
source_episode_file
prompt
split
```

约束如下：

- 同一个 episode 不能同时出现在 train 和 val。
- normalization statistics 只能读取 train episode。
- val episode 不参与梯度更新、数据增强统计或 checkpoint 选择之外的训练决策。
- 输出各 prompt 的 train/val 数量用于检查，但不根据数量重新平衡划分。
- 如果全局随机结果导致某个 prompt 在 val 中为零，产生明确警告，但不自动改成分层划分。

首阶段不再额外保留 test split。这里的 val 同时承担离线模型比较用途，因此最终指标只能解释为当前验证集上的离线模仿误差，不能解释为真机任务成功率。

## 6. Mobile ALOHA 专用 transforms

现有 `AlohaInputs`、`AlohaOutputs` 固定假设 14D action，且输出会截断为前 14 维，因此需要独立的 Mobile ALOHA policy transform。

输入变换：

1. 对 state 的前 14 维应用标准 ALOHA 到 π 内部空间的关节符号与夹爪变换。
2. 对 action 的前 14 维应用对应的 ALOHA 变换。
3. action 的最后两维底盘速度保持原有单位和符号，不使用手臂变换。
4. 将三路相机映射到 π0.5 的 base、left wrist 和 right wrist 图像输入。
5. 保留自然语言 prompt。

delta action mask 明确定义为：

```text
[True] * 6 + [False] + [True] * 6 + [False] + [False] * 2
```

即只有 12 个机械臂关节使用相对当前 `qpos` 的 delta；两个夹爪和两个底盘速度维度保持绝对值。

完整训练输入顺序为：

```text
raw LeRobot sample
-> repack
-> Mobile ALOHA robot transform
-> arm-joint delta transform
-> train-only quantile normalization
-> image resize / prompt tokenize
-> state and action right-pad to 32D
-> π0.5
```

推理输出必须沿逆向语义恢复：

```text
π0.5 action[50, 32]
-> unnormalize the first 16 dims while leaving structural padding unchanged
-> delta-to-absolute on the 16D semantic prefix using current transformed state
-> inverse Mobile ALOHA transform on the first 14 dims and preserve base velocity dims
-> select the first 16 physical dims as action[50, 16]
```

实现时应沿用 OpenPI transform group 的实际正反向调用顺序，而不是在评估脚本中手写另一套近似反变换。

## 7. π0.5 训练配置

新增独立训练配置，例如 `pi05_mobile_aloha`，核心配置为：

```text
Pi0Config(
    pi05=True,
    action_dim=32,
    action_horizon=50,
)
```

并满足：

- 初始化权重来自官方 `pi05_base/params`。
- `prompt_from_task=True`。
- 数据加载器只选择 manifest 中的 train episode。
- 使用 Mobile ALOHA 专用 data config 和 transforms。
- 计算并保存当前 train split 的 14D state、16D transformed action quantile statistics。
- checkpoint 中复制本次数据的 normalization assets，推理和评估不得借用普通 ALOHA 或其他机器人统计量。

首阶段保持 π0.5 的 32D 模型宽度以兼容 base checkpoint；不把模型结构改成原生 16D，也不把缺失的 16 个 padding 维度解释为机器人动作。

## 8. 离线验证与可视化

本阶段没有真机或仿真，因此采用 teacher-forced offline evaluation：每个时间点使用数据集中的真实图像、真实 `qpos` 和 prompt 作为输入，比较模型输出和记录动作。

这种评估能回答“模型在真实观测条件下与示范动作相差多少”，不能回答“模型闭环执行时是否成功”。误差不能替代真机成功率。

### 8.1 单步展示

在验证 episode 的每个时间点重新预测一个 50 步 chunk：

```text
prediction[t] = predicted_chunk_from_observation_t[0]
target[t] = recorded_action[t]
```

这组序列用于主视频，因为时间含义最直接。

### 8.2 完整 action chunk 指标

对每个预测起点 `t` 和 horizon `h`：

```text
prediction[t, h] <-> target[t + h]
```

episode 尾部不存在的未来帧必须 mask，不能把 padding 当成真实零动作。建议分别报告 `h=0`、短期、中期和长期 horizon bucket 的误差，以观察动作预测随未来距离的退化。

### 8.3 指标

所有指标都在反归一化、反 delta、反 ALOHA 变换后的真实 16D 动作空间计算：

- 每维 MAE 和 RMSE。
- 12D 双臂关节 group MAE/RMSE。
- 2D 夹爪 group MAE/RMSE。
- 2D 底盘 group MAE/RMSE。
- 底盘运动方向准确率。
- 真值底盘静止时的 false-start rate。
- 不同 action horizon 区间的误差。

32D padding 部分永远不进入指标。

### 8.4 可视化产物

每个验证 episode 输出：

```text
episode_<id>_prediction.mp4
episode_<id>_actions.png
episode_<id>_metrics.json
```

视频布局：

- 左侧：`cam_high` 主画面和两个腕部相机缩略图，同时显示 prompt、episode、帧号和时间。
- 右侧：16 个动作通道的滚动曲线，黑色为 ground truth，红色为 prediction，并使用竖线表示当前时刻。

推理使用固定随机种子并记录 checkpoint step、数据 manifest 哈希和 normalization asset 标识，使不同 checkpoint 的离线结果可复现比较。

## 9. 错误处理与数据审计

转换器默认 fail fast，不静默跳过坏 episode：

- 缺少 `qpos`、`action`、`base_action` 或必要相机时停止，并报告文件名。
- 检查所有时间序列长度是否一致。
- 检查 `qpos`、`action`、`base_action` 是否存在 `NaN` 或 `Inf`。
- JPEG 解码失败时报告 episode、相机和帧号。
- 检查 state/action 维度严格为 14/14/2。
- 检查各 task directory 是否存在 prompt 映射。

转换完成后生成审计摘要：

- episode 和总帧数。
- 每任务 episode 数。
- state/action 各通道的 min、max、mean、std 和选定 quantiles。
- 图像尺寸、通道和解码失败数。
- train/val 数量及交集检查结果。

动作范围异常只告警和统计，不在转换器中自动裁剪。

## 10. 测试与首阶段验收

实现采用小型、可独立验证的组件，并至少覆盖：

1. **转换测试**：用合成 HDF5 验证 `qpos[14]`、`action[14] + base_action[2]`、三路 JPEG 和 prompt 写入正确。
2. **split 测试**：同一输入和 seed 产生同一 manifest，比例正确且 train/val 无交集。
3. **normalization 防泄漏测试**：统计计算只访问 train episode。
4. **transform 测试**：前 14 维应用 ALOHA 变换，底盘两维保持原语义；delta mask 只覆盖 12 个机械臂关节。
5. **round-trip 测试**：16D action 经变换、归一化、32D padding 和完整反变换后恢复到容差范围内。
6. **时间对齐测试**：视频使用 chunk 第一步，完整 horizon 指标使用 `t+h`，尾部 padding 被 mask。
7. **评估指标测试**：构造可计算的预测/真值，核对每维和分组指标。
8. **真实样本 smoke test**：只读使用 `episode_0.hdf5` 跑通审计、转换和可视化所需的数据读取路径。

首阶段完成条件：

- 完整数据能转换并产生稳定 manifest。
- train-only normalization statistics 可生成。
- `pi05_base` 权重可加载到保持 32D 宽度的 Mobile ALOHA 配置。
- GPU 服务器可以启动训练并保存 checkpoint。
- 验证 episode 可以生成解码后 16D 预测与真值的指标、静态图和视频。

Apple Silicon Mac 只承担数据转换、审计和轻量测试。完整 π0.5 训练及 checkpoint 推理以 GPU 服务器结果为准。

## 11. 第二阶段：固定式 ALOHA 共训练对比实验

固定式 ALOHA 可以加入同一个 VLA 训练。其兼容表示为：

```text
state = qpos[14]
action = concat(static_action[14], [0.0, 0.0])
```

这不是臆测：Mobile ALOHA 论文的共训练目标明确把固定式 ALOHA 的底盘 action 补为 `[0, 0]`。论文还采用静态数据与目标 Mobile ALOHA 数据等概率采样、忽略静态数据的额外前置相机以统一为三路相机，并仅使用 Mobile ALOHA 数据的 action statistics 做归一化。论文报告该方案在七个移动操作任务上整体改善 ACT 表现，但该结果来自 ACT、Diffusion Policy 和 VINN，不代表对 π0.5 已经得到验证。论文公式中的策略是针对单个移动任务 `m` 的 `π_m`，而本项目是由 prompt 调节的六任务单一 VLA，因此第二阶段属于受论文启发的迁移实验，不是完全复现论文训练设置。

参考：

- [Mobile ALOHA 论文（PMLR）](https://proceedings.mlr.press/v270/fu25b.html)
- [Mobile ALOHA 论文 PDF](https://mobile-aloha.github.io/resources/mobile-aloha.pdf)

为了避免首阶段变量过多，第二阶段按受控实验实施：

| 实验 | 训练数据 | 验证数据 | normalization |
| --- | --- | --- | --- |
| A：基线 | Mobile ALOHA train | 固定的 Mobile ALOHA val | Mobile train stats |
| B：共训练 | Mobile ALOHA train + Static ALOHA train | 同一份 Mobile ALOHA val | 复用 A 的 Mobile train stats |

实验 B 的数据采样先复现论文的来源比例：每个训练样本以 50% 概率来自 Mobile、50% 概率来自 Static，使 batch 在期望上保持 1:1，而不是直接拼接后按总帧数均匀抽样。静态任务仍使用各自的自然语言 prompt，不增加任务分类器。相机统一使用 `cam_high` 和两个 wrist cameras；如果静态数据含 `cam_low` 或其他额外相机，则不送入模型。

对比时固定以下变量：

- Mobile train/val manifest。
- Mobile 数据、prompt 和 transforms。
- `pi05_base` 初始化权重。
- 优化器、训练 step、batch size、随机种子和评估脚本。
- 同一组 Mobile 验证 episode。

主要观察 Mobile val 上的 16D 解码动作误差，特别是 12D 手臂误差是否下降，同时检查补零静态样本是否导致底盘预测更保守、false-start rate 降低或移动动作召回变差。由于没有真机，第二阶段最终也只能得出离线模仿误差层面的结论。

第二阶段需要新增多数据源采样能力和 Static ALOHA adapter，因此不会预先塞进首阶段转换器或训练配置。

## 12. 实施边界

首阶段明确不包括：

- 固定式 ALOHA 共训练。
- 任务分层划分。
- `qvel` 输入实验。
- action 时间偏移和底盘平滑消融。
- 真机控制、安全限幅和 emergency stop。
- 仿真环境与闭环任务成功率。
- SLAM、全局路径规划和动态避障。
- 每任务独立模型、任务分类 head 或自动超参数搜索。

首阶段应优先保证数据语义正确、训练链路可复现以及评估使用真实 16D 动作。固定式 ALOHA 共训练只有在该基线稳定后才进入第二阶段。
