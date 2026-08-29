# R1 踢球机器人（R1 Soccer Robot）

基于**格物（Unity-RL-Playground）**平台，使用**宇树 R1 人形机器人**实现的踢球任务。
通过强化学习（ML-Agents PPO）训练双足 locomotion，推理时自动追球，目前能做到「找球 + 踢球」。

---

## 文件结构

| 路径 | 说明 |
|------|------|
| `R1LoongWalkAgent.cs` | 主力 Agent：R1 双足 locomotion + 追球转向 |
| `r1_loong_walk.yaml` | ML-Agents PPO 训练配置 |
| `play_soccer.unity` | 踢球场景（R1 + 球 + 球门） |
| `results/find_ball_new5/gewu.onnx` | 训练好的策略模型（找球 + 踢球） |

---

## 核心思路：周期步态先验 + RL 残差

单纯让 RL 从零学双足行走非常困难（样本需求大、易摔倒）。本方案采用格物平台常用的
**「前馈步态先验 + RL 学习残差」**结构：

```
关节目标角 = 前馈步态（先验，生成基本迈步动作） + RL 残差（策略网络输出，微调）
```

- **前馈**：用半余弦波生成两条腿交替的踏步动作，保证机器人「天生会迈步」。
- **RL 残差**：策略网络输出 12 个关节的修正量，用来保持平衡、跟随速度/转向命令。

RL 只需要学「在迈步的基础上如何不摔倒、如何按命令走」，难度大大降低。

---

## 关节空间（12 个腿部关节）

R1 腿部 12 个 revolute 关节，`ActionNum` 顺序：

```
索引  右腿                    索引  左腿
0    right_hip_pitch       6    left_hip_pitch
1    right_hip_roll        7    left_hip_roll
2    right_hip_yaw         8    left_hip_yaw
3    right_knee            9    left_knee
4    right_ankle_pitch     10   left_ankle_pitch
5    right_ankle_roll      11   left_ankle_roll
```

---

## 1. 前馈步态生成器（Gait Prior）

在 `FixedUpdate` 里维护一个步态相位计数器 `tp`，生成两条腿交替的半余弦波：

```csharp
if (tp <= T1)      { uf1 = (1 - cos(2π·tp/T1))/2;  uf2 = 0; }
else if (tp <= 2T1){ uf1 = 0;                       uf2 = (1 - cos(2π·(tp-T1)/T1))/2; }
```

- `T1 = 30`（半个步态周期），`dh = 30`（抬腿幅度/度），`d0 = 5`（膝盖基础弯曲/度）。
- `uf1`/`uf2` 是 0→1→0 的余弦包络，两条腿相位相差半周期，交替迈步。

在 `OnActionReceived` 中，把步态叠加到三个关节上（以右腿为例，左腿对称）：

```csharp
float g1 = dh * uf1 + d0;   // 右腿当前步态幅值
utotal[0]  += -g1;          // hip_pitch   大腿后摆
utotal[3]  += 2f * g1;      // knee        膝盖弯曲（约大腿 2 倍）
utotal[4]  += -g1;          // ankle_pitch 脚踝反向补偿
```

这个 **`[-1, +2, -1]` 的「髋-膝-踝」比例**是双足迈步的基本协调模式：大腿折、膝盖弯、脚踝反，让脚掌始终朝前。

---

## 2. 动作平滑（低通滤波）

策略网络输出的原始动作不直接用于关节，而是先做一阶低通滤波（EMA）：

```csharp
u[i] = u[i] * 0.9f + 0.1f * action[i];   // 平滑后的 RL 残差
utotal[i] = kb[i] * u[i];                // 乘反馈增益
```

这样关节目标角是**连续平滑**的，避免动作突变导致机器人瞬间失稳。

---

## 3. 反馈增益 `kb`（动作空间尺度）

`kb[i]` 决定 RL 每个关节动作（范围约 [-1,1]）映射到多少度：

```
hip_pitch=30, hip_roll=10, hip_yaw=20, knee=10, ankle_pitch=30, ankle_roll=10
```

数值越大，该关节的「动作权限」越大。hip_pitch / ankle_pitch 权限最大（30），
hip_roll / knee / ankle_roll 权限较小（10），hip_yaw 居中（20）。

---

## 4. 观测空间（35 维）

`CollectObservations` 输出 35 维：

| 观测 | 维度 | 说明 |
|------|------|------|
| 重力方向（本体坐标） | 3 | 隐式编码俯仰/横滚姿态 |
| 角速度（本体坐标） | 3 | `wel[0..2]` |
| 线速度（本体坐标） | 3 | `vel[0..2]` |
| roll 角 | 1 | `eulerAngles[2]`（左右倾） |
| 关节角 | 12 | 每个 revolute 关节 |
| 关节角速度 | 12 | |
| `wr` 转向命令 | 1 | 策略要跟踪的 yaw 角速度 |

坐标系约定：**Z 轴 = 前进方向，X 轴 = 侧向，Y 轴 = 竖直向上**。

---

## 5. 奖励函数

每步奖励（`FixedUpdate` 中）：

```
r = live + (pitchPenalty + rollPenalty)·ko + welReward·kw + velReward
```

| 项 | 公式 | 含义 |
|----|------|------|
| `live` | `1` | 存活奖励，鼓励坚持不倒 |
| `pitchPenalty` | `-0.1·|pitch|` | 惩罚前后倾（`euler[0]`） |
| `rollPenalty` | `-0.1·|roll|` | 惩罚左右倾（`euler[2]`） |
| `welReward` | `-|ω_y − wr|` | 让实际 yaw 角速度跟踪转向命令 |
| `velReward` | `v_z − |v_x|` | 前进越快越好、侧向越少越好 |

**阶段式权重**：前 900 步 `ko=0.4, kw=1`（先学「走起来」），900 步后
`ko=1, kw=4`（重点压姿态、提转向精度）。

> 设计要点：刻意**不额外罚关节动作、不罚身体角速度**，只靠「前进速度 − 侧向速度 −
> yaw 跟踪 − 小姿态惩罚 + hip roll 钳制」，这是从 LoongAgent 借鉴来的稳定走路配方。

---

## 6. PD 关节驱动

最终关节目标角通过 `ArticulationBody.xDrive` 的 PD 控制器执行：

```csharp
drive.stiffness = 2000f;   // 比例增益（刚度）
drive.damping    = 100f;   // 微分增益（阻尼）
drive.forceLimit = 300f;   // 力矩上限
```

---

## 7. 追球逻辑（推理时 `train = false`）

```csharp
angleDiff = 身体朝向 与 「身体→球」方向 的水平夹角
wr = clamp(angleDiff * 0.3, -0.8, 0.8)   // 比例控制转向
```

推理时开启 `autoChase`，把「对准球」的角度误差映射成 `wr` 转向命令，训练好的 policy
会自动跟踪这个命令完成转向追球。

---

## 8. 训练技巧

- **`Time.fixedDeltaTime = 0.01`**：100 Hz 固定控制频率。
- **克隆 13 个机器人**（`Start()` 里 `Instantiate`）：并行采样，加速训练。
- **hip roll 钳制**：`utotal[1]`/`utotal[7]` 限制在 `[-10°, 10°]`，防止双腿越张越开（劈叉）。
- **终止条件**：`|pitch|>20°` 或 `|roll|>20°`，或步数 `tt≥1000` 超时。
- **命令随机采样**：训练时 50% 概率随机给 `wr∈[0.3,0.8]`（正负随机），策略学会「按命令转向」。

---

## 训练与推理

```bash
# 训练（在完整 Unity-RL-Playground 工程中）
mlagents-learn r1_loong_walk.yaml --run-id=find_ball_new5

# 推理：Unity 里 train=false，给 ball 赋值，autoChase 自动追球
```

训练超参（`r1_loong_walk.yaml`）：PPO，`batch_size=2048`，`buffer_size=20480`，
`lr=3e-4`，`hidden_units=512 × 3 层`，`gamma=0.995`，`time_horizon=1000`，`max_steps=2000万`。

---

## 注意

本仓库只包含 `fgj_soccer` 的代码与策略模型，**不能独立运行**——场景依赖完整的
Unity-RL-Playground 工程（ML-Agents、R1 模型资源、球/球门等框架脚本）。
完整工程见 [loongOpen/Unity-RL-Playground](https://github.com/loongOpen/Unity-RL-Playground)。
