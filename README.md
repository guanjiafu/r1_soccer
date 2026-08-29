# R1 踢球机器人（R1 Soccer Robot）

基于**格物（Unity-RL-Playground）**平台，使用**宇树 R1 人形机器人**实现的踢球任务，通过强化学习训练，目前能做到「找球 + 踢球」。

## 文件说明

| 路径 | 说明 |
|------|------|
| `R1LoongWalkAgent.cs` | 主力 Agent：R1 双足 locomotion（参照 LoongAgent 的稳定走路训练方式）+ 追球转向 |
| `r1_loong_walk.yaml` | ML-Agents PPO 训练配置 |
| `play_soccer.unity` | 踢球场景（R1 + 球 + 球门） |
| `results/find_ball_new5/gewu.onnx` | 训练好的策略模型（找球 + 踢球） |

## 关节映射

R1 腿部 12 个 revolute 关节，顺序：

```
0 right_hip_pitch   1 right_hip_roll   2 right_hip_yaw
3 right_knee        4 right_ankle_pitch 5 right_ankle_roll
6 left_hip_pitch    7 left_hip_roll    8 left_hip_yaw
9 left_knee         10 left_ankle_pitch 11 left_ankle_roll
```

前馈关节索引：`idx = {-1, -4, -5, -7, -10, -11}`（对应左右腿 hip_pitch / knee / ankle_pitch）。

## 注意

本仓库只包含 `fgj_soccer` 的代码与策略模型，**不能独立运行**——场景依赖完整的 Unity-RL-Playground 工程（ML-Agents、R1 模型资源、球/球门等框架脚本）。完整工程见 [loongOpen/Unity-RL-Playground](https://github.com/loongOpen/Unity-RL-Playground)。

## 训练

```bash
mlagents-learn r1_loong_walk.yaml --run-id=find_ball_new5
```
