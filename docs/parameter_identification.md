# 硬件参数离线辨识

这个流程把机械臂的离线采样拟合成 `hardware_profile.yaml` 中的重力、
摩擦和 breakaway 参数。参数来源和模型形式与 reATRM Engineer 层保持一致；
reATRM 的参考提交是 `dfb4e83`（重力模型参数辨识）。

## 输入格式

CSV 至少包含以下列，角度单位为 rad，速度单位为 rad/s，力矩单位为 N*m：

```text
time_s,q1,q2,q3,q4,q5,q6,dq1,dq2,dq3,dq4,dq5,dq6,tau1,tau2,tau3,tau4,tau5,tau6
```

建议同时提供目标列，用于 breakaway 和摩擦拟合：

```text
q_target1..q_target6,dq_target1..dq_target6,phase
```

`phase` 可取 `gravity`、`friction`、`breakaway`。同一份 CSV 可以混合三段
实验，辨识器会按 phase 分开使用；没有 phase 时退回按速度/加速度筛选。

## 采集建议

1. **重力**：选取覆盖工作空间的 30-60 个静态姿态，每个姿态保持约 1 s，
   记录关节角和平衡力矩，标记 `phase=gravity`。不要让目标误差触发 breakaway。
2. **摩擦**：做多组恒速段，建议速度 0.05、0.1、0.2、0.4 rad/s，每个方向
   各一段，标记 `phase=friction`。恒速段中间的数据最可靠，换向点会被加速度筛选排除。
3. **breakaway**：从静止开始以很小的目标增量缓慢加载，记录目标角、实际角、
   实际速度和力矩，标记 `phase=breakaway`。没有目标列时 breakaway 会标记为未辨识。

力矩应来自与最终控制链一致的量，例如 reATRM 的
`/arm/joint_N/motor/control_torque` 或仿真中的 `/joint_states.effort`。
不要混用不同增益、不同补偿开关或不同标定阶段的数据。

## 运行

```bash
python scripts/identify_hardware_profile.py \
  --csv results/arm_identification.csv \
  --output results/identified_hardware_profile.yaml
```

默认使用 J5 的保守耦合投影形式。需要还原 reATRM 的原始经验耦合时加
`--non-conservative-j5`；默认用目标速度拟合摩擦，只有没有目标速度时才使用实际速度。

输出两个文件：

- `identified_hardware_profile.yaml`：可直接合并到
  `ros2_ws/src/jgarm_bringup/config/hardware_profile.yaml` 的参数块；
- `identified_hardware_profile.json`：每个关节的 RMSE 和有效样本数。

工具不会自动覆盖正在使用的 profile，合并前应检查残差和样本数。

## 模型与限制

- J2/J3 共用 `P*cos(theta)+Q*sin(theta)` 的三段连杆模型；J5 在保守模式下
  还共享 J3 的 `P/Q` 项。
- J1/J4/J6 的重力参数保持为 0，和 reATRM Engineer 模型一致。
- 摩擦使用 `torque * tanh(velocity / scale)`，拟合时对每个关节联合搜索
  `torque` 和 `scale`。
- breakaway 需要目标位置和目标速度；没有转入/转出摩擦死区的数据时，
  不应把拟合值当作已辨识值。
- breakaway 的误差尺度 `0.01 rad` 和速度阈值沿用 profile 默认值，本工具只拟合
  每关节的 breakaway 力矩。
- 该流程不辨识惯量、阻尼、电机时间常数、传动间隙或速度估计误差。
