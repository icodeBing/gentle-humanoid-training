# G1SETTLE：阈值下柔顺，阈值上全身跟随

实现位于 `new_settle`。这是 GentleHumanoid 上新增的站姿／随力移动任务，不是原论文实验的复现。
本版保留 safe=5–15 N，复用已有 teacher → adapt → finetune，不新增显式外力估计头。
代码和 CPU 张量测试不能证明已学会走路，仍需实际训练与 Isaac 回放验收。

## 身体参考与上肢参考

新配置启用 `threshold_following: true`。普通策略观测的 `command` 只有 safe；默认关节姿态、默认重力方向和本体历史仍保留。
柔顺参考点／参考速度、仿真力和身体参考速度仅放入 `priv`，供教师编码；学生通过现有 adapt 模块预测隐特征。
相对回合初始朝向的 cos/sin 保留在普通观测中，它由姿态状态计算，与未知外力无关，便于学生纠正持续偏航。
因此，教师可用真实仿真载荷，学生动作推理不要求力或外力派生参考作为输入。
学生用到的根部线速度／历史与相对初始朝向需要部署端状态估计支持；这里没有实现硬件部署驱动或状态估计器。

上肢仍然积分：

```
Kp = safe / 0.05
Kd = 2 * sqrt(0.1 * Kp)
drive = clamp_norm(Kp * (nominal - x_ref) - Kd * v_ref, safe)
M_hand * a_ref = drive + F_interact - D_hand * v_ref
```

上肢位置／速度跟踪在静止和移动阶段始终启用。参考距初始站姿上肢点最多 0.25 m，触界去除向外积分速度，撤力后可恢复。
这只是几何限幅，不是完整关节可达性或碰撞检查。

身体参考使用当前实际连杆位置计算的物理载荷，不使用虚拟上肢让位后的 `force_expected`：

```
F_xy = planar(sum_i F_actual_model_world[i])
F_follow = F_xy * max(norm(F_xy) - safe, 0) / max(norm(F_xy), 1e-6)
M_body * dv_ref/dt = F_follow - D_body * v_ref
```

载荷在控制区间开始时按实际状态计算，速度／加速度分别限幅 0.8 m/s、2 m/s²。
物理子步中弹簧力仍随实际手部位置更新，因此 CSV 最后一个物理子步的 applied 可与区间开始的 `follow_input` 略有差别。
采用连续正部函数和身体导纳过滤阈值附近变化；没有增加隐藏的阈值滞回偏置。

单手、D_body=45、safe=15 时：10 N 和 15 N 的稳态身体参考为零；25 N 为 10/45=0.222 m/s；35 N 为 20/45=0.444 m/s。
撤力后参考平滑减速，仍奖励减速阶段的速度跟踪。
阈值作用于世界系水平合力：双手同向各 10 N、safe=15 时，跟随驱动为 5 N；双手反向等力时合力为零，不要求平移。
这里 safe 同时用作上肢每点驱动力上限与身体合力阈值；不是外力硬限幅。

## 训练载荷

- 默认每次事件 50% 为弹簧，50% 为世界系恒力；`constant_load_probability` 可调。
- 弹簧保留 K=5–250 N/m、随机方向和锚点。offset 每次事件 50% 固定 0.20 m，50% 从 0.10–0.25 m 采样。
- 恒力事件的世界系方向在事件内固定，身体转向不会使其一起旋转；方向随机覆盖前、后、侧向。
- 恒力总幅值分层：1/3 为 safe 的 0.2–0.9 倍；1/3 为 0.9–1.1 倍；1/3 为 safe 加 2–20 N。
  多点同向施力时总幅值按启用点均分，不会每只手各施加完整总幅值。
- 教师训练的前 25% 进度，将恒力超阈值增量从 35% 逐渐升到 100%；adapt/finetune 从全强度开始。
  课程只作用于新采样恒力事件的超阈值增量，不改变事件中已有载荷，不改变显式 stable 回放。
- 施力部位沿用 20% 无力、40% 单手、20% 双手、20% 随机上肢多点。随机多点也可能产生单／双手，因此最终占比略有差异。
- 每个事件有渐增、保持、渐减阶段；保持 300–900 控制步（6–18 s），渐增、渐减各 40–100 步（0.8–2 s）。
  无力事件仍保持完整时长，不在下一步立即重采样。
- 新版训练与 stable 回放都直接承受真实偏心外力，不再在躯干加反向补偿力矩。
  训练／spring 回放将合力按比例限制在 60 N；stable 回放严格执行命令指定载荷，不使用这一限幅。
  `net_torque_limit` 只在旧兼容模式的补偿逻辑中生效。
- 弹簧锚点仍保留根部坐标定义；它模拟不同于世界恒力的环境，不把两种载荷混为一谈。

## 奖励

- 上肢柔顺位置／速度跟踪：权重 2/1，所有阶段保留。
- 移动速度指数奖励：`exp(-(norm(v_xy-v_ref_xy)/0.10)^2)`，权重 6；乘以高度／倾斜平衡系数，降低倾倒时的收益。
- 全阶段连续速度误差：归一化尺度 0.20 m/s 的 Huber 代价，权重 1，避免指数项饱和后无法区分改进。
- 移动停滞惩罚：`-clamp(1-dot(v,v_ref)/norm(v_ref)^2,0,2)`，权重 2；只在参考速率 >0.01 m/s 时启用。
- 参考速率 <=0.01 m/s 时奖励低速站立；极小超阈值载荷的参考仍非零，但落在静止容差内。
- 移动时下肢默认姿态正则降至静止的 10%；保留高度、直立、朝向、yaw、脚滑、冲击与动作正则。
- 腾空／落脚奖励要求沿目标方向的进展并乘平衡系数，不通过原地跳动或倾倒赚奖励。
- 只有弹簧事件启用 applied−expected 力匹配和超力惩罚；恒力事件屏蔽这两项，因为恒力不由策略改变。
  恒力下 applied=expected 不是力估计／控制成功的证据。
- 保存执行区间的参考、safe、施力掩码与载荷类型；奖励比较该区间目标，避免与下一事件错配。

## 从头训练

普通策略观测维度和奖励语义变化，必须为此版本重新训练教师。不要直接加载旧 settle checkpoint。
在原本可以运行 Isaac 的训练环境执行，4096 环境按显存调整：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train.py \
  task=G1/G1_settle +exp=train \
  task.num_envs=4096 total_frames=2000000000 \
  wandb.mode=offline exp_name=G1SETTLE-threshold-v3-teacher
```

先验证教师在 10/15/25 N、safe=15 下的站立／跟随／平衡，再进入学生阶段：

```bash
TEACHER=/实际的新教师训练目录/files/checkpoint_final.pt
CUDA_VISIBLE_DEVICES=0 python scripts/train.py \
  task=G1/G1_settle +exp=adapt checkpoint_path="$TEACHER" \
  task.num_envs=4096 total_frames=1000000000 \
  wandb.mode=offline exp_name=G1SETTLE-threshold-v3-adapt

ADAPTED=/实际的adapt训练目录/files/checkpoint_final.pt
CUDA_VISIBLE_DEVICES=0 python scripts/train.py \
  task=G1/G1_settle +exp=finetune checkpoint_path="$ADAPTED" \
  task.num_envs=4096 total_frames=2000000000 \
  wandb.mode=offline exp_name=G1SETTLE-threshold-v3-student
```

沿用现有 PPO 实现：train 执行教师并训练特征预测；adapt 执行学生、训练特征预测；finetune 用 PPO 更新学生。
这些帧数是实验起点，不代表已知收敛预算。不要在教师还不会跟随时指望 adapt 自动补上行走能力。

## 固定条件回放

同一 checkpoint、seed、时序分别测试 10/15/25 N，便于判断阈值作用：

```bash
CKPT=/实际的v3训练目录/files/checkpoint_final.pt
for F in 10 15 25; do
  CUDA_VISIBLE_DEVICES=0 python scripts/eval.py \
    --checkpoint "$CKPT" --num-envs 1 --video --show-force \
    --duration 15 --seed 0 --settle-force-mode stable \
    --settle-force-magnitude "$F" --settle-force-direction 1 0 0 \
    --settle-force-bodies left_hand_mimic --settle-safe-force 15 \
    --settle-force-start 3 --settle-force-end 12 --settle-force-ramp 1 \
    --output-dir "outputs_eval/threshold_v3_left_${F}N"
done
```

0–3 s 站稳；3–4 s 渐增加力；4–11 s 恒力保持；11–12 s 渐减；12–15 s 撤力回稳。
10 N 应上肢让位而身体低速，15 N 是阈值边界，25 N 保持阶段身体参考趋于 0.222 m/s。
换右手、方向 `0 1 0`、`-1 0 0` 及 seed 1/2 复测。双手回放的 magnitude 是每只手的力，不是总力。

弹簧兼容性回放：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/eval.py \
  --checkpoint "$CKPT" --num-envs 1 --video --show-force \
  --duration 12 --seed 0 --settle-force-mode spring \
  --settle-force-bodies left_hand_mimic \
  --settle-spring-k 100 --settle-spring-offset 0.20 \
  --settle-spring-direction 1 0 0 --settle-max-force 60 \
  --settle-safe-force 15 --output-dir outputs_eval/threshold_v3_spring_left
```

本版身体参考取实际弹簧力超阈值部分，因此不要再用 expected/45 推测参考速度。
使用 `--settle-force-mode random` 且不指定载荷覆盖参数，可以复测完整的弹簧／恒力训练混合。
旧 checkpoint 缺少 threshold_following 时继续使用旧逻辑；相同 stable 命令不会自动给旧模型增加新能力。

## 日志和验收

训练日志保留 `settle/moving_velocity_error_m_s`、`moving_tracking_fraction`、`rest_speed_m_s`、
`yaw_rate_abs_rad_s`、`heading_error_abs_rad`、`active_hand_error_m`、脚滑和单支撑统计；
新增 constant_load_fraction、over_safe_fraction、known_planar_load_N、follow_excess_N、hand_reference_bound_fraction、load_curriculum_scale。

- `motion-*.csv` 新增：follow_input_xyz_N（区间开始物理载荷水平合力）、follow_excess_xyz_N、safe_force_N、constant_load、over_safe、balance_factor、balanced_tracking_ok。
  root_vel_ref 使用实际执行区间的参考；同时记录脚部接触、速率、高度及朝向误差。
- `force-*.csv` 保留 applied/interact/expected，新增每点实际／参考位置、速度、误差及最后一个虚拟子步的 drive（世界坐标）。
  位置是连杆坐标原点；当前施力位置仍为刚体质心，不能将这些位置当作测力传感器坐标。
- balanced_tracking_ok 同时要求速度向量误差 <0.05 m/s、沿目标方向速度达到 80%、平衡系数 >0.8、朝向误差 <0.25 rad。
  仍需结合脚滑与视频确认正常迈步；单凭该指标不能排除滑动。
- 上肢柔顺要看位置／速度参考跟踪、受力后让位及撤力恢复；恒力力误差为零不算柔顺验收。

## 本地检查

```bash
python -m unittest discover -s tests -p 'test_standing*.py'
```

测试覆盖阈值上下、实际／虚拟力分离、恒力世界方向、双手合力阈值、撤力、参考限幅、奖励掩码、
区间对齐、课程采样、学生无特权输入、回放日志及旧模式兼容。它不启动 Isaac，不验证训练收敛或实机效果。
