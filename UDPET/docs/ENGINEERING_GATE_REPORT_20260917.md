# UDPET QuMod 工程门槛报告

**日期：** 2026-09-17  
**性质：** 工程诊断，不是模型效果实验，不读取 test split

## 结论

固定验证、两次真实 8-patch optimizer 更新、按 epoch 保存，以及中断后的严格恢复均已通过。最终恢复运行与不中断对照在以下状态上逐项完全一致：

- generator 和 discriminator 的全部张量；
- 两个 Adam optimizer 的全部状态；
- 两个 ReduceLROnPlateau scheduler；
- Python、NumPy、PyTorch CPU/CUDA RNG；
- 规范化 DataLoader generator 状态、sampler epoch 和训练合同哈希；
- `train_loss.csv`；
- 两轮 patient-metrics CSV 和 summary JSON。

因此，当前实现达到的是“相同起点、相同数据请求和确定性 CUDA 路径下，中断/恢复与连续执行得到相同训练状态和验证产物”。这不代表 QuMod 已复现成功，也不代表验证指标具有临床或统计意义。

## 最终对照协议

| 项目 | 设置 |
| --- | --- |
| 角色 | `engineering_diagnostic` |
| 数据 | train manifest + validation manifest；未读 test |
| 总 epoch / optimizer iteration | 2 / 2 |
| 中断位置 | epoch 1 的完整边界 |
| 每 iteration | 1 个真实配对，8 个 `80x80x80` patch；各执行一次 D/G update |
| QuMod | 开启 |
| validation | 固定 2 个配对，每配对 2 个 patch |
| 统计 | patient x DRF 聚合；200 次患者 bootstrap |
| 数值模式 | strict deterministic；cuBLAS `:4096:8`；TF32 关闭 |
| scheduler | ReduceLROnPlateau，以患者平衡 body RMSE SUV 为输入 |

恢复运行：

`/mnt/sdb/jinming.hu/UDPET/training_smoke/qumod_resume_exact_v6_20260917`

连续对照：

`/mnt/sdb/jinming.hu/UDPET/training_smoke/qumod_continuous_v6_20260917`

运行日志：

- `/mnt/sdb/jinming.hu/UDPET/training_smoke/qumod_resume_exact_v6_invocation1.log`
- `/mnt/sdb/jinming.hu/UDPET/training_smoke/qumod_resume_exact_v6_invocation2.log`
- `/mnt/sdb/jinming.hu/UDPET/training_smoke/qumod_continuous_v6.log`

## 固定验证定义

- body mask：NORMAL PET 中 SUV `> 0.2`；
- reference hotspot：每个采样 patch 内 NORMAL body voxel 的 top 1%；
- patient/DRF 指标：body MAE、RMSE、30-SUV data-range PSNR、body SUVmean 偏差、hotspot SUVmean 偏差；
- 汇总：先把 patch voxel 误差累加到 patient x DRF，再计算指标；总体结果先在患者内平均 DRF，再等权平均患者；
- 置信区间：患者级 bootstrap，不能对 patch 当作独立样本。

诊断子集的 body RMSE 为 epoch 1 的 `0.641635` SUV 和 epoch 2 的 `0.589368` SUV。由于只有两名患者，这两个值仅证明记录、汇总、scheduler 输入和恢复链路工作，禁止作为模型性能结论。

## 仍未通过的正式启动前事项

1. 完成旧/新 loader 的固定 200–500 batch 墙钟时间、CPU 内存和 GPU 等待对照；
2. 冻结 QuMod baseline 的完整训练配置、全 validation 请求集、训练预算和 checkpoint 选择规则；
3. 用 `run_role=qumod_baseline` 启动短预算基线，检查完整 validation 各 DRF 覆盖后再考虑长训练；
4. 在缺少 lesion mask 时继续禁用 LeMod，且不得把 reference-defined hotspot 写成病灶。
