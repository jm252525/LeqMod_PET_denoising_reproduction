# UDPET HDF5 训练门槛报告

**日期：** 2026-09-18  
**代码提交：** `3b5c2a7179175efbb6468ca152b9db81ff212792`  
**性质：** 单 GPU 工程诊断；不是 QuMod 效果实验；没有读取 test manifest

## 结论

HDF5 已通过正式训练入口所需的最后一项存储后端工程门槛：

1. train/validation 均可通过 HDF5 index 读取；
2. 后端类型、train index SHA-256、validation index SHA-256 和每 worker
   HDF5 handle 上限已写入严格训练合同；
3. HDF5 连续两轮与“一轮后中断 + 严格恢复一轮”的最终 checkpoint 全状态
   完全一致；
4. HDF5 与 NIfTI 连续两轮的模型输入、训练损失、验证结果、网络参数、两个
   optimizer、两个 scheduler 和 RNG 状态完全一致；唯一有意不同的是训练合同
   哈希，因为存储后端身份不同；
5. 使用 HDF5 checkpoint 改成 NIfTI 后端恢复时，被严格合同按预期拒绝。

因此，后续 QuMod 短预算和正式基线实验建议显式使用 HDF5 作为服务器工作
后端。NIfTI 仍是来源真值/归档格式；本实验不授权删除 NIfTI。LeMod 仍因没有
lesion mask 而不可用。

## 单 GPU 约束

- 设备：物理 GPU 2，NVIDIA L20；
- 首次启动前 GPU 2 为 17 MiB、0% utilization；每次启动前的 guard 均确认
  没有 compute process，否则命令会退出；
- 每个命令使用 `--gpu-ids 2 --device cuda`，进程内只可见一张 GPU；
- 三个实验臂严格串行运行，没有占用 GPU 0、1 或 3；
- 全部结束后，GPU 2 恢复为 17 MiB、0% 且没有 compute process。

## 存储合同

训练协议从 `udpet_training_v6_exact_resume_20260917` 升级为
`udpet_training_v7_storage_bound_resume_20260918`。严格训练合同新增：

| 字段 | HDF5 值 |
| --- | --- |
| backend | `hdf5` |
| train index SHA-256 | `fc631875bc52fb852e801b39b78c0f0bee2ea229640486452e7ed624e27266a2` |
| validation index SHA-256 | `ad5868e38695ee2af79949d23e7b1f08fc4fe310650eb13344ef25b547638ab1` |
| handle cache | 每 worker 1 个 study 文件 |

索引文件的路径记录在运行配置中，完整索引内容的哈希进入严格恢复合同。因而
索引文件所在目录可迁移，但索引行映射、元数据或指向的 cache path 发生变化时
必须重新建立实验合同，不能静默恢复。

## 有界实验设计

三个实验臂使用相同的 seed `20260918`、完整 train/validation manifest、
QuMod loss 和确定性 CUDA 设置。训练 manifest 含 5,020 个低计数配对、909 名
患者，但本工程诊断每 epoch 只抽取 1 个配对：

| 项目 | 设置 |
| --- | --- |
| 总 epoch / optimizer iteration | 2 / 2 |
| 每 iteration | 1 个配对，8 个 `80x80x80` patch；各 1 次 D/G update |
| validation | 固定 2 个配对，每对 2 个 patch |
| bootstrap | 200 次患者级重复，仅检查管线 |
| augmentation | 开启，随机性由显式 sampler request seed 控制 |
| scheduler | ReduceLROnPlateau |
| HDF5 handle cache | 每 worker 1 |

实验臂：

1. HDF5 连续执行两轮；
2. HDF5 执行第一轮后退出，从 epoch-boundary checkpoint 严格恢复第二轮；
3. NIfTI 连续执行两轮。

## 逐状态比较结果

HDF5 连续与严格恢复之间，最终 checkpoint 的以下内容逐元素完全一致：

- generator、discriminator；
- 两个 Adam optimizer；
- 两个 ReduceLROnPlateau scheduler；
- Python、NumPy、PyTorch CPU/CUDA RNG；
- loader generator、sampler epoch、训练合同哈希；
- completed epoch 和 total iteration。

下列外部产物也逐字节一致：`loader_smoke_test.json`、`train_loss.csv`、两轮
patient-metrics CSV 和两轮 validation summary JSON。

HDF5 与 NIfTI 连续运行也得到同样的逐状态、逐字节结果；比较时只排除了预先
声明且必须不同的 `training_state.training_contract_sha256`。负向测试把 HDF5
epoch-1 checkpoint 改为 NIfTI 后端恢复，进程以退出码 1 终止并报告
`Strict resume rejected: training contract mismatch`。

固定的两名验证患者仅产生工程检查值：epoch 1 body RMSE 为 `0.627760` SUV，
epoch 2 为 `0.551237` SUV。禁止把这些数值解释为泛化性能或临床结论。

提交后的 dataset cache、training state、scheduler、validation metrics、HDF5
patch cache、storage contract 和 loss-toggle 回归测试均通过；真实训练臂同时覆盖
了实际 CUDA forward/backward、D/G optimizer update 和 checkpoint 路径。

## 小样本运行时间

| 运行 | 墙钟时间 | 最大进程 RSS |
| --- | ---: | ---: |
| HDF5 连续两轮 | 12.55 s | 1,902,736 KiB |
| HDF5 中断第一段 | 7.90 s | 1,867,796 KiB |
| HDF5 恢复第二段 | 8.26 s | 1,897,980 KiB |
| NIfTI 连续两轮 | 22.99 s | 2,094,484 KiB |

连续臂中 HDF5 墙钟约为 NIfTI 的 `0.546x`（约 `1.83x` speedup）。样本只有两次
optimizer update，且操作系统页缓存状态无法完全配平，因此这里只作为“训练
入口没有抵消 HDF5 优势”的方向性证据。吞吐结论仍以 100-batch ABBA benchmark
为主。

## 产物

- HDF5 连续：
  `/mnt/sdb/jinming.hu/UDPET/hdf5_training_checks/hdf5_continuous_v7_gpu2_20260918`
- HDF5 严格恢复：
  `/mnt/sdb/jinming.hu/UDPET/hdf5_training_checks/hdf5_resume_v7_gpu2_20260918`
- NIfTI 连续：
  `/mnt/sdb/jinming.hu/UDPET/hdf5_training_checks/nifti_continuous_v7_gpu2_20260918`
- 机器可读比较结果：
  `/mnt/sdb/jinming.hu/UDPET/hdf5_training_checks/hdf5_training_equivalence_v7_gpu2_20260918.json`
- 后端漂移负向测试日志：
  `/mnt/sdb/jinming.hu/UDPET/hdf5_training_checks/hdf5_contract_negative_nifti_gpu2_20260918.log`

## 后续边界

HDF5 现在可以用于后续训练，但这不等于已经启动正式 QuMod baseline。正式长
训练前仍需冻结：训练预算、完整 validation 请求集、checkpoint 选择规则以及
最终报告指标。若服务器空间需要清理 NIfTI，应另行验证本地原始归档的逐文件
哈希和可恢复性，再执行删除；不能仅凭 HDF5 训练通过就把派生 cache 当成唯一
归档。
