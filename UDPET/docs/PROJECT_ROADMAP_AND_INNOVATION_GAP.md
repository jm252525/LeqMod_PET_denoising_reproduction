# UDPET 项目后续计划与课题创新目标差距

**更新日期：** 2026-09-17
**适用分支：** `codex/udpet-baseline-integration`

## 1. 拟回答的核心问题

本项目不把“低计数 PET 看起来更清晰”作为最终目标，而是检验：

> 相同名义剂量降低因子（DRF）对不同患者或检查并不具有相同的定量风险；与统一 DRF 阈值相比，经过校准的个体化风险模型能否在控制定量失败率的前提下，为更多检查选择更低计数或更短采集时间。

当前 UDPET 数据最适合先完成两层目标：

1. 在 D2、D4、D10、D20、D50、D100 到 NORMAL 的恢复中建立跨 DRF 的定量保真模型；
2. 根据部署时可见的低计数图像、DRF 和中心信息，预测一次恢复是否可能发生定量失败，并选择最低可接受计数水平。

“临床患者因素”和“跨扫描仪”是第三层扩展目标。现有 manifest 不含完整患者及扫描仪元数据，因此在补齐元数据以前，只能严谨地称为**检查/图像特异性风险**，不能直接宣称为由 BMI、注射剂量或扫描仪型号解释的患者特异性模型。

## 2. 当前仓库和数据状态

### 已完成

- 已将服务器上的 UDPET 适配代码纳入 Git 管理；
- 已固定患者级互斥的 train/validation/test manifests；
- train manifest 包含 5,020 个低计数—NORMAL 配对、909 名患者和 927 次检查；
- 已支持 D2、D4、D10、D20、D50、D100 联合读取；
- 已完成按检查分组的加权抽样、每 worker 有界 NORMAL 缓存、persistent workers 和回归测试；
- 在两 worker、5,020 次抽样的固定索引审计中，预计每轮 NORMAL 加载次数由 5,009 降至 1,824，约减少 64%；
- 真实 manifest 的 loader-only smoke test 已通过，没有启动正式训练。
- 已修复 PyTorch 2.7 scheduler 和训练轮数多一轮的问题；
- 已实现版本化的严格 epoch-boundary checkpoint，保存模型、optimizer、scheduler、RNG、DataLoader generator、sampler epoch 和训练配置哈希；
- 真实数据、8 个 `80³` patch、QuMod 开启的一次 D/G optimizer-step 已通过，峰值 allocated/reserved 显存分别为 7,166.5/10,196 MiB；
- 临时 checkpoint 回读后参数、optimizer、scheduler 及 CPU/CUDA RNG 均通过一致性验证。

### 尚未完成

- 尚无可审计的 QuMod/LeqMod 正式训练 checkpoint、训练曲线和验证结果；
- 尚未建立固定的患者级 SUV 定量评价程序；
- 当前网络未显式输入 DRF；
- 尚未训练定量失败风险或置信度校准模型；
- manifest 没有病灶分割路径，因此 LeMod 病灶采样和病灶定量损失当前不可用。

### 当前可用与缺失的变量

| 类别 | 当前可用 | 当前缺失或未确认 |
| --- | --- | --- |
| 配对影像 | 各低计数 PET、NORMAL PET、SUVbw 单位 | 病灶标注、器官标注 |
| 计数信息 | count label、count percent、DRF | 原始 prompts/counts 的统一物理量 |
| 数据来源 | site、cohort、patient/study ID | 可审计的 scanner manufacturer/model |
| 患者信息 | 仅匿名患者和检查标识 | 年龄、性别、身高、体重、BMI |
| 扫描协议 | 部分文件名线索、SUV scale factor | 注射活度、采集时长、摄取时间、重建算法与参数 |

Ruijin 2023 不含 D2，导致中心/年份与 DRF 覆盖不完全平衡。D2 的跨中心比较不能与其他 DRF 使用完全相同的设计，也不能把中心差异误解释为 DRF 效应。

## 3. 与创新目标的距离

| 阶段 | 目标 | 当前状态 | 到达下一阶段的关键缺口 |
| --- | --- | --- | --- |
| P0 数据与工程基线 | 患者级划分、可复现 loader、I/O 可用 | optimizer-step 与严格恢复已通过 | 完成墙钟速度实测和多 iteration 小训练 |
| P1 QuMod 可复现基线 | 固定配置完成训练并报告患者级结果 | 未完成 | 固定 validation、患者级指标和短预算基线 |
| P2 全 DRF 定量保真 | 恢复质量提升且 SUV 偏差受控 | 未开始 | 固定定量终点、DRF 条件化、损失消融 |
| P3 定量失败校准 | 对每次检查输出可靠失败概率 | 未开始 | 失败定义、校准集、风险覆盖评价 |
| P4 患者特异性解释 | 证明患者因素改变可接受 DRF | 数据受限 | BMI、剂量、时间等患者/协议元数据 |
| P5 分扫描仪安全范围 | 各 scanner 给出带置信区间的安全 DRF | 当前不可做严格结论 | scanner 型号、重建协议和足够独立样本 |

因此，项目已经具备工程起点，但还没有到达论文创新验证阶段。完成 P1 只能证明复现链路可运行；完成 P2 才形成“定量保真”主体；完成 P3 才能检验“统一 DRF 阈值不够”的核心假设。P4/P5 是否能够成为主结论，取决于能否补齐元数据。

## 4. 分阶段实验计划

### WP0：冻结工程与评价协议

任务：

1. ~~修复 PyTorch 2.7 scheduler 参数兼容性；~~（已完成）
2. ~~加入 GPU 单次 optimizer-step、保存和恢复测试；~~（已完成）
3. 用固定的 200–500 batches 对旧 loader 和新 loader 做墙钟时间、CPU 内存、GPU 等待时间对照；
4. 固定随机种子、患者划分、训练预算、验证 checkpoint 和配置哈希；
5. 明确每一轮/每个 checkpoint 的实际 optimizer updates，避免以 DataLoader batch 数代替有效更新数。

完成标准：同一配置可重复得到相同首批索引和近似相同首步损失；训练、验证、保存、恢复均能运行；正式运行产物不写入源码目录。

### WP1：建立可审计的 QuMod 基线

至少比较：

- `B0-MSE`：只使用基本重建损失；
- `B1-QuMod`：当前实现的 MSE + local SUV bias；
- `B2-QuMod-joint`：所有可用 DRF 联合训练，但不显式提供 DRF，作为后续条件化模型的公平对照。

所有模型使用相同患者划分、网络容量、训练预算和 checkpoint 规则。先用小规模训练排除工程错误，再启动完整训练。D2 因中心覆盖不完整，需要单独报告可比中心子集和全数据结果。

主要输出：每位患者、每个 DRF 的重建结果和定量误差；PSNR/SSIM 只作为图像质量辅指标。

### WP2：从“去噪”推进到“全 DRF 定量保真”

先固定评价终点，再修改模型。建议的患者级终点包括：

- body 区域 SUVmean 相对偏差；
- body 内积分 SUV（SUV × voxel volume）相对偏差；
- 由 NORMAL 定义的高摄取区域 SUVpeak 偏差；
- 预测与 NORMAL 的回归斜率、截距及 Bland–Altman 一致性；
- 每个 DRF 下超过预先规定定量容差的患者比例及 bootstrap 置信区间。

没有病灶标注时，由 NORMAL 自动定义的高摄取区域只能称为“reference-defined hotspot ROI”，不能称为真实病灶，也不能据此报告漏病灶率。

模型按一次只改变一个因素的顺序比较：

1. 所有 DRF 混合、无 DRF 标记；
2. 加入 DRF/count embedding 的同一网络；
3. 在条件化网络上加入全身或多尺度均值/积分 SUV 保真项；
4. 如获得病灶 mask，再增加 LeMod 病灶采样和病灶 SUV 项。

候选损失必须逐项消融。只有在验证集上降低 SUV 偏差，且不造成明显结构质量退化时才能保留。最终测试集只读取一次。

### WP3：校准定量失败与选择安全 DRF

先冻结 restoration 模型，再训练独立、轻量的风险模型，避免风险模型反向改变恢复结果。一次恢复是否失败应由预先冻结的定量容差定义；阈值需要临床或核医学依据，不能看测试结果后调整。

风险模型在部署时可使用：

- 名义 DRF；
- site/cohort；
- 仅从低计数 PET 计算的 body volume、SUV 分布、高摄取比例和噪声代理；
- restoration 网络的不确定性或残差代理。

不得使用同一患者的 NORMAL PET 或由 NORMAL 生成的评价标签作为风险模型输入。建议先比较正则化 Logistic 回归和一个小型非线性模型，再做概率校准。报告 AUROC/AUPRC 之外，还必须报告 Brier score、校准曲线、ECE、风险–覆盖曲线，以及患者 bootstrap 置信区间。

对每位患者，在 D2→D100 的候选水平中选择预测失败风险低于预设上限的最低计数水平，并与“所有患者使用同一 DRF 阈值”比较：

- 实际定量失败率是否仍受控；
- 平均可减少多少计数或采集时间；
- 有多少患者被允许使用更低计数；
- 有多少高风险患者被正确阻止。

这一步是当前数据条件下最接近课题核心创新的实验。

### WP4：补齐患者和扫描协议变量

若要把结论升级为真正的“患者因素改变可接受 DRF”，建议从原始 DICOM/申请方获取并加入独立 sidecar manifest：

- 年龄、性别、身高、体重和 BMI；
- 注射活度、注射至扫描间隔、每床位采集时长；
- scanner manufacturer/model；
- 重建算法、迭代次数、滤波、矩阵和 TOF/PSF 设置；
- 若开展病灶定量：病灶 mask、病理或随访确认信息。

元数据应与图像路径分离，使用匿名 patient/study key 连接，并做缺失率、单位和异常值审计。若这些数据无法获得，论文表述应限定为“基于低计数图像表型的检查特异性风险”，而不是“临床患者因素模型”。

### WP5：跨中心与跨扫描仪验证

- 现阶段可以做 leave-one-cohort/site-out，用于检验域偏移；
- site/cohort 不能自动等同于 scanner，不能把该实验写成严格 cross-scanner；
- 获得 scanner 字段后，按 scanner 划分患者，并确保同一患者只属于一个划分；
- 每台 scanner 分别报告定量失败率—DRF 曲线及患者 bootstrap 置信区间；
- scanner 样本不足时只报告探索性结果，不给出正式 safe operating envelope。

Leave-one-DRF-out 不作为临床主任务，只作为模型遇到未训练计数水平时的插值/外推压力测试。临床主任务仍是：对新患者在可获得的候选计数水平中选择最低安全 DRF。

## 5. 建议的最小论文实验矩阵

| 模型 | 多 DRF 联合 | 显式 DRF 条件 | 定量保真项 | 失败校准 |
| --- | --- | --- | --- | --- |
| MSE baseline | 是 | 否 | 否 | 否 |
| QuMod baseline | 是 | 否 | local SUV bias | 否 |
| DRF-conditioned QuMod | 是 | 是 | local SUV bias | 否 |
| Quantitative model | 是 | 是 | body + hotspot quantitative terms | 否 |
| Quantitative + calibrated risk | 是 | 是 | 同上 | 是 |

所有结果按患者聚合后再计算统计量和置信区间。模型/阈值选择只使用 validation，test 只用于一次固定报告。跨中心结果必须同时报告各中心和总体结果，不能只报告混合均值。

## 6. 建议的课题表述

在当前数据条件下，建议题目聚焦为：

> **多计数水平全身 PET 恢复的定量保真与检查特异性失败风险校准**

若后续补齐患者和扫描协议元数据，可升级为：

> **患者与扫描协议因素驱动的低计数全身 PET 定量风险建模及个体化安全 DRF 选择**

第一种表述依靠现有 UDPET 可以完整推进到 P3；第二种表述必须以 P4/P5 元数据补齐为前提。这样可以避免为了“患者特异性”而使用并不存在的临床变量，也能让研究生先完成一个闭环、可发表的主体工作。

## 7. 接下来三个代码里程碑

1. `engineering-gate`：scheduler、optimizer-step 和严格恢复已通过；剩余 loader 墙钟性能报告及多 iteration 小训练；
2. `qumod-baseline`：固定配置的全 DRF baseline、患者级定量评价脚本和验证报告；
3. `drf-conditioned-risk`：DRF 条件化消融、定量保真损失和验证集概率校准。

在第 1 个里程碑通过前不启动长时间正式训练；在第 2 个里程碑完成前不把当前代码称为 QuMod 复现成功；在第 3 个里程碑完成前不声称已经验证患者特异性安全 DRF。
