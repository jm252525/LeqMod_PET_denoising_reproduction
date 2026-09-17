# UDPET loader 墙钟与 GPU-forward 等待对照

**日期：** 2026-09-17  
**性质：** 工程性能诊断；未执行 optimizer update；未读取 test split

## 结论

当前 `volume_grouped_weighted + 1-reference worker cache + persistent workers` 不能被真实墙钟结果认定为吞吐优化。与 legacy loader 相比：

- 100-batch 段的中位墙钟时间从 `219.62 s` 增至 `220.97 s`，optimized 约慢 `0.62%`；
- 稳态取 batch 均值从 `2.0369 s` 增至 `2.0619 s`，optimized 约慢 `1.23%`；
- 同步 generator-forward pipeline 的等待占比从 `95.542%` 增至 `95.577%`，增加 `0.035` 个百分点；
- 进程树最大 PSS 从 `3438.4 MiB` 降至 `3279.2 MiB`，本次诊断约低 `4.63%`，但只有两次重复，不能把它解释成稳定的内存优势。

因此，先前“完整 epoch 的 NORMAL 解压次数预计减少约 64%”仍只是索引级预测，不能替代实际吞吐证据。当前实现可以保留为确定性读取方案，但不能再描述成已经验证的速度优化。

## 设计

采用 ABBA 顺序，减弱服务器页缓存和运行先后顺序的影响：

1. legacy repetition 1；
2. optimized repetition 1；
3. optimized repetition 2；
4. legacy repetition 2。

每段 100 batches，因此每个 loader 共测量 200 batches。每个 batch 是一个真实配对的 8 个 `80x80x80` patch。两种 loader 使用完全相同的 100 个 `(row_index, sample_seed)` 请求，并在第二次重复中原样复测：

- request multiset SHA-256：`059dceb692f0e1af0d0754a695fd3cb725ee223ae473799dca63738229bf870e`；
- 同一请求跨四段的 low/NORMAL + generator-output checksum 最大差异：`0.0`。

legacy 配置：

- `csv_weighted`；
- reference cache size `0`；
- non-persistent workers。

optimized 配置：

- `volume_grouped_weighted`；
- 每 worker 缓存 1 个 NORMAL reference；
- persistent workers。

共同配置为 2 workers、prefetch factor 2、固定增强和采样种子。GPU consumer 是 low/NORMAL H2D 加一次 deterministic generator forward；前 5 批只从稳态统计中剔除，仍计入总墙钟时间。

复现实验：

```bash
PYTHONPATH=/mnt/sdb/jinming.hu/UDPET/step5_env/pydeps:/mnt/sdb/jinming.hu/UDPET/metrics_step4/pydeps \
  /usr/bin/python3 UDPET/code/benchmark_loader_pipeline.py \
  --train-csv /mnt/sdb/jinming.hu/UDPET/metrics_step4/input/train.csv \
  --output-json /mnt/sdb/jinming.hu/UDPET/loader_benchmarks/loader_abba_200_per_arm_20260917/report.json \
  --batch-csv /mnt/sdb/jinming.hu/UDPET/loader_benchmarks/loader_abba_200_per_arm_20260917/batches.csv \
  --gpu-id 0 --num-batches 100 --steady-state-skip 5 \
  --memory-poll-seconds 0.25 \
  --arm-order legacy optimized optimized legacy
```

## 汇总结果

| 指标 | legacy | optimized | optimized 相对变化 |
| --- | ---: | ---: | ---: |
| 每 100 batches 中位墙钟时间 | 219.615 s | 220.971 s | +0.62% |
| pipeline batches/s | 0.45534 | 0.45255 | -0.61% |
| 稳态 fetch 均值 | 2.03691 s | 2.06186 s | +1.23% |
| 稳态 fetch P95 | 5.42495 s | 5.61358 s | +3.48% |
| GPU forward 均值 | 0.09505 s | 0.09542 s | +0.39% |
| forward-pipeline 等待占比 | 95.542% | 95.577% | +0.035 pp |
| 最大 tree PSS | 3438.4 MiB | 3279.2 MiB | -4.63% |
| 最大 tree RSS | 4407.9 MiB | 4247.2 MiB | -3.65% |

optimized 的单批 fetch 中位数显著较低，但 P95 更高，表现为“连续 cache hit 后立即返回，随后被新的 NIfTI 解压长时间阻塞”。这些长尾抵消了多数快速批次，因此均值和总墙钟没有改善。

## 边界

该等待占比是同步 generator-forward consumer 下的直接观测，不等同于完整 GAN 训练的 GPU idle 百分比。一次 D/G backward 明显慢于单次 generator forward，因而正式训练可能隐藏更多预取时间。本报告回答的是 loader 自身是否改善墙钟和较快 GPU consumer 是否饥饿，不能外推完整训练的 GPU utilization。

## 产物

目录：

`/mnt/sdb/jinming.hu/UDPET/loader_benchmarks/loader_abba_200_per_arm_20260917`

文件及 SHA-256：

- `report.json`：`9fb32db8bca5f8dccd1faeacbce17f2cf8c2adec088df08569f008273bb43c24`；
- `batches.csv`：`8d0bf0f1e452e2a1204efabbbecccf614a3a61d9e46323cb88d0d512a6248108`；
- `benchmark.log`：`1eecb7af7937c3bce7268d908ec92d61a583920f1f175bad5c7792f5aa3f07f1`。

## 下一步

在继续扩大 cache 或 worker 数之前，应先加入 20–50 batches 的分阶段 profile，分别计时：

1. low-count NIfTI load/decompression；
2. NORMAL load/decompression 和 reference cache hit；
3. crop/candidate selection；
4. patch copy 与 SciPy rotation；
5. pin-memory/H2D。

若 low-count `.nii.gz` 解压占主导，优先比较经过哈希审计的 float32 cropped `.npy` 或 chunked cache，而不是继续增加只缓存 NORMAL 的内存。若 CPU patch/rotation 占主导，再做 worker 数量和离线 candidate/patch metadata 消融。每次只改变一个因素，并继续用相同请求集比较。
