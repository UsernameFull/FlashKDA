# 扩展 CATLASS tile 替换（910B3，2026-10-20）

在按 AscendC 调度重写的版本上，继续替换当前固定上游版本中有兼容接口的组件。保留 chunk/head 所有权、FP32/BF16 舍入合同和阶段依赖，不改精度测试阈值。未修改 vendored CATLASS 或 AscendC 后端，没有自动回退。

## 实际替换

- 所有普通/步长 GM↔UB 搬运改为上游 `CopyGm2UbAligned` / `CopyUb2GmAligned`；无 padding 的 byte-stride 输入改为 `CopyGm2Ub`。
- 已存为分形格式的 GM→L1 加载使用 `CopyGmToL1<..., zN>`；剩余三处批量 ND→NZ 使用 `CopyGmToL1GMMPTD` 的多矩阵接口，保留单条批量 DMA，不拆成逐矩阵循环。
- 普通 L1→L0 和 BF16 分形转置使用 `CopyL1ToL0A/B`，通过明确的 zN/zZ/nZ 布局保持分形偏移和原调用的 run 长度。转置使用该上游 BF16 特化中的 `LoadData(ifTranspose=true)`，替代直接 `LoadDataWithTranspose`。
- `Brcb` 使用 `TileBroadcastOneBlk`；bias、gate 跨块传播使用 `TileRowBroadcastAdd`；状态衰减使用 `TileRowBroadcastMul`；归一化和 beta 缩放使用 `TileOneBlkColumnBroadcastMul`，仍使用原来广播后的 8-lane 系数块。
- 三角求解的动态计数 scale/cast 循环改成完整 tile 与编译期尾部，使用 `TileElemWiseMuls` / `TileCast`。所有 FP32→BF16 向量 narrowing 均走上游 tile，BF16→FP32 保持 CAST_NONE。

这些适配器调用上游组件，不是给原 AscendC 调用换名字。矩阵乘、现有单矩阵 GM/L1 与 L0C/GM 存储、跨核 flag 沿用上一轮的上游组件。

## 剩余原语审计

下表统计六个 `.cpp` 文件中去掉注释后的直接调用位置，包含共享 helper 定义，属于静态调用点数，不是运行时指令次数；不统计适配层构造参数或上游内部的 AscendC 调用。

| 操作 | 本轮前 | 本轮后 |
|---|---:|---:|
| `DataCopy` | 50 | 1 |
| `DataCopyPad` | 7 | 2 |
| `LoadData` | 17 | 0 |
| `LoadDataWithTranspose` | 7 | 0 |
| `Brcb` | 4 | 0 |
| `Mul` | 12 | 0 |
| `Muls` | 1 | 0 |
| `Add` | 7 | 3 |
| `Cast` | 7 | 6 |
| `Fixpipe` | 2 | 2 |

保留项及具体原因：

| 部分 | 原因 |
|---|---|
| 1 处 UB→UB DataCopy | 残差按块 gather/分形重排；此上游树没有匹配的 UB→UB layout copy tile。 |
| 2 处 scalar DataCopyPad | Alog 与 packed beta 的精确补零/标量步长合同；普通 copy tile 禁用 padding，上游 per-token scale copy 不支持此 packed 步长与显式右补零组合。 |
| 3 处 Add | 1 处归约前半行相加，2 处 gate prefix 跨 8 行分组的步长相加；通用连续/广播 tile 无此 repeat 布局，逐行拆分会增加指令。 |
| 6 处 Cast | 都是 BF16→FP32 CAST_NONE；上游 TileCast 固定 CAST_RINT，在此 CANN/910B 转换上不匹配。 |
| 2 处 Fixpipe | L0C→L1 的 NZ 中间结果，以及带 ndNum/srcNdStride/dstNdStride 的批量 GM 输出；现有上游 L0C→GM tile 不暴露这些合同。 |
| identity 行广播的 Adds | 原调用一次 repeat 指令把独立 Eye 输入复制到多个 chunk；上游 inplace broadcast 要先初始化目标首行再逐行 Copy，不能保持现有单指令批处理。 |
| MulAddDst | 三角求解的跨 chunk 融合乘加；拆成 Mul+Add 会改变舍入和指令数。 |
| 非线性/归约与 prefix 控制 | 此上游树没有语义匹配的独立 exp/div/sub/adds/fill/reduce/scan tile；激活/softmax 复合组件不能直接替代 KDA gate。 |
| TPipe/TQue、局部 hard event/barrier | 管理原有 UB/L1/L0 生命周期和依赖；上游跨核 flag 不代替这些本地依赖。 |

这些是当前布局、批处理、数值与性能合同下的接口限制，不意味着相关算法理论上无法以自定义 tile 表达。仅包装原语不能算进一步采用上游实现。

## 数值与性能

物理卡 5，两个后端顺序执行；相同 CPU seed=42，6 种 shape × BF16/FP32/无初始状态。每配置预热 5 次，2×10 次 NPU event 计时。下表为 FP32 初始/最终状态，固定长 T=8192、D=128、B=1。

| 输入 | AscendC eager / ms | CATLASS eager / ms | AscendC graph / ms | CATLASS graph / ms |
|---|---:|---:|---:|---:|
| short | 1.2773 | 1.2432 | 0.1076 | 0.1199 |
| tail | 1.3223 | 1.2766 | 0.1311 | 0.1344 |
| fixed_h16 | 2.8564 | 3.0373 | 1.9845 | 2.1191 |
| fixed_h96 | 8.3105 | 9.0637 | 8.2373 | 8.9773 |
| packed | 3.5311 | 3.4034 | — | — |
| empty_packed | 1.5208 | 1.5127 | — | — |

18 组完整输出和最终状态全部与 AscendC 逐位相同，最大绝对差、相对 RMS 均为 0；固定长图回放与 eager 逐位一致。这不替代独立参考精度测试。

扩大上游覆盖存在性能代价：H16 graph 约比 AscendC 慢 6.8%，H96 约慢 9.0%；短序列相对差约 11.5%，绝对增加约 12 μs。仍与原调度同量级，远小于先前独立完整 GEMM 实现的退化。上游 tile 的布局遍历、指令组织与直接专用调用不同；本轮没有硬件计数器或分组件消融证据，不能将差值断言归因于某一个 tile。eager 也受并行 CPU 参考计算影响，应以 graph 对照评估设备时间。

## 验证

全部回归完成，无失败，精度阈值未改。合并不同配置后，1027 个不同测试 ID 通过。默认组跳过的长序列、其他 chunk 配置及双设备分支已单独运行。

| 测试组 | 通过 | 跳过 |
|---|---:|---:|
| 默认 chunk64 完整测试矩阵 | 1015 | 12 |
| 长序列（含 1M token） | 8 | 0 |
| chunk16 | 341 | 104 |
| chunk32 | 373 | 72 |
| 非融合求解精度/状态矩阵 | 174 | 0 |
| 双设备切换/跨设备输入检查 | 2 | 0 |

原有 gate sweep 状态 RMS 警告保留，最高约 0.7530%，未新增豁免或修改阈值。完整测试记录与 RMS 指标见 `catlass_extra_tiles_validation_910b3.json`。18 组跨后端逐位一致不代表对独立 FP64 参考零误差。

Host launcher 构建检查通过（已是最新）；实际执行的设备内核完成 RTC 编译。冻结源码哈希核对通过；`git diff --check` 通过。

## 复现

```bash
source /usr/local/Ascend/cann-9.1.0/set_env.sh
ASCEND_RT_VISIBLE_DEVICES=5 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m flash_kda_npu.benchmarks.bench_catlass --graphs \
  --warmup 5 --iters 10 --repeats 2 --json /tmp/extra-tiles.json
ASCEND_RT_VISIBLE_DEVICES=6 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
KDA_BACKEND=catlass python -m pytest flash_kda_npu/tests -q -n 4
```

原始数据：`catlass_extra_tiles_910b3.json`。静态操作审计和冻结源码 SHA256：`catlass_extra_tiles_audit_910b3.json`。上一轮调度重写与本轮替换前的数据见 `catlass_scheduled_910b3.md`。
