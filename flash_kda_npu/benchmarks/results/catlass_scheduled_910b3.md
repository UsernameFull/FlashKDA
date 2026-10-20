# CATLASS 按 AscendC 调度重写（910B3，2026-10-20）

> Historical schedule-preserving implementation before the additional upstream tile substitutions. See [current report](catlass_extra_tiles_910b3.md).

## 修改范围

所有六个设备阶段均依照 `csrc/ascendc/` 当前实现的调度、缓冲布局、计算顺序与舍入位置重写到独立的 `csrc/catlass/` 路径。旧的逐行/逐乘积独立 BlockMmad 实现已删除。未修改 AscendC 后端或精度标准。

- 预处理：恢复 16 行 pass、跨行归约/广播、完整 gate prefix、分 band 重定心、queue ring 以及 band mask/scale epilogue。
- Gram：公共操作数保留 L1/L0，各乘积之间复用；原有两 AIV/一 AIC 握手和 unroll 所有权保持。
- 求解：恢复跨 chunk 的 repeat 布局、系数广播与 MulAddDst；保留两级三角逆的计算及 BF16 舍入顺序。
- 组装/WU：恢复片上中间结果、批量 Fixpipe、inverse 复用和融合/非融合两种入口。
- 递推：恢复原实现按阶段组织的多 head 循环、FP32 UB 常驻状态、残差分形布局、公共操作数复用和下一步条件发布。

`scheduled_tiles.hpp` 调用上游 TileMmad、单矩阵 CopyGmToL1/CopyL0CToGm、兼容的连续 add/mul/scale/cast tile 和跨核 flag。使用 tile 层而不是完整 BlockMmad，才能由 KDA block 继续管理数据驻留与跨产品流水。

不是每条指令都来自上游 CATLASS：特殊分形 LoadData、批量 DMA/Fixpipe、repeat/broadcast 向量指令、MulAddDst、队列、局部 hard event、归约/非线性运算等保留自定义 AscendC 原语，以维持精确的布局和批处理合同。上游 TileMmad 自带小 tile 的 PIPE_M barrier，是明确保留的差异。CATLASS RTC 不包含或读取 `csrc/ascendc/` 源码，无自动回退。

## 同输入、同卡性能与数值比较

物理卡 5，Ascend910B3；两个后端顺序运行，输入由相同 CPU seed=42 生成。每配置预热 5 次，2×10 次 NPU event 计时；profile 关闭。其他卡同时运行 CPU 精度参考检查，因此 eager 提交时间可能有抖动；graph replay 更能体现设备时间。

以下为 FP32 初始/最终状态，B=1、D=128，fixed 的 T=8192。short 为 T=64/H4，tail 为 T=129/H4，packed 为 6 段总长 8192/H16，empty_packed 包含零长序列。

| 输入 | AscendC eager / ms | CATLASS eager / ms | AscendC graph / ms | CATLASS graph / ms |
|---|---:|---:|---:|---:|
| short | 1.3438 | 1.3556 | 0.1178 | 0.1173 |
| tail | 1.2793 | 1.2495 | 0.1257 | 0.1249 |
| fixed_h16 | 2.8419 | 2.7928 | 2.0163 | 2.0169 |
| fixed_h96 | 8.3004 | 8.3096 | 8.2371 | 8.2383 |
| packed | 3.3119 | 3.2731 | — | — |
| empty_packed | 1.4732 | 1.4809 | — | — |

18 组输入（6 种 shape × BF16/FP32/无初始状态）的完整输出和最终状态全部逐位相同，最大绝对差和相对 RMS 均为 0；所有固定长度图回放结果与 eager 逐位一致。此结论限于实测输入，不能替代独立精度参考测试。

此前完整独立 GEMM 版本 FP32 graph 的 H16/H96 分别约为 14.720/82.664 ms，当前约为 2.017/8.238 ms，约改善 7.3×/10.0×。历史数据来自物理卡 6，本轮为卡 5，作为历史量级对比；本轮 AscendC/CATLASS 的严格对照均在同一卡。

## 分阶段检查

同一物理卡 5，T8192/H96/FP32，默认调度：预处理+Gram 3.993 ms、求解 3.299 ms、递推 2.660 ms；关闭 profile 的端到端约 8.344 ms。阶段为 5 次同步 host 计时的中位数，会改变流水重叠并包含提交成本，不应直接求和作为端到端时间。

另外扫描 10 组 pre_unroll/block 配置，输出和状态均与默认配置逐位相同。默认调度已接近原实现性能，不修改默认参数。详见 `catlass_scheduled_stages_910b3.json`。

## 验证

未改变精度阈值；所有下列回归均完成，无失败。不同配置会重复执行同一测试 ID，合并后共 1027 个不同测试 ID 通过。默认组跳过的长序列、其他 chunk 配置和双设备分支已在相应专项运行。

| 测试组 | 通过 | 跳过 |
|---|---:|---:|
| 默认 chunk64 功能/精度 | 437 | 8 |
| 完整 CUDA 对齐 FP64 参考矩阵 | 578 | 4 |
| 长序列（含 1M tokens） | 8 | 0 |
| chunk16 功能/精度 | 341 | 104 |
| chunk32 功能/精度 | 373 | 72 |
| 非融合求解（选定精度/状态矩阵） | 174 | 0 |
| 双设备切换/跨设备输入检查 | 2 | 0 |

Host launcher 构建检查通过（已是最新）；设备内核通过实际调用完成 RTC 编译。`git diff --check` 通过。

保留原有 gate sweep 状态 RMS 的 RuntimeWarning；这不是降低阈值或新增豁免，详情与全部记录属性见 validation JSON：

- `/usr/local/python3.12.13/lib/python3.12/site-packages/_pytest/python.py:159: RuntimeWarning: gate sweep state relative RMS 0.00501984 >= 0.005`
- `/usr/local/python3.12.13/lib/python3.12/site-packages/_pytest/python.py:159: RuntimeWarning: gate sweep state relative RMS 0.00509257 >= 0.005`
- `/usr/local/python3.12.13/lib/python3.12/site-packages/_pytest/python.py:159: RuntimeWarning: gate sweep state relative RMS 0.0051913 >= 0.005`
- `/usr/local/python3.12.13/lib/python3.12/site-packages/_pytest/python.py:159: RuntimeWarning: gate sweep state relative RMS 0.00675994 >= 0.005`
- `/usr/local/python3.12.13/lib/python3.12/site-packages/_pytest/python.py:159: RuntimeWarning: gate sweep state relative RMS 0.00686533 >= 0.005`
- `/usr/local/python3.12.13/lib/python3.12/site-packages/_pytest/python.py:159: RuntimeWarning: gate sweep state relative RMS 0.00753037 >= 0.005`

验证数据：`catlass_scheduled_validation_910b3.json`。

## 复现与文件

```bash
source /usr/local/Ascend/cann-9.1.0/set_env.sh
bash flash_kda_npu/csrc/build.sh
ASCEND_RT_VISIBLE_DEVICES=5 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m flash_kda_npu.benchmarks.bench_catlass --graphs \
  --warmup 5 --iters 10 --repeats 2 --json /tmp/scheduled-comparison.json
ASCEND_RT_VISIBLE_DEVICES=6 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
KDA_BACKEND=catlass python -m pytest flash_kda_npu/tests -q -n 4
```

性能原始数据：`catlass_scheduled_910b3.json`；调度扫描：`catlass_scheduled_stages_910b3.json`；本轮设备源码及对应 AscendC 基线的 SHA256：`catlass_scheduled_sources_910b3.json`。
