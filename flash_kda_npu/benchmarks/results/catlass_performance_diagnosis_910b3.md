# CATLASS 性能诊断（910B3，2026-10-19）

> Historical implementation: superseded by the AscendC-scheduled CATLASS kernels on 2026-10-20. See [current report](catlass_scheduled_910b3.md). Measurements below describe the earlier sources.

当前 `npu_catlass` 最大瓶颈是预处理/Gram，随后是求解和递推。变慢发生在将原先跨行/跨 chunk 的向量计算与片上复用，改成逐行向量操作和独立完整 GEMM 的实现过程中；不能据此判断 CATLASS 框架本身必然慢。

## 实测

固定输入种子 42，B=1、T=8192、H=96、D=128、chunk=64，FP32 初始/最终状态；两个后端在物理卡 6 顺序执行。预热 5 次，端到端 2×5 次 NPU event 计时；各阶段为 5 次同步 host 计时的中位数。所有调度配置的输出及最终状态，与各自后端默认配置逐位一致。

| 阶段 | 原 AscendC / ms | 当前 CATLASS / ms | 倍数 |
|---|---:|---:|---:|
| 预处理 + Gram | 4.001 | 66.157 | 16.53× |
| 求解（含组装、W/U） | 3.529 | 9.992 | 2.83× |
| 持久化递推 | 2.660 | 7.100 | 2.67× |
| 端到端，关闭 profile | 8.386 | 82.665 | 9.86× |

同步 host profiling 包含提交和同步成本，并改变流水重叠；阶段中位数不能直接相加当作端到端时间。约 79.5% 指 CATLASS 三个阶段 profile 值中的占比。预处理增加约 62.2 ms，是最明确的优化重点。

此前相同硬件的 graph replay 测量为原实现 8.230 ms、CATLASS 82.664 ms（见 `catlass_components_910b3.json`），与当前长序列端到端接近，说明主要问题在设备计算流水，而非 Python 提交开销。

## 源码证据和优化顺序

### 1. 优先重构预处理向量 tile

`csrc/catlass/prepare.hpp::BlockPrepare`：每个 AIV 按行执行 gate、Q/K 归一化、7 组输出。`LoadRow` 开始、`Export` 结束均调用 `Drain()`，其定义是 `PipeBarrier<PIPE_ALL>()`。完整 BF16-gate chunk 的这两个函数就执行 256 次行加载与 480 次行输出，共 736 次全流水 barrier；还没计入归一化、beta 和 Gram mask 的 barrier。H96/T8192 共 12288 个 chunk，约有 904 万次这样的 barrier。

每次仅处理 128 个元素，输出一次 256 B；计算、加载、存储频繁互等。Q/K 归一化每行归约后通过 `GetValue` 返回标量，再启动缩放；beta 同样通过标量读取形成依赖。Gram epilogue 又逐行加载，并用 `SetValue` 逐元素写上三角 mask。这些会缩小流水重叠空间。

建议：以多行 tile 批量加载 Q/K/V/g，批量做行归一化和 gate 的非递归部分；保留必要的 prefix 依赖；用矩阵/vector mask 代替逐元素标量 mask；批量输出，利用双缓冲和明确的 MTE2/V/MTE3 所有权。保留 CATLASS 上游 copy、elementwise 和 GEMM。UB 容量有限，需要按 tile 分块，不能简单把所有输出整块常驻。

不能直接删除 `Drain()`：当前共享 `landing/rounded/temp` 的复用需要保护，DMA 后的标量 UB mask 也已发现过缺少同步会导致精度错误。必须先改变缓冲区所有权，再缩小同步范围。预处理/Gram 作为一个 MIX kernel 计时，本轮未用硬件流水计数器进一步区分向量、Cube、DMA 各自等待比例；上述内部原因是源码证据，尚未分别消融测量。

### 2. 恢复求解的跨 chunk 向量化

`fwd_kernel1_solve_wide.cpp` 每次仅求解一个 chunk，`TileForwardSubstitution` 对 32 阶块的每个系数调用一次 `TileScale<32>`、一次 `TileAdd<32>`，各附向量 barrier；两个对角块共 992 次系数迭代。旧实现把多个 chunk 排列在同一向量指令的 repeat 中，并采用 `MulAddDst`。

建议：重排求解 tile，使多个 chunk 共同接受一次广播/向量指令；批量更新行，减少 scalar `GetValue` 和系数级 barrier。当前 host 的 batch 参数只让 block 串行循环多个 chunk，并未恢复这种向量化。保持 BF16 舍入位置及 FP32 累积语义。

### 3. 减少小 GEMM 的完整流水启动和重复搬运

`common.hpp::BlockGemm` 的 L1Shape、L0Shape 都等于整次 GEMM 的形状，因此 K 循环仅一次，所选 pingpong 流水没有后续 K tile 可重叠。上游 `block_mmad_pingpong.hpp` 构造设置 9 个 event，析构等待相应 event；当前每个乘积都重新创建并销毁对象。

每个 chunk 的 Gram 有 6 个 GEMM，求解组装/W/U 有 4 个，递推有 8 个，共 18 次完整调用。递推 W/Q 共用的 state，以及两个 value slice 共用的 W/Q，当前都重复从 GM 读取；旧实现会将公共操作数载入 L1/L0 后复用。

建议：先复用同形状 BlockMmad 实例，验证异步存储完成与事件所有权；再用 CATLASS 允许的 tile/block 组合设计成组乘积、公共操作数复用和更大的 tile。仅移动构造函数不会自动避免 GM 重读；也不能让两个共享 L0/event 的对象同时存活。减小 K tile 或更换 policy 需要实测，小 K 下额外分块也可能更慢。

### 4. 恢复递推的跨 head 流水

`recurrence.hpp::KernelRecurrence` 当前逐 head 完整执行 publish state → Predict → residual → Update → Finish。AIC 与 AIV 在四个跨核交接点等待，独立 head 间没有形成流水。可对多个 head 分阶段调度，双缓冲中间结果，使一侧处理 head n 时另一侧处理 head n+1。每个 head 的时间递推依赖仍需保留，不能并行展开不同 chunk 的状态更新。

## 已验证的参数优化及局限

| CATLASS 配置（H96） | 端到端 / ms | 递推阶段 / ms |
|---|---:|---:|
| 默认 pre_unroll=77、递推 blocks=40 | 82.665 | 7.100 |
| pre_unroll=2、blocks=40 | 80.564 | 7.127 |
| pre_unroll=77、blocks=24 | 86.392 | 10.997 |
| pre_unroll=77、blocks=96 | 82.719 | 7.067 |

另在物理卡 4 扫描 H16：默认约 14.768 ms，其中预处理/Gram 11.742 ms、求解 2.093 ms、递推 1.414 ms；主要瓶颈同样是预处理。该组不能与卡 6 数据作严格跨形状比例比较。

调小预处理 unroll 在 H96 形状约改善 2.5%；调整递推 block 数不能解决主体退化。没有据此修改生产默认值，其他长度/packed/head 数需要分别验证。详尽扫描保存在同名 JSON。

## 复现

先 source `/usr/local/Ascend/cann-9.1.0/set_env.sh`，然后对两个后端分别运行（顺序执行）：

```bash
ASCEND_RT_VISIBLE_DEVICES=6 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
KDA_BACKEND=catlass python -m flash_kda_npu.benchmarks.bench_schedule \
  --heads 96 --seq-lens 8192 --iters 5 --repeats 2 --json /tmp/catlass-schedule.json
```

将 `KDA_BACKEND` 换为 `ascendc`，并更换 JSON 路径。此轮未修改计算内核，也未重新运行整套精度测试；参数扫描已检查输出及状态逐位一致。建议先实现多行预处理，再依次验证严格精度、长序列 graph 性能、尾块和 packed 输入。
