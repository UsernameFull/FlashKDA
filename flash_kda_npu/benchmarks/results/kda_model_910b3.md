# KDA sigmoid 1.36B：Ascend 910B3 端到端推理评估

本轮预填充加速 1.54–1.76 倍；128-token 生成总耗时从慢约 0.4% 到快约 1.6%，共同解码路径占主导，整次生成接近持平。默认 auto 捕获数为 0，增加预填充延迟和显存占用。自由生成并非完全一致，数值范围见下文。

模型：openeurollm/kda-sigmoid-1.3B-100B，1,362,630,016 参数；24 层纯 KDA，H=16/D=128，BF16 权重。
固定 revision：`88510f58bd0bba11104869f9d83d4006e083b037`；权重 SHA256：`8ece86b3ce8e6ab36ba542f25b6260276523b5dab5565f370c3c36f772c4a55f`（已验证）。

## 测试边界

- 单张物理 NPU 1（910B3/64GB），inference_mode，固定随机种子 42，单 CPU 计算线程。
- 保留 checkpoint 的投影、portable Conv1d、RMSNorm、门控 RMSNorm、MLP 和 LM head；仅更换核心 KDA，外围实现两组相同。不是整个模型所有算子的最优性能。
- FLA 基线使用当前安装的 FLA 0.6.0 NPU 后端；关闭 FLA 自动 FlashKDA 分发，显式适配 unsigned bounded sigmoid、beta sigmoid、Q/K L2 norm、scale 和 V-first FP32 state。
- FlashKDA 替换预填充；所有路径解码均为 FLA fused_recurrent_kda，未声称加速解码。
- 模型自带分发检查要求 ComplexKDA signed fork 和 CUDA，因此需要本次独立 adapter；无任何模型文件/权重改写。不能将自带 portable Torch 回退用作 FLA 性能基线。
- 输入为固定英文段落截取的 token 序列，非标准质量评测数据集。首次 tokenize 的超长语料提示来自生成候选语料；实际单序列预填充长度均为 128–4096。
- 用完整模型 forward（含 embedding、24 层、最后 token 的 LM head）测预填充；logits_to_keep=1，use_cache=True。不含磁盘加载/tokenize/服务排队。
- 稳定阶段每配置预热 3 次、测量 3 次，使用同步 wall-clock 中位数；min/max 和原始样本、峰值 allocated 显存见 JSON。首次 FLA forward 41.1 秒（含编译），不计入稳定性能。
- auto 使用现有默认容量 8、预热 3；它是按输入地址缓存算子，不是整个模型图。模型产生新中间张量且有 24 层，实际未捕获的项需据实记录，淘汰同步也有成本。

## 正确性

- T=128 logits 相对 L2：0.401%；逐位置 argmax 一致率：100.0%。
- 24 层最大输出/状态相对 L2：1.220% / 1.089%。这些是两条完整前向链路对应层的比较，含前序层误差累积。
- 同文本 next-token cross entropy：FLA 5.600338，FlashKDA 5.603721。这不是标准测试集 perplexity。
- FLA 对模型原始 portable Torch 路径，T=16 logits 相对 L2：0.426%。
- 自由贪心生成 32 tokens 完全一致：False；不能将本轮诊断称为生成等价认证。
- 同一组输入的 24 层核心对比：最大输出/状态相对 L2 0.474% / 0.403%。它排除了前序层误差累积，详见 same_input_core.json。
- 相同 token 历史的 32 步 teacher forcing：最大 logits 相对 L2 1.523%，argmax 一致率 96.88%；自由生成首次分叉在第 11 个生成 token。分叉处候选分差和 KL 详见 teacher_forcing.json。
- 本例同历史 argmax 不同的步骤：[{'step': 10, 'logits_relative_l2': 0.0037608302664011717, 'argmax_equal': False, 'fla_top1': 731, 'native_top1': 10230, 'fla_top2_gap': 0.0, 'kl_fla_to_native': 0.0002936827950179577}]。首次分叉处 FLA top-2 logits 同分；小精度差异改变了选择。
- 混合长度 packed logits 相对 L2：0.383%。

## 预填充（毫秒）

| Batch | 输入长度 | FLA | FlashKDA | 加速比 | auto |
|---:|---:|---:|---:|---:|---:|
| 1 | 128 | 150.629 | 85.881 | 1.75× | 91.960 |
| 1 | 512 | 151.072 | 86.027 | 1.76× | 94.521 |
| 1 | 2048 | 152.870 | 98.611 | 1.55× | 109.566 |
| 1 | 4096 | 175.649 | 113.386 | 1.55× | 143.855 |
| 4 | 128 | 152.398 | 86.863 | 1.75× | 93.521 |
| 4 | 512 | 153.030 | 99.556 | 1.54× | 107.622 |
| 4 | 2048 | 334.879 | 198.168 | 1.69× | 231.735 |
| 4 | 4096 | 668.582 | 395.591 | 1.69× | 438.477 |
| 8 | 128 | 156.749 | 98.504 | 1.59× | 104.533 |
| 8 | 512 | 175.606 | 109.701 | 1.60× | 142.570 |
| 8 | 2048 | 674.286 | 401.648 | 1.68× | 440.701 |
| 8 | 4096 | 1393.233 | 844.959 | 1.65× | 861.826 |

auto 捕获图数量：[0]。

## Packed 变长

长度为 [128,512,2048,4096]，合计 6784 有效 tokens；两端使用相同的分序列 checkpoint 卷积，不跨序列卷积。packed 性能测量不返回最终状态（use_cache=False），不与上面的 fixed 数字直接算变长开销。

- fla：312.174 ms，peak allocated 3.356 GB。
- native：238.988 ms，peak allocated 3.375 GB。

## 固定生成 128 tokens

使用显式 greedy loop，不包含 Transformers generate 或推理服务调度。不因 EOS 提前停止；首 token 由预填充 logits argmax 得到，此后执行 127 次共同 FLA 解码。TTFT 包含模型前向和 argmax，不含 tokenize/服务开销；总时间为分别同步测得的预填充与解码时间之和。每组一次完整生成预热，再测量 3 次。输入4096时加上生成会超过训练上下文4096，此项仅作性能压力测试。

| Batch | 输入长度 | 路径 | 首 token ms | 解码 tokens/s（batch 合计） | 总耗时 ms |
|---:|---:|---|---:|---:|---:|
| 1 | 128 | fla | 154.211 | 14.65 | 8823.940 |
| 1 | 128 | native | 85.828 | 14.68 | 8735.164 |
| 1 | 128 | auto | 92.764 | 14.76 | 8696.445 |
| 1 | 2048 | fla | 152.847 | 14.84 | 8710.539 |
| 1 | 2048 | native | 99.197 | 14.86 | 8644.401 |
| 1 | 2048 | auto | 109.704 | 14.76 | 8715.158 |
| 1 | 4096 | fla | 175.782 | 14.81 | 8750.946 |
| 1 | 4096 | native | 113.625 | 14.77 | 8714.517 |
| 1 | 4096 | auto | 144.593 | 14.77 | 8743.122 |
| 4 | 512 | fla | 154.765 | 58.73 | 8804.661 |
| 4 | 512 | native | 99.550 | 58.16 | 8839.087 |
| 4 | 512 | auto | 110.374 | 58.49 | 8794.304 |
| 8 | 512 | fla | 176.406 | 117.45 | 8826.798 |
| 8 | 512 | native | 107.708 | 118.51 | 8682.672 |
| 8 | 512 | auto | 141.118 | 118.19 | 8737.507 |

生成性能以共同解码路径为主；原始样本显示解码耗时存在波动，不能把不同组的解码速度微小差异归因于仅用于预填充的 FlashKDA。auto 未捕获图时更快或更慢的总生成样本也不代表图重放收益。

## 文件与复现

- run.sh：复现完整测试；bench_kda_model.py 为评估入口副本，项目源码在 flash_kda_npu/benchmarks/bench_kda_model.py。
- results.json / run.log：完整数值与运行日志；teacher_forcing.py/json/log：同历史数值诊断。
- results_fla_trace.json / results_native_trace.json：独立 profiler 窗口，不计入性能结果。
- download.py / verify_download.py：固定版本下载与完整 SHA256 验证。

## 环境

- torch: 2.10.0+cpu
- torch-npu: 2.10.0.post4
- transformers: 5.5.4
- flash-linear-attention: 0.6.0
- triton-ascend: 3.2.2

## Profiler 分发证据

- B=1/T=2048 的独立窗口中，FlashKDA 设备事件包括 24 次 kda_pre_gram_mix、24 次 kda_k2_persistent_loop；FLA 窗口可见其 NPU KDA kernels。详见 profile_summary.json。
- 跨 stream 累加的全部设备 kernel 时间：FLA 101.988 ms，FlashKDA 70.132 ms。此处未扣除重叠，也不是无 profiler 的模型 wall time，不能直接当端到端加速比。
- Profiler 停止时发出 schedule/RECORD 警告；两份导出的轨迹均能看到完整 24 层核心事件。轨迹用于核实分发和辅助定位，不替代独立性能测量。
