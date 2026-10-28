# tiny-NPU 学习路线（LEARNING GUIDE）

> 从 **tiny-gpu**（SIMT GPU）走到 **tiny-NPU**（dataflow NPU）的一条渐进式学习路线。
> 本文假设你已经跟着 tiny-gpu 的 `docs/`（ISA → 架构 → 执行 → 仿真 → 分支发散 → warp 调度 → warp 内发散）学过一轮，
> 这里用**你已经掌握的 GPU 概念**做桥梁，逐阶段拆解 NPU。
>
> 仓库：本仓库 `learning` 分支，文中路径都相对仓库根目录。环境搭建见 [`SETUP_NOTES.md`](./SETUP_NOTES.md)。

---

## 0. 这份文档怎么用

- 每个阶段（Stage）给出：**目标 · 关键 RTL 文件 · 运行命令 · GPU 类比 · 建议实验 · 深入方向**。
- 命令都假设你位于**仓库根目录**并已 `source env.sh`（激活 .conda-env 里的 Verilator 5.050 + .venv）。
- 阶段大致按依赖递进；**S0–S6 是 LLM Mode 主线**，**S7–S10 是 Graph/ONNX Mode 主线**，**S11 是 FPGA 上板 capstone**。
- 想快速看构建/运行命令，直接跳到文末 [速查表](#速查表)。

### 学习进度（完成一个阶段就在这里打勾）

- [x] **S0 环境与第一个仿真** —— 2026-08-24 完成
- [ ] S1 微码 ISA 与控制平面
- [ ] S2 脉动阵列与 GEMM
- [ ] S3 定点算术与数学引擎
- [ ] S4 Transformer Block 与控制编排
- [ ] S5 端到端 LLM 推理与 INT8 量化
- [ ] S6 KV-Cache
- [ ] S7 Graph / ONNX 模式
- [ ] S8 专用引擎
- [ ] S9 混合精度 INT8 / FP16
- [ ] S10 异步调度与性能计数
- [ ] S11 FPGA 上板（VCU118）· Capstone

---

## 1. NPU vs GPU：从 tiny-gpu 到 tiny-NPU

GPU 是**通用并行处理器**：一大堆线程各自跑标量指令（SIMT），靠海量线程隐藏访存延迟。
NPU 是**领域专用加速器**：核心是一块做矩阵乘的 **脉动阵列（systolic array）** + 一圈算子引擎，
靠**数据流复用**和**算子级流水**把利用率打满。tiny-gpu 教你"通用并行"，tiny-NPU 教你"专用数据流"。

| 维度 | tiny-gpu（SIMT GPU） | tiny-NPU（dataflow NPU） |
|---|---|---|
| 计算主体 | 多个 core，每个 core 若干 thread，逐 thread 跑标量 ALU | 单一 compute core：**16×16 MAC 脉动阵列** + softmax/LN/GELU/vec 等专用引擎 |
| 并行方式 | SIMT：同一指令、多线程数据并行 | **weight-stationary 脉动数据流**：权重驻留，激活流过阵列，边流边乘累加 |
| 基本运算单元 | 每 thread 一个标量 ALU | 每个 PE 一个 **MAC（乘累加）**，支持 INT8 + FP16 双数据类型 |
| 取指 / 控制 | 每 core 一套 PC + fetcher + decoder | **微码 sequencer**（`ucode_fetch`/`ucode_decode`）：一条 128-bit 指令驱动整块引擎 |
| 程序形态 | kernel = 一段线程程序 | program = 一段微码（LLM Mode）或一张算子图（Graph/ONNX Mode）|
| 存储层次 | registers + LSU 访问 global memory | **banked SRAM**（片上暂存）+ **AXI DMA** 搬 DDR |
| 控制流 | branch divergence / SIMT stack（tiny-gpu stage5–7 的重点） | **无线程分支**；用 **scoreboard + barrier** 做引擎间同步与并行 |
| 数值 | 定点 / 整数 ALU | 定点算术 `fixed_pkg` + **LUT**（exp/gelu/rsqrt/...）+ INT8 量化 + FP16 |
| 延迟隐藏 | warp scheduler 切换 warp | **async dispatch + tensor scoreboard**（DMA+GEMM overlap） |
| 对外接口 | 简单 memory 接口 | **AXI4-Lite**（控制寄存器）+ **AXI4 master**（DMA 数据面），可综合上 FPGA |

> 一句话记忆：**GPU 把并行藏在"很多线程"里，NPU 把并行藏在"数据在阵列里流动 + 算子在流水线里重叠"里。**

---

## 2. 两种执行模式概览

tiny-NPU 靠控制寄存器 `REG_EXEC_MODE`（AXI 偏移 `0x38`）在两种模式间切换，二者**共用**同一个 compute core 和片上 SRAM：

- **LLM Mode（`EXEC_MODE_LEGACY = 0`）**：128-bit **微码 ISA**，手工/脚本编排 transformer 的每一步。
  跑真实 GPT-2 / LLaMA / Mistral / Qwen2（INT8 量化 + 多头注意力 + KV-cache）。
- **Graph Mode（`EXEC_MODE_GRAPH = 1`）**：另一套 128-bit **图 ISA**（8-bit tensor ID）+ **tensor 描述符表** +
  13 个专用引擎。由 ONNX 编译器把算子图编译成图指令，硬件 FSM 自动遍历执行（含 async DMA+GEMM overlap、性能计数）。

学习上建议**先 LLM Mode（S1–S6）**打通"微码怎么驱动硬件"，**再 Graph Mode（S7–S10）**理解"编译器 + 图执行"。

---

## 3. 快速上手（先跑通再看细节）

```sh
source env.sh   # 在仓库根目录执行：.conda-env（Verilator 5.050）+ .venv

# 用 CMake 构建全部 21 个仿真 target（唯一权威路径）
cd sim/verilator && mkdir -p build && cd build
cmake .. && cmake --build . -j"$(nproc)"

./npu_sim        # 控制平面 smoke test（AXI 寄存器 + reset）—— 你的第一个 NPU 仿真
```

> ⚠️ 顶层 `Makefile` 的 `make lint` / `make sim` **已过时**（SV 源清单缺 `rtl/graph/*` 等），会报 `Import package not found: 'fp16_utils_pkg'`——**一律用上面的 CMake 路径构建**。`make luts` / `make ucode`（python 工具）仍可用。

---

## 4. 分阶段学习路线

### S0 — 环境与第一个仿真 ✅
- **状态**：**已完成**（2026-08-24 复核）。CMake 一次构建 21 个 target 全部成功；`./npu_sim` 输出 `=== ALL TESTS PASSED ===` —— Register access / DMA loopback / 控制平面 smoke（NOP 微码 ~99 cycle 后 `STATUS=0x1`）三项 PASS。
- **目标**：工具链跑通，看懂顶层接口。
- **关键文件**：`Makefile`、`sim/verilator/CMakeLists.txt`、`rtl/top.sv`（AXI4-Lite slave + AXI4 master + mode mux）、`sim/verilator/tb_top.cpp`。
- **命令**：见上面 [快速上手](#3-快速上手先跑通再看细节)；跑 `./npu_sim`。
- **GPU 类比**：等价于 tiny-gpu 的 `make test_matadd` —— 先证明"能跑"。
- **建议实验**：读 `top.sv` 的端口列表，数清楚控制面（`s_axil_*`）和数据面（`m_axi_*`）各有哪些信号。
- **深入**：`rtl/bus/axi_lite_regs.sv` 的寄存器堆是主机 ↔ NPU 的全部约定。

### S1 — 微码 ISA 与控制平面
- **目标**：吃透 128-bit 微码指令格式、控制寄存器、`START` 启动流程。
- **关键文件**：`rtl/pkg/isa_pkg.sv`（`opcode_t`：`GEMM/VEC/SOFTMAX/LAYERNORM/GELU/DMA_LOAD/DMA_STORE/KV_APPEND/KV_READ/BARRIER/RMSNORM/ROPE/SILU/END`）、`rtl/bus/axi_lite_regs.sv`（`reg_ctrl` 的 `CTRL_START`、`reg_ucode_base/len`、`reg_ddr_base_*`、`reg_model_*`、`reg_exec_mode`）、`rtl/ctrl/ucode_fetch.sv`、`rtl/ctrl/ucode_decode.sv`、汇编器 `python/tools/ucode_asm.py`。
- **指令格式**（记住字段随 opcode 复用）：
  ```
  [127:112 imm][111:96 K][95:80 N][79:64 M][63:48 src1][47:32 src0][31:16 dst][15:8 flags][7:0 opcode]
  ```
  例：GEMM 里 `M/N/K` = 行/列/内积维，`src0`=激活地址、`src1`=权重地址、`dst`=输出地址、`imm`=scale|shift、`flags`=transpose/requant/relu/...
- **命令**：`make ucode`（生成 tiny 测试微码 hex），再 `./npu_sim`。
- **GPU 类比**：tiny-gpu `stage1_isa`（指令编码）+ 控制寄存器，只不过这里一条指令驱动的是**整块引擎**而非一个线程。
- **建议实验**：用 `ucode_asm.py` 手写一段 `DMA_LOAD → GEMM → DMA_STORE`，看 `ucode_decode` 如何解码并派发到引擎。
- **深入**：对照 GEMM flags 表（`TRANSPOSE_B/BIAS_EN/REQUANT/RELU/CAUSAL_MASK/ACCUMULATE`）理解一条 GEMM 能干多少事。

### S2 — 脉动阵列与 GEMM（NPU 的心脏）
- **目标**：搞懂 16×16 **weight-stationary systolic array** 怎么做矩阵乘、怎么 tiling、怎么 requant。
- **关键文件**：`rtl/gemm/mac_int8.sv`（单个 MAC）、`pe.sv`（处理单元）、`systolic_array.sv`（16×16，可参数化到 24×24）、`gemm_ctrl.sv`（分块/地址）、`gemm_post.sv`（INT8 requantize）；golden：`python/golden/gemm_ref.py`。
- **命令**：`./engine_sim`（含 GEMM 在内的 6 项引擎测试）。
- **GPU 类比**：tiny-gpu **没有**阵列——这正是 NPU 的分水岭。对比：SIMT 每线程各算各的标量乘 vs 脉动阵列里权重驻留、激活横向流动、部分和纵向累加，**同一个数据被多个 PE 复用**。
- **建议实验**：把 `ARRAY_M/ARRAY_N` 从 16 调到 24，观察 `gemm_ctrl` 的 tiling 循环怎么变；追一次 GEMM 的数据在阵列里流动的时序（`make wave` + gtkwave）。
- **深入**：weight-stationary vs output/row-stationary 数据流的取舍（延迟、SRAM 带宽、复用率）。

### S3 — 定点算术与数学引擎
- **目标**：没有 FPU，如何算 softmax / layernorm / GELU 这些含 exp、除法、开方的算子。
- **关键文件**：`rtl/pkg/fixed_pkg.sv`（定点助手）；`rtl/ops/softmax_engine.sv`（+`reduce_max`/`reduce_sum`/`exp_lut`/`recip_lut`）、`layernorm_engine.sv`（+`mean_var_engine`/`rsqrt_lut`）、`gelu_engine.sv`（+`gelu_lut`）；LUT 生成器 `python/tools/make_lut.py`；golden：`softmax_ref.py`/`layernorm_ref.py`/`gelu_ref.py`。
- **命令**：`make luts`（生成 LUT ROM），再 `./engine_sim`。
- **GPU 类比**：从 tiny-gpu 的定点/整数 ALU 扩展到**用查找表逼近超越函数**——这是 NPU 省面积的经典手段。
- **建议实验**：改 `make_lut.py` 的精度/表大小重新生成 LUT，对照 golden 看数值误差随量化位宽的变化。
- **深入**：softmax 的 max-subtraction 数值稳定技巧在定点下如何实现（`reduce_max` 的作用）。

### S4 — Transformer Block 与控制编排
- **目标**：一条条微码如何**编排**出完整的 multi-head attention + FFN。
- **关键文件**：`rtl/ctrl/ucode_decode.sv`（派发）、`rtl/ctrl/scoreboard.sv`（6 引擎 busy 跟踪）、`rtl/ctrl/barrier.sv`（屏障同步）、`rtl/ctrl/addr_gen.sv`（地址生成）、`rtl/top.sv`（mode mux）；测试台 `sim/verilator/integration_top.sv`、`gpt2_block_top.sv`。
- **命令**：`./integration_sim`（单 head attention 集成）→ `./gpt2_block_sim`（完整 block：23 次硬件 GEMM、4 head）。
- **GPU 类比**：tiny-gpu 用 warp scheduler 协调多 warp；这里用 **scoreboard + barrier** 协调多引擎——都是"多个执行单元 + 依赖同步"，但粒度从线程变成算子。
- **建议实验**：读 `gpt2_block_top.sv` 的微码序列，按顺序数出 `GEMM/SOFTMAX/LAYERNORM/VEC` 的调用，画出一个 transformer block 的算子依赖图。
- **深入**：`BARRIER` 指令与 scoreboard 如何保证"下一步 GEMM 等到上一步 softmax 写完"。

### S5 — 端到端 LLM 推理与 INT8 量化
- **目标**：把**真实 GPT-2 权重**从 HuggingFace 导出、量化打包，跑出逐 token 生成。
- **关键文件**：`python/tools/export_gpt2_weights.py`（导 FP32）、`quantize_pack.py`（INT8 打包 `weights.bin`）、`python/golden/gpt2_infer_golden.py`（Python 金标准）、`sim/verilator/tb_demo_infer.cpp`、`sim/verilator/run_demo.sh`。
- **命令**（HF 权重经本机代理直连 huggingface.co；若不可达先 `export HF_ENDPOINT=https://hf-mirror.com`）：
  ```sh
  cd sim/verilator   # 从仓库根目录进入
  bash run_demo.sh --prompt "Hello" --max-tokens 10                 # 贪心（脚本无执行位，用 bash 调）
  bash run_demo.sh --prompt "Hello" --max-tokens 10 --temperature 0.8 --seed 42   # 采样
  ```
  > 该 demo 用一个缩小版验证配置（layers=4 hidden=64 vocab=256）；NPU 每步 logit 与内置 C++ golden 逐位一致。
- **GPU 类比**：tiny-gpu 只跑单个 kernel；这里第一次把"整块硬件"串成**完整模型推理**。
- **建议实验**：比对 `demo_infer`（NPU）与 `gpt2_infer_golden.py`（Python）是否 bit-exact；再换 LLaMA/Mistral/Qwen（`python/tools/llama_gen_weights_hf.py`、`mistral_gen_weights_hf.py`、`qwen_gen_weights_hf.py` + `llama_demo_infer`）。
- **深入**：INT8 对称/非对称量化、per-tensor vs per-channel scale 对精度的影响（看 `quantize_pack.py`）。

### S6 — KV-Cache（自回归解码优化）
- **目标**：理解为什么缓存 K/V 能让解码变快，硬件如何管理这块缓存。
- **关键文件**：`rtl/mem/kv_cache_bank.sv`（SRAM 后端）、`rtl/ctrl/kv_ctrl.sv`（KV FSM，桥接 SRAM0 ↔ kv_cache_bank）；ISA 的 `KV_APPEND`/`KV_READ`；映射 `python/tools/kv_map.py`。
- **命令**：`./kv_cache_sim`（对 full-recompute 做 bit-exact 校验）；`./run_demo.sh --prompt "Hello" --max-tokens 10 --kv-cache`。
- **GPU 类比**：本质是"用片上存储复用历史计算"——和 GPU 里的缓存/复用一个思想。
- **建议实验**：对比 kv-cache 与 full-recompute 的 cycle 数（README 称约 1.8×），画出随序列长度增长的曲线。
- **深入**：KV 指令为何要和 DMA 共享 scoreboard 槽位（互斥），append/read 的地址如何随 `token_idx`/`head_id` 推进。

### S7 — Graph / ONNX 模式（从"手写微码"到"编译计算图"）
- **目标**：理解硬件 FSM 如何自动遍历一张 ONNX 计算图并逐算子发射 DMA/GEMM/EW。
- **关键文件**：`rtl/graph/graph_fetch.sv`/`graph_decode.sv`/`graph_dispatch.sv`（主 FSM）/`graph_top.sv`、`tensor_table.sv`（256 项 tensor 描述符表）、`graph_isa_pkg.sv`（28 个 opcode，如 `OP_G_DMA_LOAD/OP_G_GEMM/OP_G_EW_ADD/OP_G_SOFTMAX/OP_G_REDUCE_*/OP_G_EXP/...`）；编译器 `python/onnx_compiler/compile.py`；切到 Graph Mode 需 `REG_EXEC_MODE=1`。
- **图 ISA 格式**：`[imm2 32][imm1 32][imm0 16][src2 8][src1 8][src0 8][dst 8][flags 8][opcode 8]`，`src*/dst` 是 **tensor ID**（指向描述符表）。
- **命令**（图 sim 是数据驱动：先用编译器把 ONNX 编到 `--datadir`，再跑 sim）：
  ```sh
  python python/onnx_compiler/compile.py --model models/mlp_32_16_8.onnx --outdir sim/verilator/build/graph
  cd sim/verilator/build && ./onnx_smoke_sim --datadir graph   # -> PASS: All 8 bytes match golden!
  ```
  其它 `onnx_*_sim` 同理（各自的 `--datadir` 名见 `sim/verilator/regression_all.sh`）。
- **GPU 类比**：这一步最像"真实 NPU 的软件栈"——编译器把图 lower 成硬件指令，硬件像一个小 FSM 的解释器去执行。
- **建议实验**：用 `python/onnx_compiler/gen_mlp_onnx.py` 造一个 MLP，`compile.py` 编译，跑 `onnx_smoke_sim` 看结果；打印编译出的图指令序列。
- **深入**：tensor 描述符表如何记录 shape/dtype/地址，`graph_dispatch` 如何据此驱动通用引擎。

### S8 — 专用引擎（为什么每个算子值得一块硬件）
- **目标**：理解 reduce/gather/slice/concat/pool/pad/resize/cast 等专用引擎的存在意义与实现。
- **关键文件**（均在 `rtl/graph/`）：`reduce_engine.sv`、`math_engine.sv`（Exp/Log/Sqrt/Rsqrt via LUT）、`gather_engine.sv`、`slice_engine.sv`、`concat_engine.sv`、`avgpool2d_engine.sv`、`maxpool2d_engine.sv`、`pad_engine.sv`、`resize_nearest_engine.sv`、`cast_engine.sv`。
- **命令**（同 S7，先 `compile.py` 生成各自 datadir 再跑）：`onnx_reduce_sim`、`onnx_math_sim`、`onnx_gather_sim`、`onnx_slice_concat_sim`、`onnx_batchnorm_pool_sim`、`onnx_resize_pad_sim`；覆盖性测试 `onnx_fuzz_sim`（50 个随机图）、`onnx_stress_sim`（56 节点 + 性能计数）。
- **GPU 类比**：GPU 用通用线程"凑"出这些算子；NPU 直接做**专用数据通路**换吞吐/能效。
- **建议实验**：读 `python/onnx_compiler/gen_fuzz_onnx.py`，弄清 fuzz 覆盖了哪些算子与形状；随机挑一个 fuzz 图追它的引擎调用。
- **深入**：BatchNorm 如何在编译期被 lower 成 scale+bias（`gen_batchnorm_pool_onnx.py` + 编译器）。

### S9 — 混合精度 INT8 / FP16
- **目标**：理解 INT8 与 FP16 两条数据通路如何在同一个 PE 里共存，以及 dtype 推断 + cast 插入。
- **关键文件**：`rtl/gemm/mac_fp16.sv`（2 级流水 FP16 MAC）、`rtl/gemm/pe.sv`（dual-MAC + dtype mux）、`rtl/graph/cast_engine.sv`、`fp16_utils.sv`、`graph_*_lut_fp16.sv`。
- **命令**：`./onnx_fp16_smoke_sim`、`./mixed_precision_cnn_sim`（INT8/FP16 混合 CNN，含 dtype 转换）。
- **GPU 类比**：类似 GPU 的多精度 tensor core，只是这里能看到 RTL 级的 dtype mux 与 cast。
- **建议实验**：在混合精度 CNN 里追一次 `INT8 → FP16` 的自动 cast 插入，看编译器在哪里判定需要转换。
- **深入**：为什么 requant 只在 INT8 路径需要，而 FP16 路径 `gemm_post` 直接 bypass。

### S10 — 异步调度与性能计数
- **目标**：理解 tensor scoreboard 如何实现 **DMA + GEMM overlap**，把利用率打上去。
- **关键文件**：`rtl/ctrl/scoreboard.sv`、`rtl/graph/graph_dispatch.sv`（`scheduler_mode` = SERIAL / OVERLAP、overlap/stall 性能计数器）。
- **命令**：`./onnx_overlap_perf_sim`（SERIAL vs OVERLAP 对比）、`./onnx_stress_sim`（读性能计数器）。
- **GPU 类比**：GPU 用 warp scheduler 切换 warp 隐藏访存延迟；NPU 用 scoreboard 让"搬下一块权重"和"算当前 GEMM"重叠——**同一个"延迟隐藏"母题，不同机制**。
- **建议实验**：跑 overlap_perf，记录 SERIAL 与 OVERLAP 的总 cycle、overlap cycle、stall cycle，算加速比。
- **深入**：scoreboard 如何用 tensor 依赖（读写 hazard）决定哪些算子可以并发发射。

### S11 — FPGA 上板（VCU118）· Capstone
- **目标**：把 `top.sv` 综合、实现、生成 bitstream，在 **VCU118（Virtex UltraScale+ VU9P）** 上真机跑一次。
  这是 tiny-NPU README "Next Steps" 里仍标记未完成的项（`[ ] FPGA synthesis and on-board demo`），是最好的动手项目。
- **关键文件 / 现状**：`rtl/top.sv` 已经是"Synthesizable SystemVerilog - Xilinx FPGA Target"，对外是 **AXI4-Lite slave（控制）+ AXI4 master（DMA 数据面）**；README 有 "Synthesize for FPGA (Vivado)" 小节和 16×16 阵列的资源估计。仓库**尚无** `.xdc` 约束和板级封装，需要你补。
- **本地环境**：本地 Mac 有 **Vivado + VCU118**；上板/综合流程参考本地 Mac 上的 **`vivado-on-silicon-mac`** 项目（Apple Silicon 上跑 Vivado + smoke / rocket 上板示例）。
- **建议路径**：
  1. **先跑 lint + OOC 综合**：`make lint` 通过后，在 Vivado 里对 `top` 做 out-of-context 综合，读资源占用，和 README 估计对照。
  2. **搭 Block Design**：`JTAG-to-AXI Master`（Vivado 内置，主机通过 TCL 驱动 AXI）→ NPU `s_axil_*`（写控制寄存器）；NPU `m_axi_*` → `AXI SmartConnect` → VCU118 板载 **DDR4（MIG）** 或先用 BRAM 顶替，跑通再换 DDR。
     - 注意 VCU118 无 PS（不是 Zynq），主机侧用 JTAG-to-AXI 或软核 MicroBlaze，不要照搬 Zynq PS 流程。
  3. **生成 bitstream 上板**：用 Vivado TCL / JTAG-to-AXI 写 `reg_ucode_base/len` + 一段 tiny GEMM 微码，拉 `CTRL_START`，读回结果寄存器验证。
  4. **放大规模**：VU9P 资源远超 16×16 阵列，可把 `ARRAY_M/ARRAY_N` 调大做吞吐实验。
- **GPU 类比**：tiny-gpu 主要停在仿真/GDS；tiny-NPU 的 AXI 接口让它天然可上板——这一步把"仿真正确"变成"硬件真的在跑"。
- **深入**：时序收敛（约束时钟、AXI 跨时钟域）、DMA 突发长度与 DDR4 带宽匹配。

---

## 速查表

### 构建 / 运行
```sh
source env.sh      # 在仓库根目录执行

make help          # 所有 Makefile target
make luts          # 生成 LUT ROM（.mem / _init.sv）
# ⚠️ make lint / make sim 已过时（缺 rtl/graph/*），别用；构建一律走下面的 CMake
make ucode         # 生成 tiny 测试微码 hex
make wave          # 生成 VCD，提示 gtkwave 命令
make clean         # 清理 build/ waves/ 生成物

# CMake 路径：一次构建 21 个 target
cd sim/verilator && mkdir -p build && cd build
cmake .. && cmake --build . -j"$(nproc)"
```

### 21 个仿真 target 速览
| 主线 | target | 说明 |
|---|---|---|
| LLM | `npu_sim` | 控制平面 smoke（AXI 寄存器 + reset）|
| LLM | `engine_sim` | 6 项引擎计算测试（GEMM/Softmax/LN/GELU/Vec/DMA）|
| LLM | `integration_sim` | 单 head attention 集成 |
| LLM | `gpt2_block_sim` | 完整 transformer block（23 GEMM，4 head）|
| LLM | `demo_infer` | 端到端 GPT-2 推理 demo |
| LLM | `kv_cache_sim` | KV cache 正确性（对 full-recompute bit-exact）|
| LLM | `llama_block_sim` / `llama_demo_infer` | LLaMA block（GQA/RoPE/SwiGLU）/ 端到端 |
| Graph | `onnx_smoke_sim` / `onnx_cnn_smoke_sim` | MLP / CNN smoke |
| Graph | `onnx_reduce_sim` `onnx_math_sim` `onnx_gather_sim` `onnx_slice_concat_sim` | 各专用引擎 |
| Graph | `onnx_batchnorm_pool_sim` `onnx_resize_pad_sim` | BatchNorm+AvgPool / Pad+Resize |
| Graph | `onnx_fuzz_sim` `onnx_stress_sim` | 50 随机图 / 56 节点压力+perf |
| Graph | `onnx_overlap_perf_sim` | DMA+GEMM overlap（SERIAL vs OVERLAP）|
| Graph | `onnx_fp16_smoke_sim` `mixed_precision_cnn_sim` | FP16 / INT8+FP16 混合精度 |

### GPT-2 / LLaMA 推理 demo
```sh
cd sim/verilator   # 从仓库根目录进入
bash run_demo.sh --prompt "Hello" --max-tokens 10                          # GPT-2 贪心
bash run_demo.sh --prompt "Hello" --max-tokens 10 --temperature 0.8 --kv-cache
```

---

## 常见问题（Troubleshooting）

- **`verilator: command not found`**：没 `source env.sh`；本机无 sudo、系统未装 Verilator，本项目用 conda-forge 的 **5.050**（在 `.conda-env/bin`，建法见 [`SETUP_NOTES.md`](./SETUP_NOTES.md)）。
- **权重下载卡住 / 403**：本机经 HTTP 代理 `127.0.0.1:5555` 直连 `huggingface.co`，先确认 `https_proxy` 已设置（`~/.bashrc`）；仍不通时 `export HF_ENDPOINT=https://hf-mirror.com` 改走镜像。
- **`ModuleNotFoundError`（numpy/torch/onnx）**：venv 没激活；`source env.sh` 会激活 `.venv`（torch 为 CPU 版）。
- **CMake 报版本/找不到 Verilator**：确认 `verilator` 在 PATH（`source env.sh`），CMake 靠 PATH 找到 `.conda-env/share/verilator/verilator-config.cmake`；本机系统 CMake 3.28.3 已满足 ≥ 3.19 的要求。
- **`make lint` / `make sim` 报 `fp16_utils_pkg` 找不到**：顶层 Makefile 已过时，改用 CMake 构建（见速查表）。
- **`./run_demo.sh: Permission denied`**：脚本缺执行位，用 `bash run_demo.sh ...` 调用。
- **ONNX sim 报 `Could not load program`**：该 sim 是数据驱动的，需先 `python python/onnx_compiler/compile.py --model models/<x>.onnx --outdir <dir>` 生成 `program.bin` 等，再 `./<sim> --datadir <dir>`。
- **想看波形**：`make wave` 依赖已过时的 `make sim`，不可用；改用 `NPU_DUMP=1 ./demo_infer --datadir demo_data --max-tokens 1` 生成 `demo_infer.vcd`。本机有 `/usr/bin/gtkwave`，但服务器没有 `DISPLAY`，需要 X11 转发或把 VCD 拷到本地打开。

---

## 参考

- 本仓库 `README.md`：架构、两种模式、完整 ISA 表、21 个 target 详解、FPGA 综合小节。
- 姊妹项目 `tiny-gpu`（及其 `docs/` 的 stage1–7）：SIMT GPU 视角，可反复对照。本机在 `../tiny-gpu`（与本仓库同级）。
- [`SETUP_NOTES.md`](./SETUP_NOTES.md)：本机工具链版本与踩坑记录。
