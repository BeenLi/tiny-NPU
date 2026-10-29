> 本目录是 tiny-NPU 的学习文档，随仓库保存。文中的 `文件:行号` 都是相对本仓库的链接，点击可直接跳到对应代码行。

# tiny-NPU 学习计划：从会读 RTL 到读懂一块 LLM 推理加速器，再到 serving 系统

> 学习对象是 tiny-NPU：一个用 SystemVerilog 写的教学型 NPU，能跑 GPT-2 / LLaMA 的 INT8 推理（LLM Mode）和编译后的 ONNX 模型（Graph Mode）。
> 读者已经会读 RTL、学完了 tiny-gpu，但脉动阵列、量化、Transformer、编译器都是新东西，所以路线从熟悉的控制平面切入，每一步只引入一类新知识，并用 Sze 等人的书 *Efficient Processing of Deep Neural Networks*（2020）做理论参照。
> 终点是四件事：讲清 NPU 的设计取舍、能改能扩展、在 VCU118 上板、把单块加速器接到 LLM serving 系统（LLMServingSim）的视角里。

---

## 0. 仓库速览（2026-10-28 勘察，只读）

勘察时 `learning` 分支 HEAD 为 `7ad4e90`，其中 RTL / 仿真 / Python 代码与上游 `8216c22` 相同，只多了 Verilator 5.050 兼容修复（见 [`docs/SETUP_NOTES.md`](../SETUP_NOTES.md)）。下文行号都对应这个版本。

| 部分 | 位置 | 规模 | 备注 |
| --- | --- | --- | --- |
| 全局参数与 ISA | [`rtl/pkg/`](../../rtl/pkg/) | 4 文件 / 479 行 | 阵列尺寸、引擎编号、LLM 微码格式、定点工具 |
| 主机接口 | [`rtl/bus/`](../../rtl/bus/) | 4 文件 / 1069 行 | AXI4-Lite 寄存器、AXI4 DMA 读写 |
| 微码控制器 | [`rtl/ctrl/`](../../rtl/ctrl/) | 6 文件 / 1454 行 | fetch / decode / scoreboard / barrier / KV 控制器 |
| GEMM | [`rtl/gemm/`](../../rtl/gemm/) | 6 文件 / 1722 行 | INT8 + FP16 MAC、PE、16×16 脉动阵列、分块控制、重量化 |
| 算子引擎 | [`rtl/ops/`](../../rtl/ops/) | 14 文件 / 4223 行 | Softmax / LayerNorm / RMSNorm / GELU / SiLU / RoPE / Vec 与各类 LUT |
| 存储 | [`rtl/mem/`](../../rtl/mem/) | 4 文件 / 458 行 | 双口 SRAM、KV cache bank、乒乓 tile buffer |
| 图模式 | [`rtl/graph/`](../../rtl/graph/) | 25 文件 / 7754 行 | Graph ISA、调度 FSM、描述符表、13 个专用引擎、FP16 工具 |
| 芯片顶层 | [`rtl/top.sv`](../../rtl/top.sv) | 677 行 | 只是控制平面外壳，算子引擎为 done 桩 |
| 仿真 | [`sim/verilator/`](../../sim/verilator/) | 29 文件 / 19137 行 | 5 个仿真顶层 `*_top.sv` + 21 个 `tb_*.cpp` |
| Python golden | [`python/golden/`](../../python/golden/) | 13 文件 / 1263 行 | 与 RTL 逐位对齐的参考实现 |
| Python 工具 | [`python/tools/`](../../python/tools/) | 14 文件 / 2239 行 | 权重导出与量化、内存地图、微码汇编、LUT 生成 |
| ONNX 编译器 | [`python/onnx_compiler/`](../../python/onnx_compiler/) | 15 文件 / 3094 行 | `compile.py` 与各测试模型生成脚本 |

- 环境：在仓库根目录 `source env.sh`，安装与验证结果见 [`docs/SETUP_NOTES.md`](../SETUP_NOTES.md)。
- 构建：`bash build_sims.sh`（CMake，见 [`sim/verilator/CMakeLists.txt`](../../sim/verilator/CMakeLists.txt)）。
- 测试：`bash prep_sim_data.sh && bash run_tests.sh`；端到端 demo 见 [`sim/verilator/run_demo.sh`](../../sim/verilator/run_demo.sh)。
- 整体架构：[`ARCHITECTURE.md`](../../ARCHITECTURE.md)（12 节，静态源码分析）。

## 1. 读者背景与目标

- 背景：SystemVerilog / Verilog 能读，熟悉仿真（Verilator、cocotb）；学完了 tiny-gpu 的 stage1–7（ISA、架构、执行、仿真、分支发散、warp 调度）。脉动阵列 / TPU、Transformer 与量化、编译器 / ONNX 都还不熟。
- 目标：四个都要——
  1. 讲清 NPU 的设计思想，能和 GPU、工业 NPU 对照（偏研究）；
  2. 能改能扩展：加算子、修编译器、做性能实验（偏工程）；
  3. 在 VCU118 上板跑通；
  4. 为理解 LLM serving 系统打基础，最后衔接 LLMServingSim。
- 参考书（主）：Sze, Chen, Yang, Emer. *Efficient Processing of Deep Neural Networks*. Synthesis Lectures on Computer Architecture, Springer, 2020. DOI [10.1007/978-3-031-01766-7](https://doi.org/10.1007/978-3-031-01766-7)。下文写作“Sze 书 Ch.x”。
- 参考论文（辅）：同作者综述 *Efficient Processing of Deep Neural Networks: A Tutorial and Survey*, [arXiv:1703.09039](https://arxiv.org/abs/1703.09039), 2017，是书的前身，写作“Sze 综述 §X”。
- 两份 PDF 的本地副本在 `docs/learning/supplements/`（`sze2020.pdf` 与 2017 综述），受版权 / arXiv 许可限制只留本地、不入库。两者都以 CNN 为主，Transformer 相关内容基本是 🆕。
- 笔记里的 🆕 表示“Sze 书和 tiny-gpu 里都没有、这个仓库特有的东西”。
- 约束：持续学习项目，不设时间预算；每篇写到代码行级别。
- 参照：旧版路线 [`docs/LEARNING_GUIDE.md`](../LEARNING_GUIDE.md)（S0–S11，保留作速查）；姊妹仓库 tiny-gpu（`../tiny-gpu`）；LLMServingSim（`../LLMServingSim`，它自带的学习笔记在其 `docs/learningTutorial/`）。

## 2. 路线：第 00 步到第 16 步（2026-10-28 定）

顺序的逻辑：先建立“这个仓库怎么读”（00），从和 tiny-gpu 最像的控制平面切入（01）；然后按“理论 → 部件”的节奏补齐 NPU 的核心：数据流理论（02）→ 脉动阵列（03）→ 定点与量化（04）→ Transformer 模型本身（05）→ 算子引擎（06）；再把部件串成完整推理（07–09）；接着从“手写微码”走到“编译器 + 图执行”（10–12）；最后是动手改（13）、上板（14）、对照（15）和走向 serving 系统（16）。

| 阶段 | 主题 | 目标 | 总览 | 状态 |
| --- | --- | --- | --- | --- |
| 00 | 全貌：跑通 + 读法 | 说清一个 GPT-2 block 从 testbench 到 RTL 再到 golden 比对的全过程，以及为什么数据通路在仿真顶层 | [00 总览](00-0-orientation-overview.md) | 进行中 |
| 01 | 微码 ISA 与控制平面 | 能手工解码一条 128 位微码，说清 scoreboard 与 barrier 如何同步引擎 | [01 总览](01-0-control-plane-overview.md) | 框架 |
| 02 | DNN 加速的基本问题 | 说清数据复用为什么是核心，能区分 WS / OS / RS 数据流 | [02 总览](02-0-dnn-acceleration-overview.md) | 框架 |
| 03 | 脉动阵列与 GEMM 引擎 | 逐拍画出数据在 16×16 阵列里的流动，说清分块与 K 方向累加 | [03 总览](03-0-systolic-gemm-overview.md) | 框架 |
| 04 | 定点数与 INT8 量化 | 说清重量化公式每一项的来历与 shift 的取值 | [04 总览](04-0-fixed-point-quant-overview.md) | 框架 |
| 05 | Transformer 基础（软件视角） | 用 Python golden 讲清 attention、block、自回归生成与 LLaMA 的改动 | [05 总览](05-0-transformer-basics-overview.md) | 框架 |
| 06 | 非线性算子引擎 | 说清 Softmax / Norm / 激活 / RoPE 如何用定点和 LUT 实现并与 golden 逐位对齐 | [06 总览](06-0-nonlinear-engines-overview.md) | 框架 |
| 07 | Transformer block 的硬件编排 | 逐条读懂一个 block 的微码，说清 64KB SRAM 的规划 | [07 总览](07-0-block-orchestration-overview.md) | 框架 |
| 08 | 端到端 LLM 推理与 LLaMA 变体 | 说清从 HF 权重到采样的全流程、follow-actual 验证、GQA/RoPE/SwiGLU 的硬件落地 | [08 总览](08-0-llm-inference-overview.md) | 框架 |
| 09 | KV cache | 说清 KV_APPEND / KV_READ 与 kv_ctrl，以及与 PagedAttention 的关系 | [09 总览](09-0-kv-cache-overview.md) | 框架 |
| 10 | ONNX 编译器 | 读懂 compile.py 的量化、内存规划、lowering、golden 全流程 | [10 总览](10-0-onnx-compiler-overview.md) | 框架 |
| 11 | 图执行硬件与专用引擎 | 说清 graph_dispatch 的指令生命周期、描述符查表与 SRAM 仲裁 | [11 总览](11-0-graph-engines-overview.md) | 框架 |
| 12 | FP16、异步调度与性能计数 | 说清 FP16 数据通路、OVERLAP 模式的张量 scoreboard 与性能计数器 | [12 总览](12-0-fp16-async-perf-overview.md) | 框架 |
| 13 | 动手扩展 | 修已知 bug、打开 OVERLAP、加一个新算子、做性能实验 | [13 总览](13-0-hands-on-overview.md) | 框架 |
| 14 | FPGA 上板（VCU118） | 补齐 top.sv 数据通路，综合并在板上跑一个 block | [14 总览](14-0-fpga-bringup-overview.md) | 框架 |
| 15 | 对照与综合 | 用 Sze 的分类对照 TPU / NVDLA / Gemmini 与 GPU Tensor Core | [15 总览](15-0-comparison-overview.md) | 框架 |
| 16 | LLMServingSim 入门 | 从单块加速器上升到 serving 系统的性能模型，并把 tiny-NPU 接成一种硬件 | [16 总览](16-0-llmservingsim-overview.md) | 框架 |

### 第 00 步：全貌：跑通 + 读法

目标：建立这个仓库的读法——知道一个仿真 target 由哪个 `*_top.sv` 和哪个 `tb_*.cpp` 组成、用了哪些 RTL；并顺着一个 GPT-2 block 测试，看清 testbench 怎样驱动 RTL、怎样判定 PASS。环境已在 [`docs/SETUP_NOTES.md`](../SETUP_NOTES.md) 跑通，本步只读代码、按需复跑。

- [ ] 00-1 仓库地图与“两种顶层”。读：[`CMakeLists.txt:15`](../../sim/verilator/CMakeLists.txt#L15) 起的源文件分组（`SV_COMMON` / `SV_CTRL` / `SV_ONNX_SIM`，到 [`:136`](../../sim/verilator/CMakeLists.txt#L136)）、[`:149`](../../sim/verilator/CMakeLists.txt#L149) 起的 21 个 target 与它们的 `TOP_MODULE`；[`top.sv:538`](../../rtl/top.sv#L538) 的引擎桩和 [`top.sv:100`](../../rtl/top.sv#L100) 的 Graph Mode 占位；[`gpt2_block_top.sv:9`](../../sim/verilator/gpt2_block_top.sv#L9) 的端口；[`npu_pkg.sv:12`](../../rtl/pkg/npu_pkg.sv#L12) 起的全局参数。问题：21 个 target 各自用哪个顶层、编进了哪些 RTL？为什么完整的数据通路只在仿真顶层？拿到一个 target 名，怎么找到它的 RTL、testbench 和数据目录？
- [ ] 00-2 一个 GPT-2 block 的全链路。读：[`tb_gpt2_block.cpp:872`](../../sim/verilator/tb_gpt2_block.cpp#L872) 的 `main`，以及它依次调用的 [`generate_data_and_golden`](../../sim/verilator/tb_gpt2_block.cpp#L442)、[`load_microcode`](../../sim/verilator/tb_gpt2_block.cpp#L550)、[`load_data_to_srams`](../../sim/verilator/tb_gpt2_block.cpp#L736)、[`handle_dma_command`](../../sim/verilator/tb_gpt2_block.cpp#L787)；RTL 侧 [`gpt2_block_top.sv:113`](../../sim/verilator/gpt2_block_top.sv#L113) 的 `ucode_fetch`、[`:227`](../../sim/verilator/gpt2_block_top.sv#L227) 的 `ucode_decode`、[`:145`](../../sim/verilator/gpt2_block_top.sv#L145) 的 scoreboard、[`:321`](../../sim/verilator/gpt2_block_top.sv#L321) 的 `gemm_ctrl`、[`:881`](../../sim/verilator/gpt2_block_top.sv#L881) 的 `program_end` 锁存。问题：testbench 从写微码、写 SRAM、拉 start，到 tick 循环、拦截 DMA、等 `program_end`、读回比对，每一步对应 RTL 的哪个端口？PASS 的判据是什么（结果逐位比对、23 次硬件 GEMM、1 次 DMA 分别在哪里检查）？

### 第 01 步：微码 ISA 与控制平面

目标：能手工解码一条 128 位微码，说清从取指到引擎 done 回来的每一拍，以及和 tiny-gpu 控制器的异同。（开始这一步时再细化成逐篇待办）

- [ ] 128 位微码格式与各 opcode 的字段复用（`rtl/pkg/isa_pkg.sv`、`python/tools/ucode_asm.py`）
- [ ] 取指与译码：`ucode_fetch` / `ucode_decode` 如何把一条指令变成某个引擎的 cmd
- [ ] scoreboard、barrier 与 AXI-Lite 寄存器：引擎忙闲、同步与主机控制

### 第 02 步：DNN 加速的基本问题

目标：在读脉动阵列前，先回答“为什么要这样设计”。对照 Sze 书 Ch.1–2（DNN 概览）、Ch.3（指标与设计目标）、Ch.4.1–4.2（Toeplitz 展开与分块）、Ch.5.1–5.5（数据复用）。

- [ ] 计算量都在哪里：GEMM / 卷积、算术强度、访存比计算贵
- [ ] 数据复用与数据流分类：weight / output / row stationary，tiny-NPU 属于哪一类

### 第 03 步：脉动阵列与 GEMM 引擎

目标：逐拍画出数据在 16×16 阵列里的流动，说清分块控制与 INT8 / FP16 双 MAC。对照 Sze 书 Ch.5.6–5.7（数据流、循环嵌套与数据流分类）。

- [ ] MAC 与 PE：`mac_int8`、`mac_fp16`、`pe` 的双精度复用
- [ ] 阵列时序：激活、权重、部分和怎样逐拍流动（`systolic_array`）
- [ ] 分块控制器：`gemm_ctrl` 如何切 tile 并沿 K 累加，TRANSPOSE_B 怎么实现
- [ ] 对照 Sze 书 Ch.5.7 的数据流分类与 TPU v1

### 第 04 步：定点数与 INT8 量化

目标：说清 `(acc * scale + round) >> shift` 每一项的来历与 shift 的取值。对照 Sze 书 Ch.7.1–7.2（降低精度、位宽选择）。

- [ ] 对称量化与重量化：`fixed_pkg`、`gemm_post`、`python/golden/quant.py`
- [ ] 权重量化打包与 shift 选择：`quantize_pack.py`、`ddr_map.py` 的量化常量

### 第 05 步：Transformer 基础（软件视角）

目标：不看硬件，只用 Python golden 讲清一个 token 怎样变成下一个 token。

- [ ] 注意力与多头（`python/golden/attention_head_ref.py`）
- [ ] 完整 block 与 GPT-2 推理循环、prefill / decode（`gpt2_block_ref.py`、`gpt2_infer_golden.py`）
- [ ] LLaMA 系列的改动：RMSNorm、RoPE、GQA、SwiGLU（`rmsnorm_ref.py`、`rope_ref.py`、`silu_ref.py`、`llama_infer_golden.py`）

### 第 06 步：非线性算子引擎

目标：说清每个引擎如何用定点和 LUT 实现，以及 golden 如何逐位对齐。Sze 书只在 Ch.2.3 简单介绍非线性与归一化层，硬件实现基本是 🆕。

- [ ] Softmax 引擎：三遍扫描与 exp / 倒数 LUT
- [ ] LayerNorm 与 RMSNorm：均值方差、rsqrt LUT
- [ ] GELU / SiLU 与 Vec 引擎（含 COPY2D）
- [ ] RoPE 引擎

### 第 07 步：Transformer block 的硬件编排

目标：逐条读懂一个 GPT-2 block 的微码，说清 SRAM 规划和 head-blocked 权重布局。对照 Sze 书 Ch.5.8（缓冲区管理）、Ch.6.1（映射与循环嵌套）。

- [ ] SRAM0 / SRAM1 内存地图与 head-blocked 布局（`ddr_map.py`）
- [ ] `gen_block_microcode` 逐条走读：多头注意力循环
- [ ] FFN、残差与 barrier 的放置；周期都花在哪里

### 第 08 步：端到端 LLM 推理与 LLaMA 变体

目标：说清从 HF 权重到采样的全流程，以及 LLaMA 变体在硬件里怎么落地。

- [ ] 权重导出、量化打包与 `weights.bin` 布局
- [ ] `tb_demo_infer.cpp` 的推理循环与 follow-actual 验证
- [ ] LLaMA / Mistral / Qwen2：`llama_block_top.sv` 与 GQA、RoPE、SwiGLU、QKV bias

### 第 09 步：KV cache

目标：说清硬件 KV cache 的指令、控制器与加速来源，并和 serving 系统的 KV 管理对照。

- [ ] KV_APPEND / KV_READ 与 prefill / decode 微码
- [ ] `kv_ctrl` 状态机与 `kv_cache_bank` 的存储组织
- [ ] 对照：vLLM PagedAttention 解决的是什么问题

### 第 10 步：ONNX 编译器

目标：补上编译器基础，读懂 `compile.py` 的全流程。对照 Sze 书 Ch.6（映射器与编译器）、Ch.4.1（Toeplitz / im2col）。

- [ ] 计算图、量化与 DDR / SRAM 分配（`MemoryPlanner`、活跃区间分析）
- [ ] 算子 lowering：Gemm、Conv（im2col）、BatchNorm、Clip 等
- [ ] Graph ISA 编码、描述符表与 golden 计算

### 第 11 步：图执行硬件与专用引擎

目标：说清 `graph_dispatch` 的指令生命周期、描述符查表与 SRAM 仲裁。

- [ ] `graph_top` / `graph_fetch` / `graph_decode` / `graph_dispatch` 与 `tensor_table`
- [ ] `onnx_sim_top.sv` 的 SRAM0 仲裁与 DMA shim
- [ ] 代表性引擎精读：reduce、gather、pool 等

### 第 12 步：FP16、异步调度与性能计数

目标：说清 FP16 数据通路、OVERLAP 调度与性能计数器，并核实已知缺口。对照 Sze 书 Ch.7.3（混合精度）、Ch.3（吞吐与延迟指标）、Ch.6.5（Eyexam 性能分析）。

- [ ] FP16 MAC、`fp16_utils` 与 FP16 LUT、`cast_engine`
- [ ] OVERLAP 模式的张量 scoreboard
- [ ] 性能计数器与超时 / 错误检测

### 第 13 步：动手扩展

目标：在 `learning` 分支上做可提交的改动，每项单独提交。

- [ ] 修编译器：fuzz 的 `Add` / `Sub`、resize_pad 的空 `roi`
- [ ] 让 OVERLAP 模式真正跑一次，对比 SERIAL 的周期
- [ ] 加一个新算子：ISA → 引擎 → 编译器 → 测试
- [ ] 性能实验：阵列尺寸、KV cache 加速比等

### 第 14 步：FPGA 上板（VCU118）

目标：补齐 `rtl/top.sv` 的数据通路，综合并在板上跑一个 block。对照 Sze 书 Ch.3.4（硬件成本）。

- [ ] 把 `top.sv` 的引擎桩换成真实引擎（参考 `gpt2_block_top.sv` 的接线）
- [ ] OOC 综合与资源 / 时序分析
- [ ] Block Design（JTAG-to-AXI + DDR）与板上验证

### 第 15 步：对照与综合

目标：用 Sze 的分类，把 tiny-NPU 放进工业设计的谱系里。

- [ ] 对照 TPU v1 / NVDLA / Gemmini：数据流、存储层次、控制方式（Sze 书 Ch.3、Ch.5.7）
- [ ] 这个仓库没做的方向：稀疏（Sze 书 Ch.8）、高效模型设计（Ch.9）、近存 / 存内计算（Ch.10）
- [ ] 对照 GPU Tensor Core 与 tiny-gpu

### 第 16 步：LLMServingSim 入门

目标：从“一块加速器怎么算一个 block”上升到“一个 serving 系统怎么调度请求”，并把 tiny-NPU 接成一种硬件。LLMServingSim 的 PIM offload 可对照 Sze 书 Ch.10.1–10.2。

- [ ] 回顾与定位：五大组件与一次迭代的生命周期，和 tiny-NPU 的对应关系（复用 LLMServingSim 自带学习笔记的阶段 0–1）
- [ ] 逐层性能模型：`serving/core/trace_generator.py` 与 profile CSV 格式
- [ ] 实验：用 tiny-NPU 的周期数合成一份 profile，按其 `adding-hardware.md` 接入并跑一次模拟

---

## 3. 索引：资料在哪

| 内容 | 位置 |
| --- | --- |
| 学习笔记：00 全貌 | [00 总览](00-0-orientation-overview.md)（进行中） |
| 整体架构文档 | [`ARCHITECTURE.md`](../../ARCHITECTURE.md) |
| 环境与验证结果 | [`docs/SETUP_NOTES.md`](../SETUP_NOTES.md) |
| 旧版学习路线（速查） | [`docs/LEARNING_GUIDE.md`](../LEARNING_GUIDE.md) |
| 仓库自带文档 | [`README.md`](../../README.md) |
| 参考书 / 论文 | Sze 等 2020 书，DOI [10.1007/978-3-031-01766-7](https://doi.org/10.1007/978-3-031-01766-7)；2017 综述 [arXiv:1703.09039](https://arxiv.org/abs/1703.09039)；本地副本在 `docs/learning/supplements/`（不入库） |
| 上游仓库 | https://github.com/harishsg993010/tiny-NPU |
| 姊妹仓库 | tiny-gpu（`../tiny-gpu`，学习笔记在其 `docs/`） |
| serving 模拟器 | LLMServingSim（`../LLMServingSim`）；官方文档 https://llmservingsim.ai |
