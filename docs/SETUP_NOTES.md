# tiny-NPU 环境搭建记录（SETUP NOTES）

本机：Ubuntu 24.04.4 LTS（glibc 2.39，g++ 13.3.0，96 核 / 503 GB RAM），**无 sudo**。
工作分支 `learning`；文中路径都相对仓库根目录。
搭建日期：2026-10-28（初版 2026-07-05 在另一台 Debian 10 主机上完成，差异见文末“与初版主机的差异”）。

## TL;DR 怎么用

以下命令都在**仓库根目录**执行；需要进子目录的用 `( cd ... )` 子 shell，执行完仍回到根目录。

```sh
source env.sh        # .conda-env（Verilator 5.050 / Python 3.11 / uv）+ .venv

# 构建全部 21 个仿真 target（唯一权威路径，见下方"坑"一节）
bash build_sims.sh   # 等价于：cd sim/verilator && mkdir -p build && cd build && cmake .. && cmake --build . -j"$(nproc)"

( cd sim/verilator/build
  ./npu_sim            # LLM 控制平面 smoke（=== ALL TESTS PASSED ===）
  ./engine_sim         # 引擎计算测试
  ./integration_sim    # 单 head attention
  ./gpt2_block_sim )   # 完整 transformer block

# 端到端 GPT-2 推理（LLM Mode，需 .venv，HF 经本机代理直连）
( cd sim/verilator && bash run_demo.sh --prompt "Hello" --max-tokens 10 )

# Graph Mode（ONNX）：先编译模型到 datadir，再跑 sim
python python/onnx_compiler/compile.py --model models/mlp_32_16_8.onnx --outdir sim/verilator/build/graph
( cd sim/verilator/build && ./onnx_smoke_sim --datadir graph )

# 一次准备全部数据目录并批量回归
bash prep_sim_data.sh            # ONNX 编译产物 + fuzz + LLaMA 随机权重
bash gpt2_demo.sh                # 产出 demo_data（kv_cache_sim 需要）
bash run_tests.sh && cat test_results.log
```

## 安装的工具链

| 组件 | 版本 | 来源 / 位置 | 备注 |
|---|---|---|---|
| Verilator | **5.050** | conda-forge 预编译包 → `<repo>/.conda-env` | 无 sudo 装不了 apt（apt 候选也只有 5.020）；conda-forge 正好有 5.050，与初版主机验证版本一致，免去源码构建 |
| CMake | **3.28.3** | 系统 `/usr/bin/cmake` | 已满足 Verilator 5.050 `verilator-config.cmake` 的 **≥ 3.19** 要求，不用另装 |
| C++ 编译器 | **g++ 13.3.0** | 系统 `/usr/bin/g++` | `.conda-env` 里只有带前缀的 `x86_64-conda-linux-gnu-g++`，不会遮蔽系统编译器 |
| Python | **3.11.17** | `.conda-env/bin/python3.11` | 系统 Python 是 3.12 且没有 numpy，统一用 conda 里的 3.11 建 venv |
| uv | **0.13.3** | `.conda-env/bin/uv` | 同一个 conda 环境里装 |
| Python venv | 3.11.17 | `uv venv` → `<repo>/.venv` | 依赖直连 PyPI，torch 走 pytorch CPU index |

`.conda-env` 的创建命令（约 1.3 GB）：

```sh
# 在仓库根目录执行
conda create -y -p ./.conda-env -c conda-forge --override-channels python=3.11 verilator=5.050 uv
bash venv_setup.sh               # 建 .venv 并装依赖，日志 venv_setup.log
```

### Python 依赖（.venv 关键项）
```
numpy 2.4.6      scipy 1.17.1     onnx 1.23.2
torch 2.14.1+cpu   transformers 5.21.0
huggingface_hub 2.3.0   safetensors 0.8.0   socksio
# python 3.11.17；torch.cuda.is_available() == False（确认为 CPU 版）
```

## 网络环境

- 本机所有出站流量走 HTTP 代理 `http://127.0.0.1:5555`：`~/.bashrc` 导出 `http_proxy` / `https_proxy`，
  `~/.gitconfig` 设置了 `http.proxy` / `https.proxy`。
- 经代理 **huggingface.co、pypi.org、download.pytorch.org、github.com 都可直连**，因此：
  - `env.sh` **不再**设置 `HF_ENDPOINT`；若 huggingface.co 将来不可达，手动 `export HF_ENDPOINT=https://hf-mirror.com`。
  - pip/uv **不再**用 Aliyun 镜像（实测 PyPI 约 9.8 MB/s，而 Aliyun 对 `packages/source/...` 路径返回 404）。
- torch CPU 仍用 `--index-url https://download.pytorch.org/whl/cpu`（**不要**再加 `--extra-index-url`，
  否则解析器会拉成 CUDA 版、带一堆 `nvidia-*` 包）；torch 的纯 python 依赖先从 PyPI 预装。
- 代理是 HTTP 而不是 SOCKS，所以不会触发 `socksio` 缺失的 `ImportError`；`venv_setup.sh` 仍顺手装了 `socksio` 以防万一。

## 对仓库做的改动（为兼容 Verilator 5.050 / 新版 g++）

这些改动在 `learning` 分支上，在初版主机（g++ 8.3）上引入，在本机（g++ 13.3）上同样需要并已验证。

1. **`sim/verilator/CMakeLists.txt`**：`VERILATOR_COMMON_ARGS` 追加 `-Wno-SYNCASYNCNET` 和 `-Wno-fatal`。
   - Verilator 5.050 **默认把 warning 当致命错误**，比仓库预期的旧版严格；`axi_dma_rd.sv` 的 `rst_n` 触发
     `SYNCASYNCNET`（复位信号同时被当同步/异步用，良性 lint）。`-Wno-fatal` 让 warning 只打印不中断。
2. **5 个 C++ 测试台**（`tb_gpt2_block.cpp`/`tb_demo_infer.cpp`/`tb_kv_cache_sim.cpp`/`tb_llama_block.cpp`/`tb_llama_demo_infer.cpp`）：
   `memset(dut->uc_wr_data, ...)` → `memset(&dut->uc_wr_data, ...)`。
   - Verilator 5.050 把 128-bit 端口表示成 `VlWide<4>` 对象（不再隐式退化为指针），需取地址。
3. **辅助脚本不再写死仓库路径**：`env.sh`、`build_sims.sh`、`run_tests.sh`、`gpt2_demo.sh`、`gpt2_demo100.sh`、
   `venv_setup.sh`、`prep_sim_data.sh` 都用脚本所在目录（`$ROOT`）定位仓库，换机器或换目录无需修改。
4. **`.gitignore`**：新增 `.conda-env/`、`.venv/`、`models/fuzz/`、`*.FAIL`。

## 坑 / 注意事项

- **顶层 `Makefile` 的 `lint` / `sim` target 已过时，别用**：其 SV 源清单缺 `rtl/graph/fp16_utils.sv`（定义
  `fp16_utils_pkg`，被 softmax/layernorm/gemm 引用）、`mac_fp16.sv`、整个 `rtl/graph/*`，`make lint` 会报
  `Import package not found: 'fp16_utils_pkg'`。**一律以 CMake 构建为准**（`sim/verilator/CMakeLists.txt` 才是
  完整、权威的源文件清单，能构建全部 21 个 target）。`make luts`/`make ucode`（python 工具）仍可用。
- **LUT 无需运行时文件**：RTL 里没有 `$readmemh`，LUT 内容编译进 `.sv`，所以构建/运行仿真**不需要先 `make luts`、
  也不需要 venv**；venv 只在跑权重导出 demo（GPT-2/LLaMA）和 ONNX 编译器时才需要。
- **ONNX 图 sim 是数据驱动**：每个 `onnx_*_sim` 用 `--datadir <dir>` 读 `program.bin/tdesc.bin/ddr_image.bin`。
  这些要先用 `compile.py` 生成；`prep_sim_data.sh` 按 `sim/verilator/regression_all.sh` 的目录名一次生成全部。
  不带 `--datadir` 时 tb 默认读相对路径 `build/graph_xxx`，从 `build/` 目录里运行会找不到——`run_tests.sh` 已显式传参。
- **`sim/verilator/*.sh` 缺执行位**：用 `bash <script>` 调用（如 `bash run_demo.sh ...`）。
- **conda 环境不必 `conda activate`**：`env.sh` 只把 `.conda-env/bin` 加到 PATH 最前；CMake 通过 PATH 推出前缀，
  找到 `.conda-env/share/verilator/verilator-config.cmake`，不需要设 `VERILATOR_ROOT`。
  注意 `.conda-env/bin` 里有 conda 自带的 `make`，source 后 `make` 指向它（不影响构建）。
- **长任务放后台**：`setsid bash gpt2_demo.sh >/dev/null 2>&1 </dev/null &`，日志看 `gpt2_demo.log`，
  完成标记 `gpt2_demo.DONE` / `gpt2_demo.FAIL`。用 `pkill -f` 清理时用 **bracket 模式**（`'[g]pt2_demo.sh'`）避免误杀自身。
- **cycle-accurate 仿真很慢**：`demo_infer` 全重算（无 `--kv-cache`）约 O(seq²)；10 token 的耗时见下方验证结果，
  100 token 约几小时到几十小时。要跑长序列加 `--kv-cache`，或先用小 `--max-tokens`（如 10~20）验证。

## 已知问题（上游代码，非环境问题）

- **fuzz 用例 50 个里只有 2 个能编译**（case_39、case_48）：`gen_fuzz_onnx.py` 会生成 `Add` / `Sub`（加常量），
  但 `compile.py` 的 `GraphCompiler.compile` 分派表里没有 `Add` / `Sub` / `Mul`，打印 `WARNING: Unsupported op` 跳过后，
  下一个算子查不到输入描述符而 `KeyError`。Phase 3（`1dd692f`）和 Phase 4（`8216c22`）的 `compile.py` 都如此，
  所以 README 里“50 个 fuzz 全部通过”在当前提交上无法复现；`onnx_fuzz_sim` 报 `Fuzz: 2/50 passed`。
- **`models/resize_pad_test.onnx` 编译失败**：`Resize` 的 `roi` 是空 initializer，`quantize_int8` 对空数组求
  `np.max` 抛 `ValueError: zero-size array to reduction operation maximum`，因此 `onnx_resize_pad_sim` 没有数据可跑。

## 复现脚本（都在仓库根目录）

- `venv_setup.sh` — 用 `.conda-env` 的 Python 3.11 `uv venv` + PyPI 装依赖 + pytorch index 装 CPU torch。
- `env.sh` — PATH 加 `.conda-env/bin` + 激活 `.venv`。
- `build_sims.sh` — `source env.sh` + cmake 构建全部 target，日志 `sim_build.log`。
- `prep_sim_data.sh` — 编译全部 ONNX 测试模型、生成并编译 fuzz、生成 LLaMA 随机权重。
- `run_tests.sh` — 批量跑 sim（带 `--datadir`）并汇总 PASS/FAIL，日志 `test_results.log`。
- `gpt2_demo.sh` / `gpt2_demo100.sh` — 后台跑 10 / 100 token 的 GPT-2 demo，日志 `gpt2_demo*.log`。

## 验证结果（2026-10-28，本机）

### 工具链构建
- Verilator 5.050（conda-forge）+ CMake 3.28.3 + g++ 13.3.0：`sim/verilator/build` 下 **21 个 target 全部构建成功**，
  96 核约 81 s，无 `%Warning` 输出。

### LLM Mode 单元/集成测试（无需外部数据，cycle-accurate）
| target | 结果 | 耗时 |
|---|---|---|
| `npu_sim` | **PASS**（=== ALL TESTS PASSED ===）| <1 s |
| `engine_sim` | **PASS**（=== ALL ENGINE TESTS PASSED ===）| 38 s |
| `integration_sim` | **PASS**（INTEGRATION TEST: PASS）| <1 s |
| `gpt2_block_sim` | **PASS**（GPT-2 BLOCK TEST: PASS，23 GEMM / 4 head）| 156 s |
| `llama_block_sim --datadir llama_data` | **PASS**（LLAMA BLOCK TEST: PASS，随机权重）| 212 s |

### Graph Mode（ONNX）
| target | datadir | 结果 |
|---|---|---|
| `onnx_smoke_sim` | `graph` | **PASS**: All 8 bytes match golden（14 条图指令）|
| `onnx_cnn_smoke_sim` | `graph_cnn` | **PASS**: All 4 bytes |
| `onnx_reduce_sim` | `graph_reduce` | **PASS**: All 1 bytes |
| `onnx_math_sim` | `graph_math` | **PASS**: All 16 bytes |
| `onnx_gather_sim` | `graph_gather` | **PASS**: All 16 bytes |
| `onnx_slice_concat_sim` | `graph_slice_concat` | **PASS**: All 8 bytes |
| `onnx_batchnorm_pool_sim` | `graph_batchnorm_pool` | **PASS**: All 36 bytes |
| `onnx_stress_sim` | `graph_stress` | **PASS**: All 1 bytes（51 s）|
| `onnx_overlap_perf_sim` | `graph_overlap_perf` | **PASS**: All 32 bytes（20 s）|
| `onnx_fp16_smoke_sim` | `graph_fp16_smoke` | **PASS**: All 4 bytes |
| `mixed_precision_cnn_sim` | `graph_mixed_cnn` | **PASS**: All 64 bytes |
| `onnx_fuzz_sim` | `graph_fuzz` | **FAIL**：Fuzz: 2/50 passed（48 个编译失败，见“已知问题”）|
| `onnx_resize_pad_sim` | `graph_resize_pad` | 未运行（模型编译失败，见“已知问题”）|

### 端到端推理
- **GPT-2**：`bash gpt2_demo.sh`（即 `run_demo.sh --prompt "Hello" --max-tokens 10`）
  - HF 权重**不设 `HF_ENDPOINT`、经代理直连 huggingface.co 成功拉取** → 导出 45 tensors → INT8 量化 `weights.bin`（230976 B）
    → 构建并运行 NPU `demo_infer`。
  - 结果：**`GPT-2 DEMO: PASS`** — 10 个 step 的 NPU logits 对 C++ golden **逐位一致**（`max_err=0 BIT-EXACT`），
    共 **9,967,070 cycles**（与初版主机完全相同）。整个脚本 wall time 约 35 min，期间有其他仿真并行占用 CPU。
  - 该 demo 是**缩小版验证配置**（layers=4 / hidden=64 / ffn=256 / vocab=256），NPU 生成文本为 `Hello!!!!!!!!!!`
    （token 全 0）——toy 配置 + INT8 量化下的预期退化；Python 浮点 golden 生成 `HelloO(#"""""""`，
    这一差异被 tb 标注为 informational，通过标准是“NPU vs C++ golden 逐位一致”。
- **KV cache**：`./kv_cache_sim --datadir demo_data` → **`KV CACHE TEST: PASS`**（prefill S=5 与 decode S=6 的 block 输出和 logits
  均 `max_err=0`），约 579 s。
- **LLaMA（随机权重）**：`python python/tools/llama_gen_weights.py --outdir sim/verilator/build/llama_data`，
  `./llama_demo_infer --datadir llama_data` → **`LLAMA DEMO: PASS`**，4 个 step 全部 `logit_max_err=0 EXACT`，
  共 2,694,592 cycles，约 533 s。HF 版 MicroLlama / Mistral / Qwen2 权重本次未运行。

> 结论：**21 个 target 全部构建成功；除上游已知问题（fuzz 48/50、resize_pad 编译失败）外，LLM Mode 与 Graph Mode 两条主线的单元测试、ONNX 用例和 GPT-2 / LLaMA 端到端 demo 均在本机通过 golden 校验，环境可用。**