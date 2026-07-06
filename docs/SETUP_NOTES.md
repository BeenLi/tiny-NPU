# tiny-NPU 环境搭建记录（SETUP NOTES）

远端主机 **<host>**：Debian 10 buster（glibc 2.28，g++ 8.3.0，64 核 / 251 GB RAM）。
仓库路径 `<repo>`（= `<repo>`）。
搭建日期：2026-07-05。

## TL;DR 怎么用

```sh
ssh <host>
cd <repo> && source env.sh     # venv + Verilator 5.050 + CMake 3.28.3 + HF 镜像

# 构建全部 21 个仿真 target（唯一权威路径，见下方"坑"一节）
cd sim/verilator && mkdir -p build && cd build
cmake .. && cmake --build . -j"$(nproc)"

./npu_sim            # LLM 控制平面 smoke（=== ALL TESTS PASSED ===）
./engine_sim         # 引擎计算测试
./integration_sim    # 单 head attention
./gpt2_block_sim     # 完整 transformer block

# 端到端 GPT-2 推理（LLM Mode，需 venv + HF 镜像，env.sh 已配）
cd <repo>/sim/verilator && bash run_demo.sh --prompt "Hello" --max-tokens 10

# Graph Mode（ONNX）：先编译模型到 datadir，再跑 sim
cd <repo>
python python/onnx_compiler/compile.py --model models/mlp_32_16_8.onnx --outdir sim/verilator/build/graph
cd sim/verilator/build && ./onnx_smoke_sim --datadir graph
```

## 安装的工具链

| 组件 | 版本 | 来源 / 位置 | 备注 |
|---|---|---|---|
| Verilator | **5.050** | 源码构建（`stable` 分支）→ `~/.local` | apt 的 **4.010** 太老，不支持本仓库的 package/struct/`--trace-structs` |
| CMake | **3.28.3** | 官方 Linux 二进制 → `~/.local/cmake-3.28.3-linux-x86_64` | apt 的 **3.13.4** 太老；Verilator 5.050 的 `verilator-config.cmake` 要求 **CMake ≥ 3.19** |
| Python venv | **3.11.15** | `uv venv` → `<repo>/.venv` | 依赖走 Aliyun 镜像，torch 走 pytorch CPU index |
| 构建依赖 | — | 系统已装 flex/bison/autoconf/help2man/ccache/g++ | 构建 Verilator 所需，无需额外装 |

### Python 依赖（venv，`pip freeze` 关键项）
```
numpy 2.4.6      scipy 1.17.1     onnx 1.22.0
torch 2.12.1+cpu   transformers 5.13.0
huggingface_hub 1.22.0   safetensors 0.8.0   socksio 1.0.0
# python 3.11.15；torch.cuda.is_available() == False（确认为 CPU 版）
# socksio：交互式 shell（~/.zshrc）会 export all_proxy=socks5h://127.0.0.1:5555，
#   httpx 走 SOCKS 代理需要它，否则 huggingface 下载报 ImportError。
```

## 关键网络约束（本机在受限网络内）

- **huggingface.co 被墙** → `env.sh` 设 `HF_ENDPOINT=https://hf-mirror.com`，GPT-2/LLaMA 权重从镜像拉（已验证可用）。
- **pypi.org 直连极慢**（~0.5 MB/s）→ pip/uv 用 **Aliyun 镜像** `https://mirrors.aliyun.com/pypi/simple`（实测 ~5 MB/s）。
  - 清华 tuna 镜像对部分 wheel 返回 **403**，已弃用；Aliyun 稳定。
- **torch CPU**：用 `--index-url https://download.pytorch.org/whl/cpu`（**不要**再加 `--extra-index-url`，
  否则解析器会拉成 CUDA 版、带一堆 `nvidia-*` 包）。torch 的纯 python 依赖（sympy/networkx/...）先用 Aliyun 预装，
  避免从 pytorch index 慢速拉取（那一步会卡住）。
- **github.com 仅经代理** `http://127.0.0.1:5555`（`~/.gitconfig` 已配 github.com）；下载 GitHub release 资产
  （如 CMake 二进制）用 `https_proxy=http://127.0.0.1:5555 curl ...`。

## 对仓库做的改动（为兼容 Verilator 5.050 / g++ 8.3）

1. **`sim/verilator/CMakeLists.txt`**：`VERILATOR_COMMON_ARGS` 追加 `-Wno-SYNCASYNCNET` 和 `-Wno-fatal`。
   - Verilator 5.050 **默认把 warning 当致命错误**，比仓库预期的旧版严格；`axi_dma_rd.sv` 的 `rst_n` 触发
     `SYNCASYNCNET`（复位信号同时被当同步/异步用，良性 lint）。`-Wno-fatal` 让 warning 只打印不中断。
     原文件备份为 `CMakeLists.txt.bak`。
2. **5 个 C++ 测试台**（`tb_gpt2_block.cpp`/`tb_demo_infer.cpp`/`tb_kv_cache_sim.cpp`/`tb_llama_block.cpp`/`tb_llama_demo_infer.cpp`）：
   `memset(dut->uc_wr_data, ...)` → `memset(&dut->uc_wr_data, ...)`。
   - Verilator 5.050 把 128-bit 端口表示成 `VlWide<4>` 对象（不再隐式退化为指针），需取地址。

## 坑 / 注意事项

- **顶层 `Makefile` 的 `lint` / `sim` target 已过时，别用**：其 SV 源清单缺 `rtl/graph/fp16_utils.sv`（定义
  `fp16_utils_pkg`，被 softmax/layernorm/gemm 引用）、`mac_fp16.sv`、整个 `rtl/graph/*`，`make lint` 会报
  `Import package not found: 'fp16_utils_pkg'`。**一律以 CMake 构建为准**（`sim/verilator/CMakeLists.txt` 才是
  完整、权威的源文件清单，能构建全部 21 个 target）。`make luts`/`make ucode`（python 工具）仍可用。
- **LUT 无需运行时文件**：RTL 里没有 `$readmemh`，LUT 内容编译进 `.sv`，所以构建/运行仿真**不需要先 `make luts`、
  也不需要 venv**；venv 只在跑权重导出 demo（GPT-2/LLaMA）和 ONNX 编译器时才需要。
- **ONNX 图 sim 是数据驱动**：每个 `onnx_*_sim` 用 `--datadir <dir>` 读 `program.bin/tdesc.bin/ddr_image.bin`。
  这些要先用 `python python/onnx_compiler/compile.py --model models/<x>.onnx --outdir <dir>` 生成。
  各 target 对应的 datadir 名见 `sim/verilator/regression_all.sh`。
- **`sim/verilator/*.sh` 缺执行位**：用 `bash <script>` 调用（如 `bash run_demo.sh ...`）。
- **detached 远端任务**：长命令用 `setsid bash script >log 2>&1 </dev/null &`，否则 SSH 前台连接一断就被 SIGHUP。
  用 `pkill -f` 清理时务必用 **bracket 模式**（`'[v]env_setup.sh'`）避免误杀 ssh 自身那条命令。
- **`ImportError: Using SOCKS proxy, but the 'socksio' package is not installed`**：交互式 shell 的 `~/.zshrc`
  会 export `all_proxy=socks5h://127.0.0.1:5555`，venv 里的 httpx（huggingface 下载用）走 SOCKS 需 `socksio`
  （已加入 `venv_setup.sh`）。或运行前 `unset all_proxy`，直连 hf-mirror 也可。
- **cycle-accurate 仿真很慢**：`demo_infer` 全重算（无 `--kv-cache`）约 O(seq²)——10 token ≈ 9 min / ~10M cycles，
  100 token 约几小时到几十小时。要跑长序列加 `--kv-cache`，或先用小 `--max-tokens`（如 10~20）验证。

## 复现脚本（留档在 `<workspace>/` 与 `<repo>/`）

- `build_verilator.sh` — clone `stable` + `./configure --prefix=~/.local` + `make -j` + install。
- `install_cmake.sh` — 经代理下载 CMake 3.28.3 官方二进制解包到 `~/.local`。
- `tiny-NPU/venv_setup.sh` — `uv venv` + Aliyun 装依赖 + pytorch index 装 CPU torch。
- `tiny-NPU/build_sims.sh` — `source env.sh` + cmake 构建全部 target。
- `tiny-NPU/run_tests.sh` — 批量跑 sim 并汇总 PASS/FAIL。

## 验证结果（2026-07-05）

### 工具链构建
- Verilator 5.050 + CMake 3.28.3：`sim/verilator/build` 下 **21 个 target 全部构建成功**。

### LLM Mode 单元/集成测试（无需外部数据，cycle-accurate）
| target | 结果 |
|---|---|
| `npu_sim` | **PASS**（=== ALL TESTS PASSED ===）|
| `engine_sim` | **PASS**（=== ALL ENGINE TESTS PASSED ===）|
| `integration_sim` | **PASS**（INTEGRATION TEST: PASS）|
| `gpt2_block_sim` | **PASS**（GPT-2 BLOCK TEST: PASS，23 GEMM / 4 head）|

### Graph Mode（ONNX）冒烟
- `compile.py --model models/mlp_32_16_8.onnx --outdir build/graph` → 14 条图指令 / 10 描述符；
  `./onnx_smoke_sim --datadir graph` → **PASS: All 8 bytes match golden!**（graph execution done @ cycle 2517）。

### 端到端 GPT-2 推理（LLM Mode，真实 HF 权重，验证配置 layers=4 hidden=64）
- `bash run_demo.sh --prompt "Hello" --max-tokens 10`
- HF 权重经 **hf-mirror.com 成功拉取** → 导出 45 tensors → INT8 量化 `weights.bin`（230976 B）→ 构建并运行 NPU `demo_infer`。
- 结果：**`GPT-2 DEMO: PASS`** — NPU logits 对内置 C++ golden **逐位一致**（`max_err=0 BIT-EXACT`），共 **9,967,070 cycles**。
  - 该 demo 是**缩小版验证配置**（layers=4 / hidden=64 / ffn=256 / vocab=256），生成文本为 `Hello!!!!!!!!!!`（token 全 0）——
    这是 toy 配置 + INT8 量化下的预期退化。demo 的通过标准是"NPU vs 设计 golden 逐位一致"（对 Python 浮点 golden 的
    token 差异被标注为 *informational*，非失败）。

> 结论：**LLM Mode 与 Graph Mode 两条主线均已在本机跑通并通过 golden 校验，环境可用。**
