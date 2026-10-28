# CLAUDE.md

## 隐私规则（最高优先级）

- 不要在任何会进仓库的文件里暴露**主机名、用户名、邮箱、绝对路径**（`/home/…`、`/Users/…`、`/data…`、`~/具体目录`）。
- 路径一律写**相对仓库根目录**的形式（`sim/verilator/build`、`docs/SETUP_NOTES.md`）；需要指代仓库根时写“仓库根目录”或 `<repo>`；
  同级的姊妹仓库写 `../tiny-gpu`。命令默认在仓库根目录执行，进子目录用相对路径或 `( cd sim/verilator && … )`。
- 机器只用泛称描述（“本机”“初版主机”），硬件/系统版本可以写，主机名不写。
- 提交前自检，以下命令应无输出：

  ```sh
  git grep -nE '/home/|/Users/|/data00/|autoResearch|myDevbox' -- . ':!CLAUDE.md'
  ```

## 先读这些

- `ARCHITECTURE.md`：架构和使用文档（12 节，静态源码分析）。第 3 节组件表、第 9 节源码入口索引、第 12 节 README 与源码不一致之处。
  它在仓库外的笔记目录里有一个**硬链接**，覆盖更新时用 Python `open(path, "w")` 原地写，不要“写临时文件再 rename”或先删后建，否则硬链接会断。
- `docs/LEARNING_GUIDE.md`：从 tiny-gpu 到 tiny-NPU 的分阶段学习路线（S0–S11）。
- `docs/SETUP_NOTES.md`：本机工具链、网络、踩坑记录与验证结果。

## 分支与远程

- `main`：保持与原作者 `upstream`（harishsg993010/tiny-NPU）一致，**不在 main 上改**。
- `learning`：学习笔记、环境脚本、兼容性修复都放这里；`origin` 是个人 fork。
- 同步上游：`git fetch upstream && git switch main && git merge --ff-only upstream/main && git push origin main`，再 `git switch learning && git merge main`。

## 环境与常用命令（仓库根目录）

```sh
source env.sh              # PATH 加 .conda-env/bin（Verilator 5.050 / Python 3.11 / uv）+ 激活 .venv
bash build_sims.sh         # CMake 构建全部 21 个仿真 target，日志 sim_build.log
bash prep_sim_data.sh      # 编译 ONNX 测试模型、生成 fuzz、LLaMA 随机权重 → sim/verilator/build/*
bash run_tests.sh          # 批量回归（带 --datadir），结果 test_results.log
bash gpt2_demo.sh          # 10 token GPT-2 端到端 demo（约半小时），产出 demo_data
```

- 构建以 `sim/verilator/CMakeLists.txt` 为准；顶层 `Makefile` 的 `sim` / `lint` / `wave` 源文件清单过时，不要用（`make luts` / `make ucode` 可用）。
- `.conda-env/`、`.venv/` 不入库，重建方法见 `docs/SETUP_NOTES.md`；本机无 sudo，不要尝试 apt。
- 仿真是 cycle-accurate 的，GPT-2 / LLaMA demo 和 `kv_cache_sim` 要几分钟到几十分钟，放后台跑并看 `*.log` / `*.DONE`。
- 只读分析时不要运行仿真；需要运行结论时，先说明是实测还是引自 README。

## 已知问题（上游代码，非环境问题）

- `python/onnx_compiler/compile.py` 不支持 `Add` / `Sub` / `Mul`，fuzz 50 例只有 2 例能编译（`onnx_fuzz_sim` 报 2/50）。
- `models/resize_pad_test.onnx` 编译时 `quantize_int8` 对空的 `roi` 求 `np.max` 崩溃，`onnx_resize_pad_sim` 无数据。
- `rtl/top.sv` 只是控制平面外壳（Softmax/LayerNorm/GELU/Vec 为 done 桩，Graph Mode 未实例化）；完整数据通路在 `sim/verilator/*_top.sv`。
- 修这些问题时在 `learning` 分支上单独提交，便于以后挑出来给上游提 PR。
