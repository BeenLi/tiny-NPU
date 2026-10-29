> 本文是 tiny-NPU 的学习文档，随仓库保存；图在 `figs/`，来源见 [`figs/README.md`](figs/README.md)。文中的 `文件:行号` 都是相对本仓库的链接，点击可直接跳到对应代码行。

# 01 微码 ISA 与控制平面总览

> 学习计划第 01 步，见 [学习计划](README.md)。标 ✅ 的是直接从代码或配置读出的事实，标 ❓ 的是待核实，🆕 是 Sze 书（2020）和 tiny-gpu 里都没有的东西。框架已搭，内容待写。

---

## 0. 一张图：微码 ISA 与控制平面的全貌

（待写）

---

## 1. 细节分篇

| 篇 | 内容 | 主要结论 |
| --- | --- | --- |
| （开始这一步时细化） | | |

---

## 2. 待核实清单

- ❓ scoreboard 的引擎槽数：`npu_pkg.sv` 定义 `NUM_ENGINES = 6`，而 `gpt2_block_top.sv` 实例化时传 `NUM_ENGINES(8)`，第 6、7 号槽给 RMSNorm / RoPE？线索：[`npu_pkg.sv:60`](../../rtl/pkg/npu_pkg.sv#L60)、[`gpt2_block_top.sv:145`](../../sim/verilator/gpt2_block_top.sv#L145)。
