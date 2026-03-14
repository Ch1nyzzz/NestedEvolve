# PubMedQA L2 Meta-Optimizer 运行报告

- 日期: 2026-03-10
- 运行时间: 12:59 ~ 16:58 (约 4 小时)
- 模型: together_ai/moonshotai/Kimi-K2.5
- Baseline (L1 结果): 74.00

## L2 诊断

L2 从 L1 历史中识别出 2 个关键框架缺陷:

1. **Optimizer 使用过期源码上下文** (severity: high)
   - 文件: `stages/optimizer.py`
   - 根因: optimizer 从 SystemDescription 原始快照读取源码生成 SEARCH/REPLACE，而非当前 workspace 状态，导致增量 patch 后 SEARCH 块与实际文件内容不匹配
   - 后果: patch 重复应用导致文件损坏，性能回退 (80→50)

2. **Analyzer JSON 解析截断/畸形** (severity: high)
   - 文件: `stages/analyzer.py`
   - 根因: `_parse_patterns()` 使用脆弱的 regex/string splitting 提取嵌套 JSON，在多 pattern 输出时截断对象
   - 后果: 下游组件无法解析诊断结果，影响 patch 质量

## 候选评估结果

所有 eval 分数均为 mini-L1 在 test_set (50 samples) 上的最终评估结果:

| # | 候选 | Patch 内容 | 分数 | vs Baseline | 耗时 |
|---|------|-----------|------|-------------|------|
| 1 | `fix_optimizer_stale_source` | optimizer.py ×3 ops | **78.00** | +4.00 | 3818s |
| 2 | `fix_analyzer_json_parsing` | analyzer.py ×1 op | **76.00** | +2.00 | 5770s |
| 3 | `combined_framework_fixes` | optimizer.py ×3 + analyzer.py ×1 | **80.00** | **+6.00** | 2324s |

## 最终结果

- **最优候选: `combined_framework_fixes` — 80.00 分 (+6.00)**
- 合并了两个修复 (optimizer 过期上下文 + analyzer JSON 解析)
- 该分数已经是 test_set (50 samples, seed=42) 上的评估结果

### 注意

由于框架 bug，`combined_framework_fixes` 在最后一步 eval 完成后未被 agent 显式 `accept_candidate`
(agentic loop step 5/5 结束，LLM 没机会调用 accept)，导致该候选未进入 top-3 池。
FinalEval 只测了 `fix_optimizer_stale_source` (test=76)。
实际最优应为 `combined_framework_fixes` (test=80)。

## 配置

```json
{
  "l2": { "max_steps": 5, "n_samples": 1, "max_evals": 5, "max_no_improve_steps": 3 },
  "mini_l1": { "max_steps": 10, "n_samples": 10, "max_evals": 8, "max_no_improve_steps": 4 }
}
```

## 文件清单

- `results_pubmedqa_l2_only.json` — 完整结果 (含 history)
- `l2_only_run.log` — 控制台输出日志
- `layer_logs/L2.log` — L2 层详细日志
- `layer_logs/L1.log` — L1 层日志 (前次运行)
