"""Workflow templates for noa_new."""

from __future__ import annotations

L1_WORKFLOW = """# L1 Optimizer Workflow

## Scope
你负责优化 `target_system/`，只能使用 `bash` 和 `execute_code` 两个工具。

## Directory Layout
- `target_system/`: 允许读写的目标系统
- `train_pool.pkl`: 训练池
- `val_set.pkl`: 验证集
- `test_set.pkl`: 测试集，优化过程中禁止使用
- `l1_config/`: L2 可调的 6 个槽位
- `trace.jsonl`: 每步结构化轨迹
- `metrics.json`: 当前指标
- `pareto.json`: 帕累托前沿

## Mandatory Loop
1. Observe: 抽样运行系统并记录失败模式
2. Diagnose: 明确 evidence / attribution / confidence
3. Patch: 只做有假设支撑的改动
4. Verify: 同时记录 accuracy 与辅助指标
5. Decide: commit 或 rollback，并记录原因

## Trace Rules
- 每个阶段都要 append 一条 JSON 到 `trace.jsonl`
- 每轮结束必须更新 `metrics.json` 和 `pareto.json`
- 连续 3 轮同一策略无提升时，必须记录 `strategy_change`

## Guardrails
- 不碰 `test_set.pkl`
- 改代码前先备份或复制候选目录
- 先做 cheap eval，再决定是否扩大验证
"""

L2_WORKFLOW = """# L2 Meta Optimizer Workflow

## Scope
你不直接优化 target system。你只审计 L1 的轨迹，并通过修改 `l1_config/` 里的 6 个槽位纠偏。

## Inputs
- `trace.jsonl`
- `metrics.json`
- `pareto.json`
- `l1_config/*.md`
- `parent_context.json`（如果存在）

## Six Slots
- `system_prompt.md`
- `loop_policy.md`
- `tool_policy.md`
- `budget_policy.md`
- `eval_schema.md`
- `playbook.md`

## Mandatory Loop
1. 读 L1 轨迹，定位模式性偏差
2. 只修改一个槽位的一条规则
3. 把 patch append 到 `l2_patches.jsonl`
4. 用 mini-L1 或等价验证脚本验证
5. 根据结果 keep 或 rollback

## Guardrails
- 每次只改一个槽位
- `playbook.md` 只能追加
- 每个 patch 必须有 rollback_condition
- 不直接改 `target_system/`
"""
