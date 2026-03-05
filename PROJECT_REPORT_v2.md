# NOA v2 架构报告：从脚本式循环到 Planner 驱动的自主优化

> **日期**: 2026-03-04
> **对比基线**: PROJECT_REPORT.md (2026-02-26)

---

## 一、架构演进总览

### 核心变化：从"固定循环"到"LLM 自主决策"

| 维度 | v1 (旧架构) | v2 (新架构) |
|------|------------|------------|
| **控制流** | 硬编码 I-O-A-O-E 顺序循环 | Planner Agent 每步自主决定下一个 action |
| **嵌套方式** | L2 是独立的 Orchestrator 外层循环 | `spawn_sublayer` 是 Planner 可选的一个 action，L1/L2 完全同构 |
| **Stage 执行** | 每个 stage 是单次 LLM 调用 | 三个核心 stage 升级为多轮 tool-calling（agentic） |
| **状态管理** | 散落在 engine 各处的局部变量 | 统一的 `PlannerState` + Redux 风格 `Reducer` |
| **资源控制** | `max_iterations` + 连续无改善计数 | 四维预算（步数/LLM 调用/eval 次数/spawn 次数） |
| **安全边界** | 工作空间文件隔离 | 新增 `LayerContext` 权限控制（读写 ACL + 路径穿越防护） |
| **评估策略** | 单阶段全量评估 | 三级级联（语法检查 → 烟雾测试 → 全量评估） |
| **诊断方式** | 整批 failure 一次性 LLM 分析 | 逐条并行增量诊断 + FailurePool 累积去重 |

---

## 二、新增核心模块：Planner（`noa/planner/`）

这是 v2 最大的结构性变化。旧架构中 `engine.py` 硬编码了 "Observe → Analyze → 对每个 pattern: Optimize → Evaluate" 的循环；新架构引入了一个 **LLM Planner** 作为元控制器。

### 2.1 工作原理

```
NOptimizer.run()
  ├── Initiate（一次性，不变）
  └── Planner Loop:
        ┌─────────────────────────────────────────────┐
        │  PlannerAgent.decide(state)                  │
        │    → 返回 PlannerDecision(action, params)    │
        │                                              │
        │  Guardrails.validate(state, decision)        │
        │    → 合法性校验 + 必要时重定向               │
        │                                              │
        │  ActionExecutor.run(action, params, state)   │
        │    → 执行对应 stage，返回 ActionResult       │
        │                                              │
        │  Reducer.apply_result(state, decision, result)│
        │    → 更新 PlannerState                       │
        │                                              │
        │  TraceWriter.write(record)                   │
        │    → JSONL 持久化                            │
        └─────────────────────────────────────────────┘
```

### 2.2 六个 Action

| Action | 说明 | 旧架构对应 |
|--------|------|-----------|
| `observe` | 收集/复用执行轨迹 | Observer 阶段 |
| `analyze` | 增量并行诊断 | Analyzer 阶段 |
| `propose_patch` | 生成 SEARCH/REPLACE 补丁 | Optimizer 阶段 |
| `evaluate_patch` | 三级级联沙箱验证 | Evaluator 阶段 |
| `spawn_sublayer` | 启动 L2 子优化器 | **旧架构中是 Orchestrator 的外层逻辑** |
| `stop` | 退出循环 | 硬编码的停止条件 |

**关键区别**: Planner 可以自由组合这些 action。例如：
- 可以连续 observe 两次（补充轨迹覆盖率）
- 可以 analyze 后发现信息不足，回到 observe
- 可以在任意时机决定 spawn L2

旧架构中这些决策都是硬编码的 if-else。

### 2.3 Guardrails — 硬约束兜底

Planner 虽然自主决策，但有 9 条硬约束防止非法操作：

- 没有轨迹时不能 analyze → 重定向到 observe
- 没有诊断时不能 propose_patch → 重定向到 analyze
- intermediate 覆盖率 < 95% 时不能 propose_patch → 重定向到 observe
- 没有候选补丁时不能 evaluate → 重定向到 propose_patch
- 有未评估的补丁时不能 stop → 强制 evaluate
- 连续无改善且 spawn 预算未耗尽 → 强制 spawn_sublayer
- 至少 3 次 eval 才可 stop（或预算 80% 耗尽）

### 2.4 PlannerState — 统一状态

```python
@dataclass
class PlannerState:
    layer: str                      # "l1" / "l2"
    budget: PlannerBudget           # 四维预算
    baseline_score: float
    current_score: float
    trajectories: list[Trajectory]
    diagnosis: Diagnosis | None
    candidate_patch: DeltaPatch | None
    last_eval: EvalResult | None
    no_improve_steps: int
    accepted_patches: int
    history: list[dict]             # 完整决策历史
    layer_context: LayerContext | None
    action_counts: dict[str, int]   # 各 action 执行次数
```

旧架构中这些状态散落在 `engine.py` 的局部变量和 `orchestrator.py` 的实例变量中，难以追踪和调试。

---

## 三、Stage 升级：从单次调用到 Agentic 多轮

### 3.1 通用 agentic_loop（`noa/stages/agentic.py`）

新增了一个**通用的 tool-calling 循环函数**，三个核心 stage（Observer、Analyzer、Optimizer）都复用它：

```python
def agentic_loop(
    messages, tools, tool_executor, model,
    max_tool_calls=5,        # =0 时退化为旧架构的单次调用
    parse_fn=None,           # 输出解析，成功则退出
    early_stop_fn=None,      # 提前终止
) -> str
```

### 3.2 Observer 的变化

| 旧架构 | 新架构 |
|--------|--------|
| 直接运行 N 条数据，收集轨迹 | LLM 多轮决策：先扫描已有轨迹文件 → 评估质量 → 决定复用/replay/fresh |
| 每次都全量 observe | 支持轨迹复用，节省资源 |
| 无中间输出概念 | 要求 intermediate 覆盖率 ≥ 95% |

`_ObservationProbe` 工具集：
- `discover_trajectory_files()` — 扫描已有轨迹
- `inspect_trajectory_file(path)` — 检查轨迹质量
- `run_sample_observation(n, seed)` — 运行新实验
- `replay_trajectories(questions)` — 补充 intermediate
- `finalize_observation(strategy, ...)` — 最终决策

### 3.3 Analyzer 的变化

| 旧架构 | 新架构 |
|--------|--------|
| 所有 failure 一次性送入 LLM | 逐条并行诊断（`ThreadPoolExecutor`） |
| 一次性输出所有 patterns | `FailurePool` 增量累积 + 相似度去重 |
| 无实验能力 | 有 `ComponentProbe` 时走 ReAct 实验循环 |

**FailurePool** 机制：
- 每条 failure 轨迹独立诊断，结果合并到 pool
- 相似度阈值 0.6 的字符串匹配自动合并
- LLM 可指定 `merge_to` 合并到已有 pattern
- `consolidate()` 最终去重，`top_n()` 按频次+严重度排序

### 3.4 Optimizer 的变化

| 旧架构 | 新架构 |
|--------|--------|
| 单次 LLM 调用生成 patch | 多轮 tool-calling，边写边验证 |
| 生成后才发现 SEARCH 不匹配 | 写之前可调 `verify_search_block()` 验证 |
| 无 dry-run | 可调 `dry_run_patch()` 内存中预览效果 |
| 无质量自检 | `_self_check_quality()` LLM 自评 0~1 分 |

`_OptimizerProbe` 工具集：
- `verify_search_block(file_path, search_text)` — 验证 SEARCH 是否匹配
- `read_source_file(file_path, start_line, end_line)` — 读源文件
- `dry_run_patch(file_path, search_text, replace_text)` — 内存 dry-run

---

## 四、嵌套机制重构：L1/L2 同构化

### 旧架构

```
Orchestrator（L2 专用逻辑）
  ├── Round N:
  │     ├── run_l1_subprocess()        # L1 子进程
  │     ├── _run_l2():                 # L2 硬编码在 Orchestrator 中
  │     │     ├── meta_analyze()       # L2 专用分析函数
  │     │     ├── optimize()           # 复用 L1 的 optimizer
  │     │     └── evaluate: run_l1_subprocess()  # 整个 L1 作评估
  │     └── snapshot
  └── 返回结果
```

- L2 是 Orchestrator 中的**专用代码路径**，有独立的 `meta_analyze()` 函数和 L2 专用 prompt
- L1 和 L2 的代码完全不同

### 新架构

```
Orchestrator（薄层编排）
  └── NOptimizer.run()（L1，同进程）
        └── Planner 决定 spawn_sublayer
              └── NOptimizer.run()（L2，同进程，同一个类）
                    └── 评估：mini-L1 subprocess
```

- **L1 和 L2 是同一个 `NOptimizer` 类的不同实例**，差异仅通过 `LayerContext` 注入
- `LayerContext` 控制：
  - `writable_root`: L1 写目标系统代码，L2 写 noa/ 框架代码
  - `readable_roots`: 各层可读范围
  - `max_depth` / `max_spawn_calls`: 递归深度限制
  - `parent_history`: L2 可看到 L1 的优化历史
- 删除了 L2 专用 prompt，统一使用 `{layer_context}` 占位符动态注入层级信息
- 理论上支持 L3+（通过 `max_depth` 配置）

### LayerContext 权限控制

```python
@dataclass
class LayerContext:
    layer_id: str           # "L1", "L2"
    level: int              # 1, 2, ...
    writable_root: str      # 唯一可写目录
    readable_roots: list[str]
    parent_history: list[dict]
    max_depth: int
    max_spawn_calls: int

    def check_write_permission(self, path) -> bool  # 防 ../ 穿越
    def check_read_permission(self, path) -> bool
    def can_spawn_sublayer(self) -> bool
```

---

## 五、评估策略升级：三级级联

### 旧架构

```
apply_diffs_in_memory() → write_to_temp_dir() → 全量评估 → accept/reject
```

### 新架构

```
Stage 1: 语法检查
  └── apply_diffs_in_memory() → 无修改（SEARCH 不匹配）→ 快速拒绝

Stage 2: 烟雾测试（3 样本）
  └── smoke_score < baseline × 0.8 → 快速拒绝
  └── 崩溃 → 拒绝

Stage 3: 全量评估（n_samples 样本）
  └── after_score > before_score → accept, commit_to_source()
  └── 否则 reject
```

**收益**: 大部分无效补丁在 Stage 1-2 就被过滤，节省全量评估的开销。

---

## 六、新增组件：ComponentProbe（`noa/tools/probe.py`）

旧架构中 Analyzer 只能"看"轨迹数据；新架构引入 `ComponentProbe`，允许 Analyzer **动态实验**：

```python
class ComponentProbe:
    def run_component(name, inputs) -> dict    # 运行单个组件
    def run_from(start_component, inputs) -> dict  # 从某组件开始运行后续管道
```

- 包装目标系统，暴露为 OpenAI function calling 工具
- Analyzer 可以在 ReAct 循环中隔离测试单个组件
- 30 秒超时保护

---

## 七、Trace 持久化（`noa/planner/trace.py`）

新增 `PlannerTraceWriter`，每一步 Planner 决策都写入 JSONL 文件：

```json
{
  "step": 3,
  "action": "analyze",
  "params": {},
  "success": true,
  "score_before": 38.88,
  "score_after": 38.88,
  "budget_used": {"steps": 3, "llm_calls": 5, "evals": 0},
  "timestamp": "2026-03-04T10:23:45"
}
```

旧架构没有结构化的决策日志，调试时只能从 print 输出反推。

---

## 八、配置统一化

### 旧架构

```bash
python scripts/run_l1.py --n=50 --iterations=5 ...       # L1 专用入口
python scripts/run_nested.py --n=50 --l1-iterations=5 ... # 嵌套专用入口
```

### 新架构

```bash
python scripts/run_nested.py --config configs/hotpotqa_l1_only.json   # L1-only
python scripts/run_nested.py --config configs/hotpotqa_nested.json    # 嵌套
```

所有参数统一在 JSON 配置中，通过 `max_spawn_calls=0` 禁用嵌套即退化为 L1-only。

配置示例（`hotpotqa_nested.json`）：
```json
{
  "max_depth": 3,
  "max_spawn_calls": 2,
  "budget": {
    "max_steps": 20,
    "max_llm_calls": 50,
    "max_evals": 8,
    "max_no_improve_steps": 3
  },
  "model": "claude-haiku-4-5-20251001"
}
```

---

## 九、目录结构对比

### 新增

```
noa/planner/                 # 全新 — Planner 驱动控制流
├── agent.py                 # PlannerAgent（LLM 决策 + tool-calling）
├── executors.py             # ActionExecutor（六个 action 的执行器）
├── guardrails.py            # 9 条硬约束校验
├── protocol.py              # PlannerState, PlannerBudget, PlannerDecision
├── prompts.py               # Planner 专用 prompt
├── reducer.py               # Redux 风格状态更新
└── trace.py                 # 步骤轨迹 JSONL 持久化

noa/stages/agentic.py        # 全新 — 通用 tool-calling 循环
noa/tools/probe.py           # 全新 — ComponentProbe 单组件实验

configs/hotpotqa_nested.json  # 全新 — 嵌套优化配置
configs/hotpotqa_l1_only.json # 全新 — L1-only 配置
```

### 删除

```
scripts/run_l1.py             # 合并到 run_nested.py
scripts/run_baseline.py       # 合并到 run_nested.py
results*.json                 # 清理实验结果
.beads/                       # 移除 beads issue tracking
```

### 重大修改

```
noa/engine.py                 # 从硬编码循环重构为 Planner 驱动
noa/orchestrator.py           # 从 L2 专用编排简化为薄层 workspace 管理
noa/stages/analyzer.py        # 增量并行诊断 + FailurePool
noa/stages/evaluator.py       # 三级级联评估
noa/stages/observer.py        # Agentic 轨迹管理
noa/stages/optimizer.py       # Agentic patch 生成 + 工具验证
noa/core/protocol.py          # 新增 LayerContext, EvalResult.artifacts 等
noa/diff_utils.py             # 新增路径安全检查 + LayerContext 写权限校验
```

---

## 十、总结：关键改进

| # | 改进 | 旧痛点 | 新方案 |
|---|------|--------|--------|
| 1 | **Planner 自主决策** | 固定循环不灵活，无法根据实际情况调整策略 | LLM 根据当前状态动态选择下一步 |
| 2 | **L1/L2 同构** | L2 是独立的代码路径，维护成本高 | 同一个 NOptimizer 类，LayerContext 注入差异 |
| 3 | **Agentic Stage** | 单次 LLM 调用，信息不足时无法补充 | 多轮 tool-calling，可实验、验证、迭代 |
| 4 | **增量并行诊断** | 所有 failure 一次性分析，token 爆炸 | 逐条并行 + FailurePool 累积去重 |
| 5 | **三级级联评估** | 每个 patch 都做全量评估，浪费资源 | 语法→烟雾→全量，快速过滤无效 patch |
| 6 | **四维预算** | 仅靠 max_iterations 控制 | 步数/LLM/eval/spawn 独立限制 |
| 7 | **Guardrails** | 依赖硬编码 if-else | 声明式约束 + 自动重定向 |
| 8 | **ComponentProbe** | Analyzer 只能看静态轨迹 | 可动态运行单组件实验 |
| 9 | **Trace 持久化** | 无结构化日志 | 每步决策写 JSONL |
| 10 | **权限控制** | 仅文件系统隔离 | LayerContext ACL + 路径穿越防护 |
