我同意你的判断：**现在最缺的不是再加一层 prompt，而是把每一层都做成一个能持续观察、调用工具、回滚、再试的 optimizer agent。**
更准确地说，你们缺的是 **persistent tool loop**，不是“多几个工具名”这么简单。Codex/Claude Code 之所以看起来会“自己觉得有问题然后继续查、继续修”，并不只是模型更强，而是它们运行在一个持久的 agent harness 上：Codex 的 Agent mode 默认就能读文件、跑命令、写改动；OpenAI 的 tool calling 也是明确的多轮循环；GPT-5 官方还专门建议在 agentic tool flow 里用 Responses API，因为 reasoning 会在工具调用之间持久化。 ([OpenAI开发者][1])

所以如果你的目标是：

* **L1** 优化任意 target system
* **L2** 同构地优化 L1
* 且不对某个 target（比如当前 RAG）做硬编码

那我会这样设计。

---

## 1. “L2 与 L1 同构”应该怎么理解

我建议把“同构”定义成：

> **同一个 Optimizer Kernel + 同一套动作本体 + 同一套记忆/评估协议**
> 只是 optimizee 不同。

也就是说，不是简单地把 L1 的 prompt 复制一份给 L2，而是两层都跑同一个内核：

```python
OptimizerKernel(
    role_card,
    optimizee_adapter,
    toolset,
    memory_store,
    evaluator,
    sandbox_manager,
)
```

区别只在于：

* **L1 的 optimizee** = target system
* **L2 的 optimizee** = L1 本身（代码、策略、skills、轨迹、评估器）

这件事一旦做对，L2 就不需要“玄学地意识到自己是 L2”，而是**环境明确告诉它**：

* 我是谁
* 我能看什么
* 我能改什么
* 我的 reward 是什么
* 我不能碰什么

所以真正的“自我意识”不是靠文艺化的 soul，而是靠 **machine-readable layer context + 限定过的工具命名空间 + 明确的 editable surface**。

---

## 2. 先不要想“支持任何 target”，先想“统一协议”

如果你想支持任意 target system，关键不是零适配，而是做一个**窄腰协议**。
任何 target 只要实现这个协议，L1 就能优化它；L2 再去优化 L1。

我会要求每个 optimizee 暴露一个极薄的 adapter：

```python
class OptimizeeAdapter(Protocol):
    def describe(self) -> dict: ...
    def list_entrypoints(self) -> dict: ...
    def smoke(self, seed: int | None = None) -> dict: ...
    def eval(self, mode: str, slice: str | None = None, seed: int | None = None) -> dict: ...
    def reproduce(self, case_id: str) -> dict: ...
    def collect_artifacts(self, run_id: str) -> dict: ...
```

这层协议只解决 4 件事：

1. **怎么读这个系统**
2. **怎么跑它**
3. **怎么评估它**
4. **怎么拿回 artifact / trace**

这样就不需要对 RAG、Agent、Compiler、Web service 分别写“智能逻辑”，只需要各自提供 adapter。

---

## 3. 每一层都应该是“一个持续运行的 optimizer”，而不是一次性 stage pipeline

你现在如果还是 `initiator -> observer -> analyzer -> optimizer -> evaluator` 这种一次走完的流水线，我会建议保留这些模块名，但把它们从 **stage** 变成 **skills / modes**。

因为多轮工具使用的关键不是模块数量，而是：

* 有**一个持续 thread**
* 有**外部世界状态**
* 有**每步 tool result 回流**
* 有**失败后继续下一步**的控制器

这和 OpenAI 的 tool-calling 流程是同一类东西：模型先选工具，应用执行，再把结果回传给模型继续推理；Codex 作为 MCP server 甚至专门提供了 `codex` 和 `codex-reply` 两个工具来维持同一 thread 的连续会话。 ([OpenAI开发者][2])

我会把每层都做成下面这个状态机：

```text
DISCOVER -> LOCALIZE -> EXPERIMENT -> VERIFY -> COMMIT -> REFLECT
                ^            |             |
                |            v             v
                +------ RETRY / ROLLBACK --+
```

### 这 6 个状态具体做什么

**DISCOVER**
先看 optimizee 描述、现有失败样本、历史 episode、仓库结构，不急着 patch。

**LOCALIZE**
形成 2~5 个可验证 hypothesis，不是写一段空泛 diagnosis。

**EXPERIMENT**
围绕 hypothesis 读更多文件、跑 reproducer、产生多个 patch candidate。

**VERIFY**
先走 cheap checks，再走 expensive eval。

**COMMIT**
接受最优 candidate，或者全部 rollback。

**REFLECT**
把这次 episode 结构化写回 memory，供后续 L1/L2 复盘。

---

## 4. L1 和 L2 分别该看什么、改什么

### L1：优化 target system

L1 的输入应该是：

* target repo / workspace
* target adapter
* 当前 benchmark / failure cases
* 历史 patch / 历史 episode
* target 的可运行命令与 artifact 位置

L1 能观察：

* 文件系统
* 源代码
* 执行日志
* benchmark 结果
* reproducer 输出
* 以前尝试过的 patch

L1 能修改：

* target repo
* target 配置
* target 测试 / harness（如果 policy 允许）

L1 的 reward：

* target 的评估提升
* 失败样本修复率
* no-op 减少
* 回归减少
* 成本控制

---

### L2：优化 L1

L2 的输入应该是：

* L1 的代码、prompts、skills、policies、budgets
* 一批 L1 历史 episode
* L1 的工具使用轨迹
* L1 的候选 patch 排序与 reject 原因
* L1 在 replay bench / live bench 上的表现

L2 能观察：

* L1 的 role/skill/policy 文件
* L1 的完整 episode 日志
* L1 的工具调用统计
* L1 的 patch success/failure taxonomy
* L1 的 replay 结果
* L1 的 live benchmark 结果

L2 能修改：

* **只修改 L1**
* 优先改：

  * `roles/L1.md`
  * `skills/*`
  * `policies/*.yaml`
  * `budgets.yaml`
  * `controller.py`
  * `candidate_ranking.py`
  * `evaluator.py`
* 一开始不要允许它改：

  * L2 自己
  * root contract
  * target system

L2 的 reward 不应该只看“最终 target 分数”
而应该看：

```text
meta_score =
  α * replay_improvement
+ β * live_generalization
+ γ * fewer_no_op_patches
+ δ * better_tool_usage
- ε * compute_cost
```

这一步非常关键。
否则 L2 看到的就只有一个跨两层传播的稀疏终局 reward，几乎学不到东西。

---

## 5. 你真正需要哪些“一等工具动作”

我建议把工具分成 **必需** 和 **增强** 两组。

### 必需工具：没有这些，就不会有真正 agentic 行为

#### A. 代码与仓库观察

* `list_tree(path, depth)`
* `search_text(query, glob)`
* `read_file(path, start, end)`
* `read_symbol(name)`
* `show_references(symbol)`
* `read_diff(base, head)`

#### B. 执行

* `run_command(cmd, cwd, timeout, env_profile)`
* `run_smoke(seed?)`
* `run_reproduce(case_id)`
* `run_eval(mode=cheap|full, slice?, seed?)`
* `inspect_artifacts(pattern)`

#### C. 编辑

* `apply_patch(ops)`
* `create_file`
* `delete_file`
* `snapshot_workspace(label)`
* `restore_snapshot(label)`
* `checkpoint_candidate(label)`

这里我会强烈建议用**结构化 patch**，不要把编辑退化成“输出一段建议文本”。OpenAI 官方的 `apply_patch` 就是按 create / update / delete 的结构化 diff 来设计的，目的正是支持多步、可反馈的代码工作流。 ([OpenAI开发者][3])

#### D. 记忆与轨迹

* `append_episode_event(event)`
* `read_episode(episode_id)`
* `query_episode_archive(filter)`
* `compare_runs(run_a, run_b)`

#### E. 资源与控制

* `budget_status()`
* `stop_with_summary()`

---

### 增强工具：第二阶段再加

* `spawn_subagent(role, task)`
* `rank_patch_candidates()`
* `static_analysis()`
* `dependency_map()`
* `semantic_code_search()`

---

## 6. 真正重要的不是“给 shell”，而是“给 typed tools + structured outputs”

我会强调这一点：

> **原始 shell 很重要，但它只能当 escape hatch，不能当主接口。**

因为如果所有东西都走 shell：

* 输出太噪
* 难以做 hindsight 学习
* L2 很难学到“L1 到底做对了什么”
* 很多失败只是命令写坏了，不是 optimizer 逻辑错了

所以主路径应该是 typed tools，shell 只用于兜底。
比如 `run_eval` 返回的不该是一坨 stdout，而应该是：

```json
{
  "ok": true,
  "metric_before": 0.412,
  "metric_after": 0.447,
  "delta": 0.035,
  "artifacts": ["runs/123/log.txt", "runs/123/preds.json"],
  "stderr_tail": "",
  "flaky": false
}
```

L2 才能吃这种信号。

---

## 7. sandbox 里“反复重试”应该怎么做

我会做 **分层 sandbox**：

### 外层：L2 sandbox

L2 在这里编辑 L1 的代码 / 策略，并启动多个 L1 评估任务。

### 内层：L1 sandbox

每个 L1 任务都在自己的 sandbox 里运行，去读 target、改 target、跑 target。

### 最内层：candidate snapshot

L1 针对某个 hypothesis 产生多个 patch 时，每个 patch candidate 都从**干净快照**起跑，避免互相污染。

结构大概像这样：

```text
L2 workspace
  └── candidate L1 patch A
        └── run L1 on target task 1
              └── target candidate patch 1
              └── target candidate patch 2
        └── run L1 on target task 2
  └── candidate L1 patch B
        └── ...
```

### 每个 attempt 的标准流程

1. 从 baseline snapshot 克隆
2. 读取上下文
3. 形成 hypothesis
4. 生成多个 patch candidate
5. apply patch
6. static / syntax / smoke
7. cheap eval
8. promising 才跑 full eval
9. 记录结果
10. accept 或 rollback

### 重要原则

* **每个 candidate 都从干净快照开始**
* **cheap → full 分级评估**
* **paired seeds / fixed slices**
* **明确区分 no-op / regression / flaky / crash**
* **所有工具输出都要落到 episode log**

Claude Code 官方也明确把权限 allowlist 和 OS-level sandbox 作为减少打断、但保持安全边界的核心机制；只有在隔离得很清楚的 sandbox 里，agent 才能少打断地连续执行。 ([Claude API Docs][4])

---

## 8. AGENTS.md / soul.md 能不能帮你实现 “我是 L2，我只优化 L1”

可以，但我建议你把它们分成三层，不要混在一起。

Codex 官方的 AGENTS.md 本来就是分层读取、近处覆盖远处；skills 也是按需展开完整说明。这套机制非常适合你做“稳定契约 + 角色卡 + 动态运行态”的三层设计。 ([OpenAI开发者][5])

### 我会这样拆

#### 1）`ROOT_CONTRACT.md`（人工写，永不自改）

定义系统根约束：

* 安全边界
* 评估规则
* 可编辑范围
* 必须记录哪些轨迹
* 接受 patch 的最低条件
* 预算约束

#### 2）`roles/L1.md`、`roles/L2.md`（稳定角色卡）

定义层身份：

* 你是谁
* 你的 optimizee 是谁
* 你能看什么
* 你能改什么
* 成功标准是什么
* 什么时候必须 rollback
* 不允许做什么

#### 3）`runs/<run_id>/soul.md`（运行时自我模型，可生成可更新）

这是你说的 soul，我觉得可以有，而且很有价值。
但它应该是**运行态摘要**，不是根指令。

比如：

```md
# identity
self_layer: L2
optimizee: L1
mission: Improve L1's ability to optimize arbitrary optimizees

# editable surface
- roles/L1.md
- skills/*
- policies/*
- controller.py
- evaluator.py

# current benchmark
- replay bench: 12 episodes
- live bench: 3 optimizees

# active hypotheses
1. L1 patches too early after shallow localization
2. L1 overuses shell, underuses structured eval
3. L1 candidate diversity too low

# failed attempts so far
- patch A: improved replay, regressed live
- patch B: no-op
- patch C: caused tool loop stall

# next best action
- inspect top 5 episodes where L1 stopped before reading evaluator artifacts
```

### 关键提醒

**不要让 soul.md 变成唯一真相。**
真正的权威状态应该是：

* `world.json` / `state.json`
* `episode.jsonl`
* `artifacts/`

`soul.md` 只是给模型的 compact self-model。

---

## 9. 让 L2 真正“像 L2”的关键，不是 prompt，而是工具命名空间

这一点很容易被忽略，但我觉得非常重要。

你如果希望 L2 真正知道“我优化的是 L1”，那工具就不要还是一套模糊的：

* `read_file`
* `run_eval`
* `apply_patch`

而应该带上 optimizee 语义：

### L1 的工具

* `target.read_file`
* `target.search`
* `target.run_reproduce`
* `target.run_eval`
* `target.apply_patch`
* `target.rollback`

### L2 的工具

* `optimizer.read_file`
* `optimizer.search_episodes`
* `optimizer.run_replay`
* `optimizer.run_live_bench`
* `optimizer.apply_patch`
* `optimizer.rollback`

注意这里的动词是同构的，**对象绑定不同**。
这就是我说的“same ontology, different binding”。

这样 L2 并不是“猜自己是谁”，而是被环境精确定义成了 L2。

---

## 10. 你想做的“嵌套 diagnose agent scaling”，真正该诊断什么

如果 L2 只是看 L1 的最终 patch 成败，它没法 scaling。
L2 真正该诊断的是 **L1 的决策错误类型**。

我会把 L1 的失败切成这些 hindsight label：

* `missed_observation`
* `wrong_localization`
* `premature_patching`
* `low_candidate_diversity`
* `bad_tool_choice`
* `stopped_too_early`
* `overfit_local_case`
* `ignored_artifact`
* `evaluator_misread`
* `flaky_accept`

L2 的任务不是说“L1 这次没修好”，而是说：

* **它为什么没修好**
* **在第几步本来应该做什么工具动作**
* **哪种 policy change 最可能修复这类错误**

只有这样，“嵌套 diagnose”才会产生 scaling，而不是只多一层评论。

---

## 11. 我会怎么改你当前的工程形态

如果你现在已经有：

* `initiator`
* `observer`
* `analyzer`
* `optimizer`
* `evaluator`

我不会先推翻重写，而是这样迁移：

### 第一步

把这五个模块从“固定 stage”改成“同一 persistent thread 下的 skill”。

也就是：

* `observer` 变成 `observe/SKILL.md`
* `analyzer` 变成 `localize/SKILL.md`
* `optimizer` 变成 `experiment/SKILL.md`
* `evaluator` 变成 `verify/SKILL.md`
* 再补一个 `reflect/SKILL.md`

这样很贴合 Codex 的 skills 机制：先加载 metadata，需要时再展开完整 SKILL.md，能省上下文，也更容易被 L2 修改。 ([OpenAI开发者][6])

### 第二步

加 4 个底层模块：

* `optimizee_protocol.py`
* `sandbox_manager.py`
* `trajectory_store.py`
* `run_state.py`

### 第三步

加两类基准：

* `replay_bench/`：历史 episode 回放
* `live_bench/`：真实 optimizee 评估

### 第四步

先只允许 L2 改 L1 的：

* roles
* skills
* policies
* budgets
* ranking/evaluator/controller

不要一开始就给 L2 任意改 L1 全部代码。

---

## 12. 最后直接回答你那个判断

### 你的判断对了一半

**是的，现在主要问题就是缺少多轮工具。**

但更准确的版本是：

> 你们缺少的不是“工具数量”，而是
> **持续线程 + typed tools + snapshot/rollback + 结构化轨迹 + 分层评估 + 明确的 optimizee 身份**。

### `agent.md / soul.md` 值得做

但它们的角色应该是：

* `agent/role card`：定义层身份和权限
* `soul.md`：保存运行时自我模型与当前假设
* **不是**替代 controller、memory、sandbox、tool loop

如果 controller 仍然是一次性 I-O-A-O-E，哪怕写再漂亮的 soul，也不会突然变成 Claude Code / Codex 那种“自己会继续动手”的 agent。

---

一句话收束：

**你们的下一步，不是“再加一个更聪明的 diagnose prompt”，而是把 L1/L2 都做成同一个 persistent optimizer kernel，并通过 optimizee adapter、工具命名空间、分层 sandbox、结构化轨迹，让 L2 真的只优化 L1。**

如果你愿意，我下一条可以直接继续往下走，给你一版更落地的东西：

1. `OptimizeeAdapter` / `SandboxManager` / `EpisodeSchema` 的接口草图
2. `roles/L1.md`、`roles/L2.md`、`soul.md` 的具体模板
3. L1/L2 controller 的伪代码流程图
4. 你现有 `observer/analyzer/optimizer/evaluator` 如何迁移成 skills

[1]: https://developers.openai.com/codex/quickstart/ "Quickstart"
[2]: https://developers.openai.com/api/docs/guides/function-calling/ "Function calling | OpenAI API"
[3]: https://developers.openai.com/api/docs/guides/tools-apply-patch/ "Apply patch | OpenAI API"
[4]: https://docs.anthropic.com/en/docs/claude-code/best-practices "Best Practices for Claude Code - Claude Code Docs"
[5]: https://developers.openai.com/codex/guides/agents-md/ "Custom instructions with AGENTS.md"
[6]: https://developers.openai.com/codex/skills/ "Agent Skills"
