# NOA Development Progress

## 2026-03-15

### 修复 L1 候选选拔逻辑 — hall_of_fame 持久候选记录

**问题**: L1 在 val 上产生好候选（+22），但 test 只提升 +8.5。三个选拔缺陷导致好候选跨轮丢失：
1. `_commit_best_before_observe` 只 val eval top-1，其余候选未验证
2. top-K 按 train score 排序，方差大
3. `_top_candidates` 每轮清空，前几轮好候选无法进入 final eval

**方案**: `_hall_of_fame: dict[str, dict]` 跨轮持久记录（`unified_agent.py`）

**改动**:
- `__init__`: 新增 `_hall_of_fame` 字段
- `_update_top_candidates`: 同步写入 hall_of_fame
- `_commit_best_before_observe`: val eval 所有 top-K 候选（不再只 top-1），每个写入 hall_of_fame
- `_final_eval_top_candidates`: 开头从 hall_of_fame 恢复磁盘上仍存在的候选，按 val_score 优先排序
- `_auto_accept_unadded_candidates`: 额外从 hall_of_fame 补充 val_score > baseline 的候选

## 2026-03-13

### Review 修复: 4 个正确性回归

**P1 — 拓扑编辑后刷新 checkpoint 基线** (`unified_agent.py:_save_pipeline_config`)
- `toggle_component`/`reorder_pipeline` 直接改 source_dir，但未更新 sandbox accepted snapshot
- 修复: `_save_pipeline_config` 末尾加 `sandbox.save_accepted_snapshot()` + `_source_version += 1`

**P2 — eval 缓存在 source baseline 变更后失效** (`unified_agent.py`)
- `_eval_result_cache` 仅按 ops_hash 索引，baseline commit 后不清理
- 修复: 所有 `_source_version += 1` 处加 `_eval_result_cache.clear()`

**P2 — 取消 eval budget 上限** (`core/protocol.py:OptimizationBudget.reached_limit`)
- 移除 `evals_used >= max_evals` 条件，只保留 step 限制

**P3 — merge 候选时保留文件删除** (`sandbox_manager.py:merge_candidates`)
- 原逻辑只遍历候选中存在的文件，删除操作丢失
- 修复: 对比基线文件集与每个候选文件集，若所有候选都删了某文件则从合并结果移除

**Codex Review 额外修复:**
- subprocess 路径丢失 val_set — `run_layer_subprocess` + `_run_mini_l1_subprocess` 新增 `val_set_pickle_path` 参数穿透
- 删除 9 个无法导入的过时 planner-era 测试文件

### 声明式拓扑 + 组件级结构操作

**目标**: 将 pipeline 拓扑从硬编码改为数据驱动，给 agent 新增拓扑操作工具。

**改动**:
1. **新增 `pipeline_config.json`** (PubMedQA + HotpotQA) — 声明式 pipeline 定义，包含 `inputs`、每个组件的 `requires`/`produces` 依赖关系和 `enabled` 开关
2. **pipeline.py 改造** — `_build_components()` 优先从 `pipeline_config.json` 加载，支持 `source_dir` 参数定位配置文件，硬编码作为 fallback
3. **unified_agent.py 新增 3 个工具** (仅 L1):
   - `get_pipeline_config`: 查看当前拓扑
   - `toggle_component`: 启用/禁用组件（带依赖警告）
   - `reorder_pipeline`: 重排组件顺序（带依赖校验，无效拓扑拒绝操作）
   - 拓扑修改后自动重建 target + probe + source files
4. **unified_prompts.py** — 新增 Structural Optimization 指导段落

### 性能优化: hardlink copytree, eval 缓存, 并行 final eval, trajectory 缓存

**问题**: 优化循环中多处串行瓶颈导致整体运行时间过长。

**方案** (4 项优化):

1. **Hardlink copytree** (`sandbox_manager.py`)
   - `_copytree_fast` 使用 `os.link` 替代文件复制，失败时自动 fallback
   - 仅用于 `checkpoint_candidate` 和 `merge_candidates`（写入路径做了 `os.unlink` 处理）
   - snapshot/accept/restore 保持普通复制（避免 inode 共享导致快照被污染）

2. **Eval 结果缓存** (`unified_agent.py`)
   - `_compute_ops_hash(ops_summary)` 计算 patch 的 SHA-256 哈希
   - `_eval_result_cache: dict[ops_hash, result]` 缓存评估结果
   - `_tool_eval_candidate` 评估前检查缓存，命中时跳过 sandbox eval

3. **并行 Final Eval** (`unified_agent.py`)
   - `_final_eval_top_candidates` 使用 `ThreadPoolExecutor` 并行评估 top-K 候选
   - 各候选独立沙盒，互不干扰；仅 1 个候选时走原始串行路径

4. **Trajectory 缓存** (`unified_agent.py`)
   - `_trajectory_cache: dict[question, Trajectory]` 缓存轨迹
   - `_source_version` / `_traj_cache_source_version` 追踪 source 变更
   - source 未变时: 全量命中跳过 observe，部分命中仅跑缺失样本（`_run_fresh_observe_inline`）
   - source 已变时: 清空缓存，走正常 observe 流程

5. **批量并行评估工具** (`unified_agent.py`, `unified_prompts.py`)
   - 新增 `eval_candidates_batch(labels=[...])` 工具，LLM 可一次提交多个候选并行评估
   - `ThreadPoolExecutor(max_workers=4)` 并行执行，结合 eval 缓存跳过已评估的候选
   - Prompt 新增 Path C 引导 LLM 在有多个行为类候选时使用批量评估

6. **修复 rejected validation 污染 accepted snapshot** (`unified_agent.py`)
   - `_commit_best_before_observe()` 原先 accept 后再 eval，rejected 时 accepted snapshot 已被污染
   - 改为先在 candidate_dir 评估，通过后才 accept，失败不影响 accepted snapshot

**修改文件**: `noa/sandbox_manager.py`, `noa/unified_agent.py`, `noa/unified_prompts.py`
**新增测试**: `tests/test_perf_optimizations.py` (11 个测试用例)

## 2026-03-12

### compact 改进: LLM 摘要 + 原始记录保存

**问题**: `_compact_messages` 对旧消息做粗糙字段提取，agent 的推理链和 tool 返回的详细数据全被截断/丢弃，导致多轮优化时 agent 失去关键上下文。

**方案**:
- compact 时将**完整原始 messages** 保存到 `.noa_meta/compact_raw_N.json`（包括 tool_calls 名称和参数）
- 用 **LLM 生成结构化摘要**（observations / patches attempted / key insights / current state），替代机械字段提取
- LLM 失败时 fallback 到改进的规则提取（也包含 agent reasoning）
- 摘要中包含原始文件路径，agent 觉得摘要不够时可自主 `read_source_file` 读完整记录
- prompt 增加 Rule 13: CONTEXT COMPACT，告知 agent compact 机制和如何读取原始记录

**修改文件**: `noa/unified_agent.py`, `noa/unified_prompts.py`

### merge_candidates: Patch 叠加合并

**问题**: 每轮优化生成多个 patch，但只能接受最优的一个。其他有效但非最优的 patch（修复不同问题）被浪费。

**方案**: 新增 `merge_candidates` 工具，允许将多个独立候选的 patch 合并为一个组合候选：
- `SandboxManager.merge_candidates(label, source_labels)` — 文件级 diff 检测 + 行级合并
  - 仅一个候选修改的文件 → 直接复制
  - 多个候选修改同一文件 → `difflib.SequenceMatcher` 行级合并，冲突区域取第一个候选版本
- `UnifiedOptimizerAgent._tool_merge_candidates()` — 工具入口，返回源候选分数 + 冲突信息
- Prompt 增加 Rule 11: PATCH STACKING 引导 LLM 在有 2+ 有效 patch 时使用合并策略

**工作流**:
1. 生成并评估多个单独 patch → 2. 识别 beat baseline 的 → 3. merge_candidates 合并 → 4. eval 组合候选 → 5. 取组合或最优单个

**修改文件**: `noa/sandbox_manager.py`, `noa/unified_agent.py`, `noa/unified_prompts.py`

## 2026-03-11

### 去掉 Orchestrator Round 1 重跑

**问题**: L2 eval 每个 candidate 时启动 mini-L1，内部已做完整优化 + test eval。但 Orchestrator 看到 `spawn_restart=True` 后又从头重跑一遍 L1（Round 1），耗时 ~2h，结果与 L2 eval 基本相同——纯粹浪费。

**修复**:
- `orchestrator.py`: 删除 `while spawn_restart` 循环，不再重跑 L1
- `unified_agent.py` `_build_result()`: 当 `_spawn_noa_modified=True` 时，从 history 中取 spawn_sublayer 的 `child_score`，若优于 L1 自身 final_score 则直接复用
- 移除 orchestrator 中不再使用的 `run_layer_subprocess` import

**修改文件**: `noa/orchestrator.py`, `noa/unified_agent.py`

### Bug Fix: eval_candidate 使用错误 baseline 比较

**问题**: `eval_candidate` 和 `accept_candidate` 用 test_set (50条) 算的 `_baseline_score` 去和 train 样本 (10条) 上的候选分数比较。两个不同数据集的分数不可比，导致本应 accept 的候选被误 reject。

**修复**:
- 新增 `_current_train_baseline` 字段，每次 observe 抽样后在同一批 train 样本上跑原始 pipeline 得到该轮 baseline
- `eval_candidate` / `accept_candidate` / `_auto_accept_unadded_candidates` 改为与 `_current_train_baseline` 比较
- L2 层不走 observe 抽样，fallback 到 `_baseline_score`（L2 初始化时传入的 precomputed baseline）

**待修改 (TODO)**:
- **Patch 累积问题**: 当前所有候选都是从原始代码出发的独立 patch，accept 一个后下一个候选不会基于已 accept 的代码。top-K 池中的候选互不叠加，最终只挑分数最高的一个 commit。应改为：accept 后更新 baseline 代码和 baseline 分数，后续 patch 在已修改的基础上累积。

**修改文件**: `noa/unified_agent.py`

### Bug Fix: accept_candidate 遗漏 + L2 冗余 FinalEval

**问题 1**: 最后一步 eval 的候选无法被 accept。agentic loop 在 eval_candidate 返回后达到 max_steps 结束，LLM 没机会调用 accept_candidate，导致高分候选未进入 top-K 池。
- **修复**: `_auto_accept_unadded_candidates()` — 在 `_build_result()` 开头自动将所有已 eval 且超过 baseline 但未在 top-K 池中的候选加入池。

**问题 2**: L2 的 FinalEval 冗余。L2 的 eval_candidate 启动完整 mini-L1，mini-L1 内部已在 test_set 上做 final eval，返回的分数就是 test 分数。L2 再跑一次 FinalEval（又启动 mini-L1）只引入随机噪声。
- **修复**: `_final_eval_top_candidates()` — level >= 2 时跳过重新 eval，直接用已有的 eval_candidate 分数 commit 最优候选。

**修改文件**: `noa/unified_agent.py`

## 2026-03-10 (续)

### 统一 Deadline 传播 + 结果协议修复

**根因**: L2 score=0 的直接原因是 mini-L1 子进程超时被杀 (3607s ≈ 3600s limit)，result_file 未写出，
L2 只拿到 stderr 进度条作为"错误信息"。三套独立超时机制 (run_forked/run_subprocess/agentic_loop wall_timeout)
语义不一致，deadline 不向下传播。

**修改内容**:

1. **统一 deadline 传播** (`unified_agent.py`)
   - `UnifiedOptimizerAgent` 持有绝对 `_deadline` (monotonic)
   - `_remaining_sec()` 方法供所有子操作动态获取剩余时间
   - `_nested_mini_l1_timeout()` 和 `_nested_eval_timeout()` 从剩余预算计算，不再硬编码
   - spawn 时 `wall_budget_sec` 向子层传播: `child_budget = remaining - margin`
   - eval/spawn 前检查 `_check_time_for_eval()`，时间不够直接拒绝

2. **统一结果协议** (`subprocess_runner.py`)
   - 所有返回 dict 包含 `status` 字段: success | timeout | crash | partial
   - 新增 `result_written` 字段标识 result_file 是否有效
   - `_extract_meaningful_error()` 过滤 tqdm 进度条噪音
   - timeout 场景的 error 消息包含具体信息 (duration, limit)

3. **sandbox eval 结构化** (`sandbox_manager.py`)
   - `eval_in_sandbox()` 返回 `status/timed_out/duration_sec` 字段
   - timeout 场景错误消息明确标注超时来源

4. **agentic_loop 超时语义** (`stages/agentic.py`)
   - 工具执行**前**检查墙钟，超时则跳过 tool 返回错误
   - 剩余不足 10min 时打 warning log
   - `exit_reason` 字段记录退出原因

5. **默认 4h 预算**
   - `DEFAULT_WALL_BUDGET_SEC = 14400` (从 max_steps*600)
   - `_DEFAULT_WALL_TIMEOUT = 14400` (从 600)
   - `run_layer_subprocess` default timeout 14400 (从 3600)
   - `NOA_SANDBOX_EVAL_TIMEOUT_SEC` default 14400 (从 600)

**修改的文件**: `noa/unified_agent.py`, `noa/subprocess_runner.py`, `noa/sandbox_manager.py`,
`noa/stages/agentic.py`, `noa/engine.py`, `scripts/run_l2_only.py`, `scripts/run_pubmedqa_l2_only.py`

---

## 2026-03-10

### PubMedQA L2-only 运行结果与问题诊断

**运行结果**: Baseline 74.00 → Final 74.00, 0 patches accepted

L2 生成了 1 个候选 patch (`fix_analyzer_optimizer_prompts_and_history`)，eval 得分 76（> baseline 74），但未被 accept。

#### 已修复的 Bug
1. **`max_tool_calls` 配额不足** — 原来 `max_steps * 4 = 20`，L2 的 read/search/dry_run 等轻量调用很多，eval 后配额耗尽 LLM 无法再调 accept_candidate。已改为 `max_steps * 8`。
2. **history patterns 存储为截断字符串** — `str(p)[:150]` 导致 L2 读到乱码。已改为结构化 dict 存储 (pattern, root_cause, affected_file, severity)。
3. **`run_pubmedqa_l2_only.py` KeyError** — history 条目没有 `label` 字段，已改用 `h.get('action')`。

#### L2 诊断发现的问题（框架层面）
1. **L1 把 "maybe" 误判为错误** — analyzer prompt 缺乏区分真正不确定性和模型错误的指引，导致生成有害 patch 强制 definitive answers
2. **L1 重复失败的 patch** — 同类 prompt 修改反复尝试又反复失败，optimizer prompt 对 past rejected patches 的引导不足

#### LLM 超时问题（根因分析）
- 每次 LLM 调用 spawn 独立子进程 (`_completion_with_hard_timeout`)
- 双层超时: 软超时 `LLM_TIMEOUT_SEC=120s` (litellm) + 硬超时 `_HARD_TIMEOUT_SEC=300s` (kill worker)
- **根因**: L2 agentic loop 的 messages 积累很大（observe trajectories + analyze 诊断 + read_source_file 源码），到后期 50k+ tokens。Kimi-K2.5 处理长上下文慢 + Together AI 服务不稳定 (502 Bad Gateway)
- **潜在优化方向**:
  1. 压缩 messages — observe/read 结果只保留摘要
  2. 减少 observe samples — L2 不需要 20 条，5-10 条够
  3. 换更快/更稳定的模型
  4. 增加超时（治标不治本）

#### 修改的文件
- `noa/unified_agent.py` — max_tool_calls 4→8, history patterns 结构化存储
- `utils/llm.py` — _HARD_TIMEOUT_SEC 180→300
- `scripts/run_pubmedqa_l2_only.py` — 修复 KeyError

---

## 2026-03-09

### 连接健康监测 + 三层防死机机制

**动机:** PubMedQA 嵌套运行中，L2 spawn 的 mini-L1 子进程在 Together AI 关闭连接后（TCP CLOSE_WAIT），httpx 连接池复用死连接导致 100% CPU spin，整个进程卡死 2+ 小时。

**根因:** 三层防御全部缺失——无连接池重置、无墙钟超时、无连续错误熔断。

**修复内容:**
1. **`utils/llm.py` — CircuitBreaker 熔断器**: 连续 5 次 LLM 调用失败后自动清理 litellm 内部 httpx 客户端缓存，强制重建连接
2. **`noa/stages/agentic.py` — 墙钟超时 + 连续错误熔断**: 新增 `wall_timeout_sec` 参数；连续 3 次 LLM 异常则自动退出当前 loop
3. **`noa/unified_agent.py` — 主循环超时**: 按 max_steps × 10min 计算墙钟上限
4. **`noa/stages/analyzer.py` — 单条轨迹超时**: per-trajectory agentic analysis 限制 120s
5. **`noa/subprocess_runner.py` — 子进程超时**: 从 4h 降到 1h

**改动文件:** `utils/llm.py`, `noa/stages/agentic.py`, `noa/unified_agent.py`, `noa/stages/analyzer.py`, `noa/subprocess_runner.py`

## 2026-03-08

### Train/Test 隔离闭环修复

**动机:** 虽然之前引入了 train/test 隔离，但存在 5 条泄漏/断裂通道：

**修复内容:**
1. **`run_eval` test 泄漏** — `_tool_run_eval` 改为用当轮 train 样本，不再碰 test_set
2. **L1 重启链 + L2 mini-L1 未传 train/test** — `subprocess_runner` 新增 `train_pool_pickle_path`/`test_set_pickle_path`/`train_sample_size`/`top_k` 参数；orchestrator 和 spawn_sublayer 都传入
3. **PubMedQA subprocess 硬编码 `f1_score`** — 改为动态获取 `f1_score` or `exact_match`
4. **Prompt 与实现不一致** — accept_candidate 描述改为"加入 top-K 池"而非"commit 到 workspace"
5. **`run_l2_only.py` 旧数据接口** — 改用 `total_n`/`test_n`/`train_sample_size`，正确构造 train/test split

**改动文件:** `unified_agent.py`, `subprocess_runner.py`, `orchestrator.py`, `unified_prompts.py`, `scripts/run_l2_only.py`

### Train/Test 隔离 + Top-K 候选池

**动机:** 之前 observe/eval 都用同一批固定 val_set，有过拟合风险；贪心策略只保留最优 patch，容易因单次采样噪声误杀好 patch。

**新数据流:**
- `test_set` (固定 50 条, seed=42) — 只在开头 (baseline) 和结尾 (final eval) 使用
- `train_pool` (剩余数据) — 每轮 observe 随机抽 25 条（seed 递增，保持多样性）
- 所有 patch eval 在当轮的 25 条 train 样本上进行
- 维护 top-3 候选池，finish 时自动用 test_set 对 top-3 做 full eval 选最优 commit

**关键改动:**
- `unified_agent.py`:
  - `val_set` → `train_pool` + `test_set` + `train_sample_size`
  - 新增 `_current_train_samples`、`_top_candidates` 状态
  - `_compute_baseline()`: 用 test_set 跑 baseline
  - `_final_eval_top_candidates()`: finish 时对 top-K 做 test_set full eval
  - `_update_top_candidates()`: 维护按分数降序的 top-K 池
  - `_tool_run_observe()`: 每轮从 train_pool 随机抽样（seed=42+episode）
  - `_tool_eval_candidate()`: 在 `_current_train_samples` 上评估
  - `_tool_accept_candidate()`: 加入 top-K 池，不再直接 commit
  - `_tool_run_eval()`: 用 test_set
  - `_tool_run_smoke()`: 用当轮 train 样本
  - `_build_result()`: 自动触发 final eval
- `engine.py`: `val_set` → `train_pool/test_set/train_sample_size/top_k`
- `orchestrator.py`: 同上参数透传
- `scripts/run_pubmedqa.py`: 加载 500 条 → 分 test_set(50) + train_pool(450)
- `scripts/run_nested.py`: 加载 300 条 → 分 test_set(50) + train_pool(250)
- `configs/*.json`: `data.n` → `data.total_n/test_n/train_sample_size`
- `unified_prompts.py`: 更新 Rule #9 说明 train/test 隔离和 top-3 池

**测试:** 78 个测试全部通过（旧 planner 测试除外，为 pre-existing issue）

### 新增 PubMedQA Target System（来自 OPTIMAS 论文）

**动机:** RAG (HotpotQA) 系统每次优化迭代时间过长（~2K system runs），PubMedQA 仅需 ~0.5K runs，迭代速度快 4 倍。

**架构（4 组件，异构配置）:**
- `ContextModelSelector` — 离散模型选择（可优化 selected_model）
- `ContextAnalyst` — prompt 优化 + LLM 参数
- `SolverModelSelector` — 离散模型选择
- `ProblemSolver` — prompt 优化 + LLM 参数

**评估:** yes/no/maybe 三分类精确匹配 (accuracy)

**新增文件:**
- `target_systems/pubmedqa/` — 完整 target system（config, components, pipeline, evaluate, prompts, _noa_adapter）
- `target_systems/pubmedqa/combined_PubMedQA_{train,test}.jsonl` — 500+500 条数据
- `configs/pubmedqa_default.json` — L1-only 配置
- `configs/pubmedqa_nested.json` — L1+L2 嵌套配置
- `scripts/run_pubmedqa.py` — 运行入口

**修改文件:**
- `utils/data.py` — 新增 `load_pubmedqa()` + `QAExample.context` 字段（向后兼容）

## 2026-03-07

### L2 Meta-Optimizer 关键 Bug 修复 — 分数尺度不一致 + 错误传播

**问题 1 (致命): observe 和 eval_candidate 分数尺度完全不同**
- observe 用 `child_score_fn(pred/target)` → 0-1 范围，乘 100 后存 baseline
- eval_candidate 用 `child_eval_fn` 直接取 `float(result.answer)` 原始值
- 例: mini-L1 得分 60, reasonable_target=80 → observe baseline=75, eval=60
- **结果: L2 永远无法接受任何 candidate**（eval 得分永远 < observe baseline）
- **修复:** `child_eval_fn` 改为使用与 `child_score_fn` 相同的 `pred/target` 归一化

**问题 2: subprocess 错误信息被吞**（上一会话已修复）
- `child_eval_fn` 不读取 `result.intermediate` 中的 error 信息
- `eval_in_sandbox` 不传递 `subprocess_errors`
- `_tool_eval_candidate` 不检测 subprocess 崩溃
- **修复:** 三处链路全部补齐错误传递

**问题 3: subprocess 错误缺乏结构化分类**
- `run_layer_subprocess` 只返回原始 error 字符串，L2 无法区分错误类型
- **修复:** 新增 `error_type` 字段（syntax_error/import_error/timeout/runtime_crash）

**问题 4: L2 prompt 缺乏 meta-optimizer 专属指导**
- L2 和 L1 用完全相同的 prompt，L2 没有关于 subprocess 诊断的指导
- **修复:** `unified_prompts.py` 新增 "Meta-Optimizer (L2) Specific Guidance" 段落

**修改文件:**
- `noa/unified_agent.py` — child_eval_fn 归一化修复 + error_type 传递
- `noa/subprocess_runner.py` — 结构化错误分类
- `noa/unified_prompts.py` — L2 诊断指导
- `noa/sandbox_manager.py` — subprocess_errors 透传

## 2026-03-06

### 三核心 Bug 修复 — 沙盒种子/L2 触发/checkpoint 漂移

**Bug 1: 沙盒 vs 主线采样种子不一致**
- **根因:** `sandbox_manager.py` `eval_in_sandbox` 用无种子 `random.sample()`，`unified_agent.py` 用 `Random(seed=42).sample()`
- **修复:** `eval_in_sandbox` 加 `seed` 参数（默认 42）；`_tool_eval_candidate` 透传 seed；工具 schema 加 `"seed": {"type": "integer"}`
- **修改文件:** `noa/sandbox_manager.py`, `noa/unified_agent.py`

**Bug 2: L2 层永远不会被触发**
- **根因:** `OptimizationBudget` 有 `max_no_improve_steps` 但无追踪字段，evals 耗尽直接停止
- **修复:**
  - `protocol.py`: 新增 `no_improve_count: int = 0` + `stagnation_detected()` + `to_summary()` 加入 `no_improve_count`
  - `unified_agent.py`: `_tool_eval_candidate` 和 `_tool_run_eval` 追踪 `no_improve_count`；`_tool_accept_candidate` 改为严格大于（`<=` 拒绝）
  - `unified_agent.py`: 新增 `_spawn_escape_used` 字段 + `_should_stop` 逃生口逻辑（预算耗尽+停滞+可spawn → 放行一次）
  - `unified_agent.py`: `_dispatch_tool` 入口逃生窗口限制（只允许 `spawn_sublayer`/`finish`/`get_state`/`get_budget_status`）
- **修改文件:** `noa/core/protocol.py`, `noa/unified_agent.py`

**Bug 3: checkpoint_candidate 状态漂移 + search_not_found**
- **根因:** `checkpoint_candidate` 从 `source_dir` 复制，但 `apply_patch` 会就地修改 `source_dir`，后续 checkpoint 的 search block 对不上
- **修复:**
  - `sandbox_manager.py`: 新增 `_accepted_snapshot` + `save_accepted_snapshot()` + `_get_checkpoint_base()`；`checkpoint_candidate` 从 accepted 快照复制；`accept_candidate` 刷新快照
  - `engine.py`: 创建 SandboxManager 后调用 `save_accepted_snapshot()`
  - `patch_protocol.py`: 新增 `_find_rstrip_occurrences` + `_matched_span_length`；`_validate_search` 和 `_resolve_position` 加 fuzzy fallback；`_apply_update` 用 `_matched_span_length` 计算实际 span
  - `unified_agent.py`: checkpoint 失败时 `search_not_found` 附带 nearby 行上下文
- **修改文件:** `noa/sandbox_manager.py`, `noa/engine.py`, `noa/patch_protocol.py`, `noa/unified_agent.py`

**测试:** 34 个测试全部通过（原 27 + 新增 7）
- `test_sandbox_trajectory.py` 新增: `test_eval_in_sandbox_seed_parity`, `test_checkpoint_uses_accepted_snapshot`, `test_accept_refreshes_snapshot`
- `test_patch_protocol.py` 新增: `test_fuzzy_rstrip_match`, `test_fuzzy_preserves_unique_constraint`, `test_fuzzy_occurrence`, `test_fuzzy_apply_preserves_char_level`

### 清理旧 Planner 架构 — 全面迁移到 Unified Agentic Loop

**目标:** 删除旧的 planner-driven 控制流代码，全部只保留 unified agentic loop 架构。

**删除清单:**
- `noa/planner/` 整个目录（8 个文件: agent.py, executors.py, guardrails.py, protocol.py, prompts.py, reducer.py, trace.py, __init__.py）

**迁移:**
- `PlannerBudget` → `OptimizationBudget`，从 `noa/planner/protocol.py` 迁移到 `noa/core/protocol.py`

**重写文件:**
- `noa/engine.py` — 删除 planner 循环分支，`run()` 直接执行 unified agent；删除 `unified_mode` 开关和所有 planner-specific 参数
- `noa/orchestrator.py` — 删除 `unified_mode` 开关和旧参数（tool_calls、l2_*、orchestrator_* 等）
- `noa/subprocess_runner.py` — 删除 `unified_mode` 参数和旧 tool_calls 参数
- `scripts/run_nested.py` — 删除 `unified_mode` 透传和 tool_calls 参数
- `noa/unified_agent.py` — import 改为 `OptimizationBudget`，删除 `unified_mode=True` 透传
- `noa/__init__.py` — 导出 `OptimizationBudget`
- `configs/hotpotqa_nested.json` — 删除旧的 tool_calls 配置项

**修改文件:** `noa/core/protocol.py`, `noa/stages/analyzer.py`（result key 兼容）

### L2 同构 Unified Agentic Loop 改造

**问题:** L2 meta-optimizer 被 spawn 时永远走 planner-driven 模式，缺少 snapshot/checkpoint/eval_candidate 等工具，Analyzer 无 probe 退化为单次 LLM 调用，spawn 信号链断裂。

**改动清单 (5 文件):**

| 优先级 | 改动 | 文件 |
|---|---|---|
| P0-1 | spawn 计数仅成功后 +1 + noa_modified 时立即终止 loop | `unified_agent.py` |
| P0-2 | `run_layer_subprocess` 签名新增 `unified_mode` 并透传给 NOptimizer | `subprocess_runner.py` |
| P0-3 | orchestrator 重启 L1 时透传 `self.unified_mode` | `orchestrator.py` |
| P0-4 | mini-L1 subprocess (L2 reward target) + child NOptimizer 都透传 unified_mode | `unified_agent.py`, `executors.py`, `engine.py` |
| P0-5 | `_build_result` 返回 `spawn_restart` 字段 | `unified_agent.py` |
| P1-1 | child NOptimizer (L2) 传 `unified_mode=True` | `unified_agent.py` |
| P1-2 | executor child NOptimizer 传 `self.unified_mode` | `executors.py` |
| P1-3 | ws_dir 按层级隔离 (`unified_l1` vs `unified_l2`) | `engine.py` |
| P2-1 | `_tool_inspect_artifacts` L2 fallback: 直接扫描 source_dir | `unified_agent.py` |
| P2-2 | `_tool_run_reproduce` L2 fallback: 直接调用 target | `unified_agent.py` |

**关键设计决策:**
- spawn 计数时序: 仅 `result_dict.get("ok")` 时才 `spawn_calls_used += 1`，失败不消耗预算
- 立即终止: 用 `_spawn_noa_modified` flag + `_should_stop()` 检查，不污染 budget 计数
- unified_mode 透传: subprocess runner 模板中注入，orchestrator/executor 各自从 self 取值
- L2 probe: 有意接受差异，L2 analyzer 走 simple 模式，主动探索由 unified agent 工具完成

**修改文件:** `noa/unified_agent.py`, `noa/subprocess_runner.py`, `noa/orchestrator.py`, `noa/planner/executors.py`, `noa/engine.py`

## 2026-03-05

### Unified Mode 运行修复 — 3 个关键 Bug

**问题:** `unified_mode=True` 运行时 LLM agent 无法正常完成优化循环

**Bug 1: 工具名称含 `.` 不符合 Anthropic API 规则**
- Anthropic API 要求工具名匹配 `^[a-zA-Z0-9_-]{1,128}$`，但工具名用 `.` 分隔命名空间（如 `target.read_source_file`）
- **修复:** `unified_agent.py` + `unified_prompts.py` — `.` 分隔符改为 `__`（如 `target__read_source_file`）

**Bug 2: step_count 每个工具调用都递增，导致 budget 快速耗尽**
- `_dispatch_tool` 对所有带前缀工具都 `step_count += 1`，30 个工具调用就耗完 budget
- 实际上 `read_source_file`、`list_source_files`、`search_text` 等只读工具不应消耗 step
- **修复:** `unified_agent.py` — 新增 `_HEAVY_TOOLS` 集合，仅 `run_observe`/`run_eval`/`analyze`/`apply_patch`/`checkpoint_candidate`/`eval_candidate`/`accept_candidate` 等重量级工具消耗 step budget

**Bug 3: PatchOp schema 不完整 + ops 参数字符串化**
- `ops` 参数 schema 只写 `{"type": "array", "items": {"type": "object"}}`，LLM 不知道需要什么字段
- 导致 `file_path` 为空（`"code": "file_not_found", "file_path": ""`）
- LLM 有时将 ops 作为 JSON 字符串传入而非 array
- **修复:**
  - `unified_agent.py` — 在 `_build_tool_schemas` 中定义详细的 `_PATCH_OP_SCHEMA`（含 op/file_path/search/replace 等字段定义 + 可选文件列表提示）
  - `_parse_ops()` — 支持字符串形式 JSON 自动解析（`isinstance(raw_ops, str)` → `json.loads`）

**Prompt 改进:**
- `unified_prompts.py` — Key Rules #3 扩展为详细的 commit 流程说明（dry_run → checkpoint → eval → accept）
- 新增 Rule #6: 优先完成一个完整优化周期

**修复效果:**
- 修复前: LLM 空转 30 步，0 eval，0 accepted
- 修复后: LLM 正确执行 observe → analyze → read_source → dry_run → checkpoint → eval 完整流程
- 4 个候选方案被评估（3 个 checkpoint_candidate + 3 个 eval_candidate），工作流正常运行

**修改文件:** `noa/unified_agent.py`, `noa/unified_prompts.py`

### Persistent Diagnostic Optimization Agent — Phase 1 + Phase 2

**Phase 1: 底层基础设施**
- `noa/core/protocol.py` — 新增 `PatchOp`, `PatchValidationError`, `StructuredPatch` dataclass；`EvalResult.patch` 支持 `StructuredPatch`
- `noa/patch_protocol.py` — **新建** 结构化 patch 引擎：`apply_patch_ops()` (事务性) + `validate_patch_ops()` (干运行验证)
  - 5 种操作: update/create/delete/insert_after/insert_before
  - 安全特性: path traversal 防御, must_be_unique 唯一性检查, context_before/after 消歧, occurrence 指定匹配
- `noa/sandbox_manager.py` — **新建** candidate 级沙盒：snapshot/restore/checkpoint_candidate/eval_in_sandbox/accept_candidate
  - `accept_candidate` 是唯一 commit 路径，确保 candidate 隔离不被绕过
- `noa/trajectory_store.py` — **新建** 持久化轨迹存储：save_episode/load_episode/query/compare_episodes
- `noa/optimizee_protocol.py` — **新建** 窄腰协议 `OptimizeeAdapter` (Protocol): describe/smoke/eval/reproduce/collect_artifacts
- `noa/diff_utils.py` — re-export `apply_patch_ops`/`validate_patch_ops`
- `noa/tools/probe.py` — ComponentProbe 新增 `reproduce()` + `inspect_artifacts()`

**Phase 2: Unified Agent + Worker Tool Matrix**
- `noa/unified_agent.py` — **新建** UnifiedOptimizerAgent 核心实现
  - 完整工具集 (6 类 25+ 工具): 观察探索/执行评估/诊断/编辑验证/沙盒候选管理/状态控制
  - 工具命名空间前缀: `target.*` (L1) vs `optimizer.*` (L2) 硬约束
  - L1 独有: run_component/run_from 微实验工具
- `noa/unified_prompts.py` — **新建** unified agent prompt 模板
- `noa/stages/optimizer.py` — `_OptimizerProbe` 新增 `apply_structured_patch` 工具
- `noa/stages/evaluator.py` — `evaluate()` 支持 `StructuredPatch`；新增 `evaluate_candidate()` 函数
- `noa/planner/guardrails.py` — 新增 `check_preconditions()` + `_has_candidate_patch()` 兼容双 patch 类型
- `noa/planner/executors.py` — 新增 `get_auxiliary_tool_schemas()`/`execute_auxiliary_tool()`；`_do_evaluate_patch` 兼容 StructuredPatch
- `noa/planner/protocol.py` — `PlannerState.candidate_patch` 支持 `StructuredPatch`
- `noa/engine.py` — 新增 `unified_mode: bool = False` + `_run_unified()` 分支
- `noa/orchestrator.py` — 透传 `unified_mode`
- `noa/workspace.py` — 新增 `get_sandbox_manager()` 方法

**测试:** 113 个测试全部通过（原 86 + 新增 27）
- `tests/test_patch_protocol.py` (17 个) — 验证/应用各 op 类型 + 事务性 + occurrence + context 消歧
- `tests/test_sandbox_trajectory.py` (10 个) — sandbox snapshot/restore/candidate + trajectory store CRUD

**设计要点:**
- `unified_mode=False` 为默认，旧模式完全不变（零回归）
- StructuredPatch 和 DeltaPatch 并存，所有下游检查点都兼容双类型
- candidate 隔离: checkpoint 从干净快照创建，不改 source_dir；accept_candidate 是唯一 commit 路径

### Fix L2 Meta-Optimizer Role Confusion — 修复 L2 误改 L0 目标系统文件
- **问题:** L2 元优化器应优化 `noa/` 框架代码，但持续尝试修改 L0 文件（`components.py`, `retriever.py` 等），导致所有 patch 因 `search_mismatch` 失败
- **根因:** L2 analyzer 读取 L1 轨迹（含 L0 诊断信息），无信号区分"L0 问题描述"与"L1 策略缺陷"
- **修改文件:**
  - `noa/core/protocol.py` — `to_prompt_context()` 为 L2+ 添加角色约束声明（明确可写范围、诊断方向）
  - `noa/core/prompts.py` — `SINGLE_ANALYZER_PROMPT`, `AGENTIC_ANALYZER_PROMPT`, `ANALYZER_PROMPT` 添加 meta-optimizer affected_file 约束；`OPTIMIZER_PROMPT` 添加文件范围约束
  - `noa/stages/analyzer.py` — `_format_l1_intermediate()` 添加 META-NOTE 标签；`analyze_incremental()` 末尾添加 affected_file 后验校验（不在 source_files 中的文件被清空并标注）

## 2026-03-04

### AlphaEvolve 启发的双改善方向

**方向1: Multi-Pattern Parallel Pipeline（`parallel_optimize` action）**

核心设计：解耦"评估"与"提交"，所有 patch 在同一 baseline 上独立评估，最后贪心组合不冲突的 patch 一次性提交。

- Phase 1: **Parallel Generation** — 对 top-N patterns 各生成 M 个候选（不同 temperature: 0.0/0.4/0.7），使用 `ThreadPoolExecutor` 并行调用 `optimize_agentic`
- Phase 2: **Parallel Cascade Evaluation** — Stage 0 冲突预检 → Stage 1 Smoke(3条) → Stage 2 Medium(10条)，全部基于同一 baseline 并行评估
- Phase 3: **Greedy Combination** — 按 delta 降序贪心选择不冲突 patch → 合并后 Full Eval → 组合退化时退化到单最优

**方向2: 结构化 LLM 评估反馈（`generate_eval_feedback`）**

- `evaluator.py` 新增 `generate_eval_feedback()` — 用 LLM 分析样本级 F1 变化，输出 improved_types/degraded_types/insights/next_focus/causal_links
- `PlannerState` 新增 `eval_feedback_history` 和 `baseline_details`
- 反馈注入 Analyzer prompt 的 `past_attempts` 段，指导下一轮诊断方向
- `evaluate()` 新增 `auto_commit=False` 参数，支持批量评估时延迟提交

**新增工具函数:**
- `extract_patch_regions()` — 提取 patch 修改的文件行范围
- `check_patch_conflict()` — 检测两个 patch 是否修改重叠区域
- `merge_patches()` — 顺序合并多个不冲突 patch

**修改文件:**
- `noa/core/protocol.py` — `EvalResult` 新增 `feedback` 字段
- `noa/stages/evaluator.py` — `auto_commit` 参数, `generate_eval_feedback`, `extract_patch_regions`, `check_patch_conflict`, `merge_patches`
- `noa/stages/optimizer.py` — `optimize`/`optimize_agentic` 支持 `temperature` 参数
- `noa/stages/analyzer.py` — `analyze_incremental` 新增 `eval_feedback` 参数
- `noa/planner/protocol.py` — `ActionName` 新增 `parallel_optimize`; `PlannerState` 新增 `candidate_patches`, `eval_feedback_history`, `baseline_details`
- `noa/planner/executors.py` — 新增 `_do_parallel_optimize` 完整实现
- `noa/planner/prompts.py` — 添加 `parallel_optimize` action 及使用指引
- `noa/planner/guardrails.py` — 添加 `parallel_optimize` 前置条件验证
- `noa/planner/reducer.py` — 处理 `parallel_optimize` 状态更新，记录 baseline_details 和 eval feedback
- `noa/planner/agent.py` — heuristic 优先选择 `parallel_optimize`
- `noa/engine.py` — 默认预算调大 `max_llm_calls=120`, `max_evals=20`

**测试:** 86 个测试全部通过（原 74 + 新增 12）
- 新增 `tests/test_parallel_optimize.py` (12 个测试) — patch region 提取/冲突检测/合并/reducer/guardrail/action_counts

---

## 2026-03-03

### Bug 修复 + Agentic Loop 统一重构

**6 个 Bug 修复：**
1. **`target_delta=0.0` 导致过早停止** — `PlannerBudget.target_delta` 默认值 `0.0` → `float('inf')`，`engine.py` 同步。第一个 patch 被接受后不再触发 `should_stop()`
2. **`_format_trajectory` 静默丢失非 dict intermediate** — `analyzer.py` 新增 else 分支，字符串/数值 intermediate 输出到格式化文本
3. **`ComponentProbe.run_from` 不累积上下文** — `probe.py` 改用 `dict(inputs)` + `current.update(result)`，下游组件可看到上游输出
4. **`_run_fresh_observe` intermediate_complete 判断不完整** — 新增 `required_intermediate_keys` 参数，替换 `bool(result.intermediate)` 为 `_is_intermediate_complete()`，所有内部调用点传入 `self.required_intermediate_keys`
5. **`spawn_calls_used` 双重独立计数** — 删除 executor 的递增，保留 reducer 统一管理；reducer 新增 `layer_context.spawn_calls_used` 同步；engine 从 `LayerContext` 传入 `max_spawn_calls`
6. **死代码清理** — 删除 `orchestrator.py` 的 `_render_source_context`、`_l2_score_fn`、`PlannerBudget` import；删除 `executors.py` 的 `action_result_to_dict`、`asdict` import

**Agentic Loop 统一重构：**
- `noa/stages/agentic.py` — 新增 `early_stop_fn` 和 `no_tool_call_prompt` 参数
- `noa/stages/observer.py` — `observe_agentic` 内联循环替换为 `agentic_loop()` 调用
- `noa/stages/analyzer.py` — `_diagnose_one_agentic` 内联循环替换为 `agentic_loop()` 调用
- `noa/stages/optimizer.py` — `optimize_agentic` 内联循环替换为 `agentic_loop()` 调用
- 三个组件的 `llm_call_with_tools` 直接 import 已移除，统一走 `agentic_loop`

**修改文件:** `noa/planner/protocol.py`, `noa/engine.py`, `noa/stages/analyzer.py`, `noa/tools/probe.py`, `noa/stages/observer.py`, `noa/planner/executors.py`, `noa/planner/reducer.py`, `noa/orchestrator.py`, `noa/stages/agentic.py`, `noa/stages/optimizer.py`

**测试:** 74 个测试全部通过（原 55 + 新增 19）
- 新增 `tests/test_agentic_loop.py` (8 个测试) — early_stop_fn、no_tool_call_prompt、parse_fn 重试、预算耗尽
- 新增 `tests/test_probe_run_from.py` (3 个测试) — 上下文累积、部分管道、错误保留
- 新增 `tests/test_format_trajectory.py` (6 个测试) — 非 dict intermediate、混合类型、截断
- 更新 `tests/test_budget_stop_conditions.py` — 新增默认 target_delta 不过早停止
- 更新 `tests/test_unified_executor.py` — spawn_success 断言改为 `ctx.spawn_calls_used == 0`
- 更新 `tests/test_reducer_state_updates.py` — 新增 spawn 后 budget/layer_context 同步测试

### 统一运行入口
- 删除 `scripts/run_l1.py`，统一使用 `scripts/run_nested.py`
- 所有参数通过 config JSON 注入，不再使用 argparse
- L1-only 等价于 `nesting.max_spawn_calls = 0`
- 新增 `scripts/run.sh` bash 脚本: `./scripts/run.sh [config_path]`
- 新增 `configs/hotpotqa_l1_only.json` (L1-only 配置)
- 重构 `configs/hotpotqa_default.json` 为统一 config schema: `{data, optimizer, nesting, output}`

## 2026-02-22

### Initial Setup
- Created `memory_bank/PRD.md` — full English PRD for the NOA framework
- Created `CLAUDE.md` — project guidance based on PRD architecture
- Created `memory_bank/progress.md` — this progress log
- Existing ACE pipeline implementation lives on `main` branch (SpanAnalyzer → InsightReflector → InsightsOptimizer)
- Current branch `yuhan` is clean for new framework development

### L0 Target System: HotpotQA RAG Pipeline
- 实现了完整的 5 组件 RAG pipeline (QuestionRewriter → InfoExtractor → Retriever → HintGenerator → AnswerGenerator)
- **文件结构:** (`utils/` 和 `target_systems/` 在项目根目录，`noa/` 留给 NOA 框架)
  - `utils/llm.py` — litellm 薄封装，指数退避重试
  - `utils/data.py` — HotpotQA 数据加载 (HuggingFace, 固定 seed 采样)
  - `target_systems/hotpotqa_rag/components.py` — 5 个组件 (BaseComponent 基类)
  - `target_systems/hotpotqa_rag/pipeline.py` — Pipeline 编排 + state_dict/update_config
  - `target_systems/hotpotqa_rag/retriever.py` — ColBERTv2 + Wikipedia fallback
  - `target_systems/hotpotqa_rag/evaluate.py` — token-level F1 (与 OPTIMAS 一致)
  - `target_systems/hotpotqa_rag/prompts.py` — 4 个 LLM 组件的默认 prompt
  - `target_systems/hotpotqa_rag/config.py` — dataclass 配置 + JSON 序列化
  - `scripts/run_baseline.py` — CLI 基线评估入口
  - `configs/hotpotqa_default.json` — 默认配置
- **验证:** F1 函数单元测试通过，Config 往返测试通过，Pipeline 创建/更新测试通过
- **Baseline 目标:** F1 ≈ 30-37 (unoptimized)

### 检索器修复：本地离线检索 + Wikipedia UA 修复
- **问题:** ColBERT 服务器 (20.102.90.50:2017) 国内不可达；Wikipedia API 缺 User-Agent 返回 403
- **方案 A (主力):** 新增 `LocalRetriever` — 从 HotpotQA distractor context 字段收集去重段落，用 TF-IDF + cosine similarity 检索 top-k
  - `utils/data.py` 新增 `build_corpus()` 函数
  - `retriever.py` 新增 `LocalRetriever` 类 (sklearn TfidfVectorizer)
  - `config.py` backend 默认改为 `"local"`，支持 `"local" | "colbert" | "wikipedia"`
  - `components.py` / `pipeline.py` / `run_baseline.py` 适配 corpus 传递
  - `requirements.txt` 添加 `scikit-learn`
- **方案 B:** `WikipediaRetriever` 添加 User-Agent header
- **验证结果:** 5 条样本 Mean F1 = 49.44 (远超之前无检索的 20.63，也超过预期的 30-40)
- **⚠️ 问题:** Local TF-IDF 使用数据集自带 context 做语料属于"作弊"（gold 段落就在索引中），不能作为真实 baseline

### NOA 框架实现（I-O-A-O-E 闭环）
- **架构更新**: O-A-O-E → I-O-A-O-E，在闭环前增加 Initiator 阶段
  - Initiator 用 LLM 内省 `state_dict()` 提取结构化 `SystemDescription`，一次性运行
  - 后续迭代只刷新 `current_config` 和 mutable field 的 `current_value`
- **新增文件**:
  - `noa/protocol.py` — 统一数据结构 (SystemDescription, MutableField, Trajectory, Diagnosis, DeltaPatch, EvalResult)
  - `noa/prompts.py` — 所有 LLM agent 的 prompt 模板（简洁风格，便于 L2 优化）
  - `noa/initiator.py` — LLM 内省 + baseline 计算
  - `noa/observer.py` — 采样运行目标系统，收集执行轨迹
  - `noa/analyzer.py` — LLM 错误诊断（核心阶段，max_tokens=2000）
  - `noa/optimizer.py` — LLM 生成 delta patch
  - `noa/evaluator.py` — 沙盒验证 patch，接受或回滚
  - `noa/engine.py` — NOptimizer 主类，编排 I-O-A-O-E 循环
  - `noa/__init__.py` — 导出 NOptimizer + 协议数据类
  - `scripts/run_l1.py` — L1 优化入口脚本
- **设计原则**: 奥卡姆剃刀 — 同一个 NOptimizer 类服务 L1 和 L2
  - NOptimizer 自身暴露 TargetSystem 接口 (state_dict / update_config / load_state_dict)
  - L2 的 target 就是 L1 的 NOptimizer 实例
- **收敛条件**: 连续 2 轮 patch 被拒绝则停止
- **统一模型**: 所有 LLM agent 默认使用 gpt-4.1-mini
- **CLAUDE.md 更新**: 架构描述改为 I-O-A-O-E，新增奥卡姆剃刀 + prompt 简洁 + 统一模型约定

---

## 当前状态总结

### ✅ 已完成
1. **项目初始化** — PRD、CLAUDE.md、memory bank 结构
2. **L0 RAG Pipeline 完整实现** — 5 组件 (QuestionRewriter → InfoExtractor → Retriever → HintGenerator → AnswerGenerator)
3. **评估框架** — token-level F1、evaluate_batch 并发评估
4. **配置系统** — dataclass + JSON 序列化、增量 patch 更新
5. **Wikipedia API 修复** — 添加 User-Agent header，已可正常访问
6. **Local TF-IDF 检索** — 临时可用，但属于作弊方案
7. **NOA 框架 (Unified Agentic Loop)** — UnifiedOptimizerAgent 驱动，LLM 自主决定工作流
8. **Unified System Description Protocol** — SystemDescription + SourceFile 数据结构
9. **NOptimizer 嵌套设计** — 同一类服务 L1/L2，L2 通过 spawn_sublayer 触发
10. **旧 Planner 架构完全删除** — 全面迁移到 Unified Agentic Loop
11. **PubMedQA Target System** — 4 组件异构配置，yes/no/maybe 三分类

### WikiSemanticRetriever 实现
- **问题:** ColBERT 不可达，Local TF-IDF 用 gold context 属于作弊
- **方案:** 新增 `WikiSemanticRetriever` — 三步检索：
  1. Wikipedia Search API → 文章标题列表（召回 n_search 篇）
  2. Wikipedia Extracts API (`exintro=true`) → 每篇文章的引言摘要（与 ColBERT wiki17_abstracts 对齐）
  3. sentence-transformers (`all-MiniLM-L6-v2`, 80MB) → 向量余弦相似度 → top-k（语义重排）
- **修改文件:**
  - `retriever.py` — 新增 `WikiSemanticRetriever` 类
  - `config.py` — `RetrieverConfig` 新增 `embed_model`、`n_search` 字段，默认 backend 改为 `"wiki_semantic"`
  - `components.py` — 添加 `wiki_semantic` backend 分支
  - `configs/hotpotqa_default.json` — backend 改为 `"wiki_semantic"`
  - `requirements.txt` — 添加 `sentence-transformers`
- **新增可调参数:** `embed_model`、`n_search` 可作为 L1 优化目标

### Auto Adapter — 自动代码内省适配
- **目标:** 让 NOA 支持任意未适配的 pipeline，无需手动实现 4 个接口方法
- **新增文件:**
  - `noa/auto_adapter.py` — `auto_adapt(source_dir, entry_hint, model, force)` 核心模块
    - 扫描源码 → LLM 生成 Adapter 类 → exec 动态加载 → 验证 state_dict 返回 dict
    - 最多 3 次重试，失败时把错误信息追加到 prompt
    - 生成的代码缓存为 `{source_dir}/_noa_adapter.py`，下次直接加载
- **修改文件:**
  - `noa/prompts.py` — 新增 `AUTO_ADAPTER_SYSTEM` 和 `AUTO_ADAPTER_PROMPT`（含接口规范 + 参考示例）
  - `noa/observer.py` — 消除硬编码 `from target_systems.hotpotqa_rag.evaluate import f1_score`，改为 `score_fn` 参数注入（`None` 时 lazy import 作为 fallback）
  - `noa/engine.py` — `NOptimizer.__init__` 新增可选 `score_fn` 参数，传递给 `observe()`
  - `noa/__init__.py` — 导出 `auto_adapt`
- **向后兼容:** `score_fn=None` 时行为与改动前完全一致

### Code-Diff 重构 (2026-02-23)
- **目标:** 从 Config-Only 优化升级为 Code-Diff 优化，让 Optimizer 可以修改任何代码
- **核心设计:** 参考 OpenEvolve 的临时文件策略 — 内存中应用 diff → 临时目录评估 → 接受后写回原始文件
- **修改文件:**
  - `noa/protocol.py` — 删除 `MutableField`，新增 `SourceFile`、`DiffBlock`；`SystemDescription` 改用 `source_files`；`DeltaPatch` 改用 `diffs: list[DiffBlock]`
  - `noa/diff_utils.py` — **新建**，实现 `extract_diffs`、`apply_diffs_in_memory`、`write_to_temp_dir`、`commit_to_source`
  - `noa/prompts.py` — 全部 prompt 改为源码上下文 + SEARCH/REPLACE diff 输出格式
  - `noa/initiator.py` — 改为读取源码文件 + LLM 输出 workflow/components（不再调用 state_dict）
  - `noa/analyzer.py` — Prompt 新增完整源码上下文，输出新增 `affected_file`、`suggested_fix`
  - `noa/optimizer.py` — 输出 SEARCH/REPLACE diff，调用 `diff_utils.extract_diffs()` 解析
  - `noa/evaluator.py` — 临时目录沙盒策略：`apply_diffs_in_memory → write_to_temp_dir → target_factory(temp_dir) → eval → commit/rollback`
  - `noa/engine.py` — 新参数 `source_dir`、`target_factory`；每轮刷新 `source_files`；patch 接受后用 `target_factory` 重建 target
  - `noa/auto_adapter.py` — 简化为只需 `__call__` 接口 + `target_factory` 工厂函数；使用 `importlib.util.spec_from_file_location` 全新加载无缓存问题
  - `noa/__init__.py` — 更新导出（移除 `MutableField`，新增 `SourceFile`、`DiffBlock`）
  - `scripts/run_l1.py` — 适配新接口（`source_dir` + `target_factory` 替代 `target` 实例）
- **验证:** diff_utils 4 项单元测试全部通过；所有模块 import 验证通过

### 解耦 + Token 限制调整 (2026-02-23)
- **目标:** 消除 3 处硬编码耦合（score_fn fallback 绑定 HotpotQA、失败阈值 0.5、eval 返回 key "mean_f1"），提升 LLM 输出 token 限制
- **修改文件:**
  - `target_systems/hotpotqa_rag/evaluate.py` — 返回 key `"mean_f1"` → `"score"`
  - `noa/initiator.py` — `baseline_result["mean_f1"]` → `["score"]`；max_tokens 1000 → 4096
  - `noa/evaluator.py` — `result["mean_f1"]` → `["score"]`
  - `noa/observer.py` — 删除 `from target_systems...` hardcoded fallback，`score_fn` 改为 keyword-only 必传参数
  - `noa/analyzer.py` — 新增 `failure_threshold` 参数，默认自适应（低于中位数视为失败）；max_tokens 2000 → 16384
  - `noa/optimizer.py` — max_tokens 2000 → 16384
  - `noa/auto_adapter.py` — max_tokens 2000 → 16384
  - `noa/engine.py` — `score_fn` 改为必传，新增 `failure_threshold` 参数传递给 analyzer；删除 `f1 < 0.5` 硬编码
  - `scripts/run_l1.py` — 显式传入 `score_fn=f1_score`
  - `scripts/run_baseline.py` — `results['mean_f1']` → `['score']`
- **Auto Adapter 结论:** 保留，角色是"怎么运行目标系统"（`__call__` + `target_factory`），与 Code-Diff（"怎么修改目标系统"）正交
- **验证:** import 通过，无 `"mean_f1"` key 残留，noa/ 内无 `from target_systems`，无 `f1 < 0.5` 硬编码

### 历史摘要 Diff 模式 (2026-02-23)
- **问题:** Analyzer/Optimizer 不知道之前试过什么，可能重复生成同样的失败 patch
- **方案:** 历史尝试用 SEARCH/REPLACE diff 摘要传递给 LLM（每块≤30行，每行≤100字符截断）
- **修改文件:**
  - `noa/engine.py` — `self.history` 改存实际 `DiffBlock` 对象；新增 `_format_history()` / `_format_diff_block()` / `_truncate_line()` 辅助函数；analyzer/optimizer 调用时传入 `past_attempts`
  - `noa/prompts.py` — `ANALYZER_PROMPT` 和 `OPTIMIZER_PROMPT` 新增 `## Past Attempts\n{past_attempts}` 段落
  - `noa/analyzer.py` — 新增 `past_attempts: str = ""` 参数
  - `noa/optimizer.py` — 新增 `past_attempts: str = ""` 参数

### Orchestrator — L1+L2 自主嵌套优化 (2026-02-23)
- **目标:** 实现 L2 meta-optimizer 自动编排，L1 表现不满意时自动升级到 L2 优化 noa/ 代码
- **核心难点:** L2 修改 noa/ 代码后需重新运行 L1 验证效果，Python 模块缓存使同进程重载不可靠 → 采用 subprocess 隔离策略
- **新增文件:**
  - `noa/subprocess_runner.py` — subprocess 运行 L1 隔离逻辑
    - `run_l1_subprocess()`: 生成 runner 脚本字符串 → `subprocess.run()` 执行 → 解析 `__NOA_RESULT__` 标记行
    - `serialize_dataset()`: pickle 序列化 dataset，支持缓存复用
    - DiffBlock → dict 序列化解决 JSON 传输问题
    - PYTHONPATH 设为 `noa_parent:project_root`，确保修改后的 noa/ 代码优先加载
  - `noa/orchestrator.py` — Orchestrator 薄层编排器 + L1Target
    - `Orchestrator.run()`: Round 0 运行 L1 → 判断升级 → L2 优化 noa/ → Round 1 重新 L1 → ...
    - `_should_escalate()`: 改善量 ≤ threshold 或零接受率时升级
    - `_run_l2()`: 构建 L2 NOptimizer（同构嵌套），target=L1Target, dataset=伪触发器
    - `L1Target`: L2 的 target，__call__ 在 subprocess 中运行缩小参数 L1
    - L2 的 system_description 包含 L1 完整运行历史 + 优化约束
  - `scripts/run_nested.py` — 入口脚本
    - CLI 参数: `--l1-iterations`, `--l1-samples`, `--l2-iterations`, `--l2-rounds`, `--l2-l1-iterations`, `--l2-l1-samples`, `--model`, `--escalation-threshold`
- **修改文件:**
  - `noa/__init__.py` — 导出 `Orchestrator`
- **不修改的文件:** `noa/engine.py`, `noa/stages/`, `noa/diff_utils.py` — 接口完全不变
- **关键设计:**
  - L1 运行方式: subprocess（干净环境，无模块缓存）
  - L2 NOptimizer 不修改接口，通过 L1Target/l2_eval_fn/l2_dataset 适配
  - L2 observe: n_samples=1（L1 本身已是多样本聚合）
  - L2 评估 L1: 缩小参数（2 iterations, 10 samples）作为快速代理信号
  - 数据集传递: pickle 序列化到 .noa_cache/ 目录

### Workspace 隔离 — 原始代码不可变 (2026-02-23)
- **目标:** 原始代码（noa/、target_systems/）永远不变，所有优化修改在 workspace 副本上进行，关键节点拍快照
- **新增文件:**
  - `noa/workspace.py` — WorkspaceManager 类
    - `.noa_runs/<run_id>/workspace/` — 运行期间的工作副本
    - `.noa_runs/<run_id>/snapshots/` — 关键时间点的只读快照（init, after_l1_round_N, after_l2_round_N）
    - `setup()`: 复制 noa/ + source_dir + utils/ → workspace，拍 init 快照
    - `snapshot(label)`: workspace → snapshots/<label>/
  - `scripts/restore_snapshot.py` — 快照查看/导出 CLI 工具（--list / --run + --label + --dest）
- **修改文件:**
  - `noa/orchestrator.py` — `run()` 开头创建 WorkspaceManager，所有后续 L1/L2 操作使用 workspace 路径
    - `_run_l1_subprocess()` 传 `_ws_noa` 和 `_ws_source`
    - `_run_l2()` 中 `NOptimizer(source_dir=_ws_noa, ...)`，L1Target 的 l0_source_dir 也用 `_ws_source`
    - 每轮 L1/L2 后自动拍快照
    - 返回结果新增 `run_dir` 和 `snapshots` 字段
  - `.gitignore` — 添加 `.noa_runs/` 和 `.noa_cache/`
- **不修改的文件:** `noa/subprocess_runner.py`、`noa/engine.py`、`noa/stages/` — 接口不变，已通过参数接收路径

### 端到端链路 Bug 修复 (2026-02-23)
- **Bug 1 (致命): `_load_adapter` 未定义** — `auto_adapter.py:34` 调用 `_load_adapter()` 但从未定义
  - **修复:** 实现 `_load_adapter()` 函数，从缓存文件读取代码并调用 `_exec_adapter()`
- **Bug 2 (严重): workspace 缺少 `target_systems/__init__.py`**
  - **原因:** `workspace.py setup()` 只复制了 `target_systems/hotpotqa_rag/` 目录，未复制中间目录的 `__init__.py`
  - **影响:** subprocess 中 `import target_systems.hotpotqa_rag.*` 无法从 workspace 加载，回退到 project_root 原始代码
  - **修复:** 在 `setup()` 中遍历 source_dir 相对路径的中间层级，复制每层的 `__init__.py`
- **Bug 3 (严重): 模块缓存导致 evaluator 无法测试 patch 效果**
  - **原因:** `target_factory(temp_dir)` 通过 importlib 加载 adapter，但 adapter 的 import 走 `sys.modules` 缓存（原始代码）
  - **修复:** 新增 `_isolate_target_modules()` / `_restore_target_modules()` 函数：
    - 清除 source_dir 和 temp dir 下的模块缓存
    - 重定向父包 `__path__` 使新目录优先
    - 加载完成后恢复 `__path__`
- **Bug 4: 相对导入失败** — LLM 生成的 adapter 用 `from pipeline import RAGPipeline`，但 `pipeline.py` 有 `from .config import ...` 相对导入
  - **原因:** `_exec_adapter()` 用 `exec()` 加载 adapter，目标模块未作为包导入，相对导入无上下文
  - **修复:** 新增 `_ensure_package_imported()` 函数：
    - 从 source_dir 向上查找 `__init__.py` 层级，确定完整包名
    - 用 `__import__()` 正式导入包和所有子模块
    - 为子模块创建顶级别名（`sys.modules['pipeline'] = sys.modules['target_systems.hotpotqa_rag.pipeline']`）
- **Bug 5: `_ClassNamespace` 不可迭代** — 某些模块的 `__path__` 是不可迭代对象
  - **修复:** `_isolate_target_modules()` 中用 `try: list(mod_path)` 保护
- **Bug 6: subprocess 错误信息被进度条淹没**
  - **原因:** `subprocess_runner.py` 取 `stderr[:2000]`，sentence-transformers 的 Loading weights 进度条占满缓冲区
  - **修复:** 过滤掉进度条行，只保留有意义的错误信息
- **修改文件:** `noa/auto_adapter.py`, `noa/workspace.py`, `noa/subprocess_runner.py`
- **验证结果:** `run_nested.py` 端到端跑通
  - Adapter 生成 ✓、Workspace 隔离 ✓、L1 subprocess ✓、L2 Meta-Optimizer ✓

### 正式运行结果 (2026-02-23)
- **参数:** `--n 50 --l1-iterations 5 --l1-samples 20` (默认值), subprocess timeout 从 600s 增至 1200s
- **结果:**
  - L0 原始基线: **38.02** F1
  - L1 优化后: **44.45** F1 (**+6.43**)
  - L2 未触发（L1 改善量 > 0 且接受率 > 0，Orchestrator 判断不需要升级）
- **L1 迭代详情 (4 轮迭代, 1/4 accepted):**
  - Iter 1: 38.02 → 32.94 (REJECTED) — 添加空检索 fallback + prompt 改进 + 关键词后处理
  - **Iter 2: 38.02 → 44.45 (ACCEPTED)** — 过滤空 passage + prompt 改进 + 未知 backend 报错
  - Iter 3: 44.45 → 41.08 (REJECTED) — 在 Iter2 基础上再加关键词后处理
  - Iter 4: 44.45 → 43.83 (REJECTED) — prompt + 关键词换行符处理
- **有效 patch 内容 (Iter 2):**
  - `components.py Retriever.forward`: 过滤空白 passage，空结果返回 fallback 文本
  - `prompts.py ANSWER_GENERATOR`: 增加 "concise and direct" 指令
  - `prompts.py INFO_EXTRACTOR`: 增加 "comma-separated keywords only" 指令
  - `components.py Retriever.__init__`: colbert backend 显式处理，未知 backend 抛异常
- **run_dir:** `.noa_runs/20260223_224828`

### L2 逻辑修正 — 饱和触发 (2026-02-23)
- **问题:** 原 `_should_escalate()` 逻辑：L1 有改善就不触发 L2。用户纠正：L1 完成所有迭代 = 饱和，应该始终触发 L2
- **修复:** 删除 `_should_escalate()` 方法和 `escalation_threshold` 参数，L2 始终在 L1 结束后触发（最多 max_l2_rounds 轮）
- **修改文件:** `noa/orchestrator.py`, `scripts/run_nested.py`

### L1+L2 完整嵌套运行结果 (2026-02-24)
- **参数:** n=50, l1-iterations=5, l1-samples=20, l2-iterations=3, l2-rounds=2
- **结果:** Baseline **36.11** → Final **52.53** F1 (**+16.42**)
- **3 轮详情:**
  - Round 0 L1: 36.11→47.74 (2/5 accepted), L2: 0 iterations (failure threshold 问题)
  - Round 1 L1: 33.50→46.32 (2/5 accepted), L2: 0 iterations (subprocess 返回 0)
  - Round 2 L1: 52.53→52.53 (0/2 accepted, 已饱和)
- **L2 两个 bug:**
  - **Bug 7: L2 failure_threshold 不适配** — L2 只有 1 个样本，median 自适应阈值 = 唯一得分，strict `<` 永远为 False → 0 failures → "No patterns found. Stopping"
  - **Bug 8: L2 的 L1 subprocess 污染共享 target 代码** — L1Target 调用 L1 subprocess 时，L1 的 `commit_to_source` 会修改 `_ws_source`，导致后续轮次的 target 代码状态不可预测（Round 1 L2 baseline=0 就是此原因）

### L2 Bug 修复 (2026-02-24)
- **Bug 7 修复:** `orchestrator.py _run_l2()` — 传 `failure_threshold=1.0` 给 L2 的 NOptimizer。任何 < 100% 的 L1 得分都视为 failure
- **Bug 8 修复:** `subprocess_runner.py` — 新增 `isolate_source` 参数；`orchestrator.py L1Target.__call__` — 传 `isolate_source=True`。L2 调用 L1 时，在临时副本上运行，不污染共享 `_ws_source`
- **修改文件:** `noa/subprocess_runner.py`, `noa/orchestrator.py`

### L2 子进程通信 Bug 修复 — 临时文件协议 (2026-02-24)
- **问题:** `run_l1_subprocess()` 基于 stdout 行扫描 `__NOA_RESULT__` 标记提取 JSON 结果，有三个 bug：
  1. **JSON 解析脆弱** — stdout 混入其他输出或子进程异常退出时，标记行丢失
  2. **stderr 过滤不完整** — 只过滤 `Loading weights` 和 `Materializing`，HuggingFace 模型加载警告（BertModel LOAD REPORT 等）被当作错误返回
  3. **未检查 returncode** — 子进程崩溃时不报告退出码
- **修复:**
  1. **临时文件传递结果** — 子进程将 JSON 结果写入 `tempfile` 文件，父进程从文件读取，完全绕过 stdout 解析问题
  2. **扩展 stderr 噪音过滤** — 新增 `_NOISE_KEYWORDS` 元组，覆盖所有 HuggingFace/PyTorch 常见警告
  3. **returncode 检查** — 过滤后无信息时用 `proc.returncode` 生成有意义的错误消息
  4. **临时文件清理** — 正常路径和超时路径都确保清理
- **删除:** `_RESULT_MARKER` 常量（不再需要）
- **修改文件:** `noa/subprocess_runner.py`

### 采样一致性修复 — Observer/Evaluator 分离 (2026-02-24)
- **问题:** Observer(seed=42)、Evaluator(seed=42)、Initiator(dataset[:n]) 三者采样不一致
  - Observer 和 Evaluator 用相同子集 → 等于"训练集=测试集"，patch 可能过拟合特定 case
  - Initiator baseline 用前 N 条、Evaluator 用随机 N 条 → baseline vs after_score 对比有偏
- **修复:**
  - Observer: `Random(42).sample()` — 诊断用（不变）
  - Evaluator: `Random(seed+1).sample()` — 评估用（seed=43），与 Observer 不重叠
  - Initiator: `Random(43).sample()` — 与 Evaluator 一致，baseline 对比公平
- **进一步改进:** 每轮迭代使用不同 seed（`obs_seed = 42 + i*2`, `eval_seed = 42 + i*2 + 1`），充分利用整个数据集
  - Evaluator 每轮在当轮 eval 子集上重新算 before_score（`target_factory(source_dir)` → `eval_fn`），确保 before/after 在完全相同的子集上对比
  - engine 的 `baseline_score` 参数保留但 Evaluator 不再依赖它
- **修改文件:** `noa/stages/evaluator.py`, `noa/stages/initiator.py`, `noa/engine.py`

### sys.modules 别名污染修复 (2026-02-24)
- **问题:** `_ensure_package_imported()` 创建短名别名（`sys.modules["config"]`, `sys.modules["pipeline"]` 等）污染全局模块命名空间
  - `sys.modules["config"]` 指向 `hotpotqa_rag.config`（dataclass 模块）
  - `sentence_transformers` / `transformers` 内部 import `config` 时拿到错误模块 → 加载崩溃
  - 这是 L2 的 L1 subprocess 返回 `final_score=0` 的根因
- **根因时序:**
  1. `_ensure_package_imported()` 创建别名（包括 `config`）
  2. `exec(code)` 执行 adapter 代码（需要别名解析 `from pipeline import ...`）
  3. `adapter_cls()` 实例化 → 创建 Retriever → `from sentence_transformers import ...` → **此时别名仍在，污染 transformers 的 import**
  4. `finally` 清理别名 — 太晚了
- **修复:** 在 `_exec_adapter` 和 `target_factory` 中，`exec()` / `exec_module()` 完成后**立即**清理别名，再实例化 adapter
  - `_ensure_package_imported()` 返回创建的别名列表
  - `exec()` 后立即 `sys.modules.pop(alias)` 清理
  - `adapter_cls()` 在干净的命名空间中运行，sentence_transformers 正常加载
- **修改文件:** `noa/auto_adapter.py`
- **验证:** L2 Baseline F1 从 0.00 恢复为 58.22（L1 subprocess 在 L2 内正常运行）

### 早停逻辑修复 — 连续空分析才停止 (2026-02-24)
- **问题:** `engine.py` 中单次 `failure_patterns` 为空就 `break`，导致 L1 只跑 1 轮就退出（max_iterations=3 名存实亡）
  - Analyzer LLM 可能偶尔解析失败（JSON 格式不符）或在新样本上暂时找不到 pattern
  - 同理 empty patch 也会立即退出
- **修复:**
  - "No failure patterns" → `continue`（跳到下一轮换样本重试），连续 2 次才 `break`
  - "Empty patch" → `continue`（换样本重试，不退出）
- **修改文件:** `noa/engine.py`

### L2 Meta-Optimizer 重构 — 直接消费 L1 轨迹 (2026-02-25)
- **问题:** L2 复用通用 NOptimizer，Initiate/Observe 阶段各跑大量 mini L1 subprocess（~140 次），而 L1 历史已可用，浪费 ~93% L2 计算量
- **方案:** L2 直接分析 L1 已有 history，跳过 Initiate 和 Observe，仅在 Evaluate 阶段跑少量 mini L1 验证 patch
- **新增:**
  - `noa/core/prompts.py` — `L2_META_ANALYZER_SYSTEM` + `L2_META_ANALYZER_PROMPT`（meta-optimizer 角色，接收 noa/ 源码 + L1 历史 + L2 过去尝试）
  - `noa/stages/analyzer.py` — `meta_analyze()` 函数（调用 L2 专用 prompt，复用 `_parse_patterns()`）+ `_format_l1_history()` 辅助函数
- **修改:**
  - `noa/orchestrator.py`:
    - import 更新：移除 `NOptimizer`，新增 `meta_analyze`、`optimize`、`evaluate`、`collect_sources`、`SystemDescription`
    - `l2_n_samples` 默认值从 `l1_n_samples`(20) 改为 `3`
    - `_run_l2()` 重写：手动构建 SystemDescription → meta_analyze → optimize → evaluate 循环
    - 删除 `_build_l2_context()` 方法（格式化逻辑移至 `_format_l1_history()`）
- **Evaluate 策略:** 每轮 L2 iteration 跑 1 次完整 L1（与主 L1 相同参数）验证 patch 效果，删除 `l2_l1_iterations` / `l2_l1_n_samples` 缩小参数
- **效果:** L1 subprocess 调用从 ~140 次降至每轮 L2 iteration 1 次

### Analyze 多 Pattern 逐个修复 + Initiate/Observe 去冗余 (2026-02-25)
- **问题 1:** 每轮 Analyze 可能诊断出多个 failure pattern（A, B, C, D），但 Optimizer 只修最高 severity 的一个。下一轮用新 seed 重新 Observe + Analyze，之前诊断的 B, C, D 被丢弃，浪费 Analyze 计算
- **方案 1:** 一次 Analyze 的所有 pattern 逐个 optimize → evaluate，用完后再重新 Observe + Analyze
- **问题 2:** Initiate 跑 20 条算 baseline，第一轮 Observe 又跑 20 条，重复浪费。与 L2 跳过 Observe 直接用 L1 历史的思路一致
- **方案 2:** Initiate 只做 LLM 内省；首次 observe 既算 baseline 又供第一轮 Analyze 使用
- **修改文件:**
  - `noa/stages/initiator.py`:
    - 移除 `dataset`, `eval_fn`, `target_factory`, `n_samples` 参数
    - 移除 baseline 计算（`baseline_score=0`，由 engine 从首次 observe 推导）
  - `noa/engine.py`:
    - 新增 `_sort_patterns_by_severity()` 辅助函数（high > medium > low）
    - `run()` 重写：initiate 后立即 observe（seed=43 与 Evaluator 一致），从轨迹算 baseline
    - 首轮复用 initiate 轨迹，后续 cycle 正常 observe
    - 内层循环遍历每个 pattern 逐个 optimize → evaluate
    - accepted 则立即刷新源码 + 重建 target + 更新 baseline
    - 每个 patch 的 history 含 `pattern` 和 `severity` 字段
    - 终止条件：连续 2 次 cycle 无任何 accepted patch
  - `noa/orchestrator.py`:
    - `_run_l2()` 同理：meta_analyze 返回多个 pattern → 逐个 optimize → evaluate
    - 导入 `_sort_patterns_by_severity` 和 `Diagnosis`
  - `noa/core/prompts.py`:
    - OPTIMIZER_PROMPT 措辞 "highest-severity pattern" → "the following pattern"
- **计算量节省:** 首轮省 20 次 target 调用（原 Initiate baseline + 第一轮 Observe = 40 次 → 20 次）

### Artifacts 机制 — 让 Analyzer 拥有完整执行上下文 (2026-03-03)
- **问题:** Analyzer 信息严重不足 — 看不到 JSON 配置（max_tokens/temperature）、finish_reason、token 用量等元数据
- **核心理念（借鉴 OpenEvolve）:** 不在 prompt 里指示 LLM 该看什么，而是给足上下文，让 LLM 自己发现所有问题
- **修改文件:**
  - `utils/llm.py` — `LLMResponse` 新增 `finish_reason: str` 字段，从 litellm 响应中提取
  - `target_systems/hotpotqa_rag/components.py` — 新增 `_make_artifacts()` 辅助函数，4 个 LLM 组件（QuestionRewriter, InfoExtractor, HintGenerator, AnswerGenerator）返回 `_artifacts` 字段（model, max_tokens, temperature, finish_reason, tokens_used, latency_ms）
  - `noa/stages/initiator.py` — `collect_sources()` 扩展支持 `.json` 文件；`source_context` 构造根据后缀选 `python`/`json` 代码块标记
  - `noa/core/protocol.py` — `get_source_context()` 根据文件后缀选择语法标记
  - `noa/stages/analyzer.py` — `_format_trajectory()` 对 `_artifacts` 字段不截断（通常 <150 字符），其他字段保持 200 字符截断
  - `noa/core/prompts.py` — 重写 ANALYZER_PROMPT / SINGLE_ANALYZER_PROMPT / L2_META_ANALYZER_PROMPT：去掉具体检查项，改为 high-level 风格（"你有完整上下文，自主发现根因"）

### Artifact 反馈闭环对齐 (2026-03-03)
- **目标:** 对齐 OpenEvolve 的 artifact 反馈闭环，让 Analyzer 能看到执行异常和评估上下文
- **3 个差距:**
  1. Artifact 内容范围太窄 — 不捕获执行异常
  2. 缺少跨迭代反馈闭环 — past_attempts 不含"为什么失败"
  3. Observer/Evaluator 无异常容错 — 单样本崩溃导致整轮失败
- **修改文件 (7 步):**
  - `noa/core/protocol.py` — `Trajectory` 和 `EvalResult` 新增 `error: str | None = None` 字段
  - `noa/stages/observer.py` — try/except 包裹 target() + score_fn()，异常时生成带 error 的 Trajectory(f1=0.0)
  - `noa/stages/evaluator.py` — except Exception 分支返回 EvalResult(accepted=False, error=traceback)
  - `noa/stages/analyzer.py` — `_format_trajectory()` 展示截尾 500 字符 traceback；`_format_l1_history()` 展示 eval_error 和低分样本摘要
  - `noa/engine.py` — history 新增 eval_details/eval_error；`_format_history()` 展示评估上下文
  - `noa/orchestrator.py` — l2_history 同样新增 eval_details/eval_error
  - `noa/core/prompts.py` — ANALYZER/SINGLE_ANALYZER PROMPT 提及 ERROR 信息；OPTIMIZER PROMPT 提及评估反馈

### Agentic 能力升级 — Analyzer + Evaluator (2026-03-03)
- **目标:** 赋予 NOA 组件实验能力，Analyzer 从"单次猜测"升级为"假设→实验→验证"循环，Evaluator 从"全量评估"升级为"级联快速淘汰"
- **核心差距（vs OpenEvolve）:** NOA 组件缺少交互式实验能力，Analyzer 只能猜测根因无法验证
- **修改文件 (10 个):**
  - `utils/llm.py` — 新增 `ToolCall`, `LLMResponseWithTools` 数据类 + `llm_call_with_tools()` 函数（litellm tool calling 支持）
  - `noa/tools/__init__.py` — 新建空包
  - `noa/tools/probe.py` — **新建** `ComponentProbe` 类：包装目标系统，提供 `run_component()` / `run_from()` 单组件执行能力，输出 OpenAI function calling 格式工具定义
  - `noa/auto_adapter.py` — `AUTO_ADAPTER_PROMPT` 新增 `get_components()` 接口要求；`_validate()` 增加可选验证
  - `noa/core/prompts.py` — 新增 `AGENTIC_ANALYZER_SYSTEM` + `AGENTIC_ANALYZER_PROMPT`（ReAct 实验循环 prompt）；`AUTO_ADAPTER_PROMPT` 加 `get_components` 示例和规则
  - `noa/core/protocol.py` — `EvalResult` 新增 `artifacts: dict` 字段
  - `noa/stages/analyzer.py` — `analyze_incremental()` 新增 `probe` + `max_tool_calls` 参数；有 probe 时 `_diagnose_one_agentic()` ReAct 循环（工具调用→结果→再推理），无 probe 时 fallback 到原有单次调用
  - `noa/stages/evaluator.py` — 三级级联评估：Stage1 语法检查（diff 能否应用）→ Stage2 烟雾测试（3 样本快速淘汰，< baseline*0.8 直接拒绝）→ Stage3 全量评估 + 结构化 artifacts 返回
  - `noa/engine.py` — `NOptimizer.__init__` 新增 `max_tool_calls` 参数；Analyze 阶段自动创建 `ComponentProbe` 并传入；history 记录 `artifacts`；`_format_history()` 格式化 artifacts 反馈（stage、rejection reason、degraded_samples 等）
  - `scripts/run_l1.py` + `scripts/run_nested.py` — 新增 `--max-tool-calls` CLI 参数
- **设计决策:**
  - 默认 agentic 模式（有 probe 时），不需要 flag 开关
  - 无 `get_components()` 的旧 adapter 自动 fallback 到反射模式提取组件
  - 反射也失败时 fallback 到非 agentic 单次 LLM 调用
  - Evaluator 级联评估借鉴 OpenEvolve 思想，Stage1/2 快速淘汰避免浪费全量评估计算

### NOA 同构化重构 — L1/L2 完全同构 (2026-03-03)
- **目标:** L1/L2（及更深层）完全同构——同一份代码、同一套 prompt，差异仅通过 `LayerContext` 参数注入。所有 LLM 组件支持多轮 tool-calling。Planner 可自主 spawn 子层。所有文件访问受层级 ACL 控制。
- **6 Phase 重构，51 个测试全部通过:**

**Phase 1: 基础设施**
- `noa/core/protocol.py` — 新增 `LayerContext` dataclass（ACL 权限、spawn 预算、path traversal 防御）
- `noa/stages/agentic.py` — **新建**通用 `agentic_loop()` 函数
- `noa/planner/protocol.py` — `PlannerBudget` 增加 spawn 预算；`PlannerState` 增加 `layer_context`；`ActionName` 增加 `spawn_sublayer`

**Phase 2: 统一 Prompt**
- `noa/core/prompts.py` — 5 个 prompt 模板增加 `{layer_context}` 占位符；**删除** L2 专用 `L2_META_ANALYZER_SYSTEM` + `L2_META_ANALYZER_PROMPT`
- `noa/stages/observer.py` — `AGENTIC_OBSERVER_PROMPT` + `observe_agentic()` 增加 `layer_context`
- `noa/planner/prompts.py` — `PLANNER_PROMPT` 增加 `{layer_context}` + `spawn_sublayer` action
- 所有调用点传 `layer_context=""`

**Phase 3: 统一 Analyzer**
- `noa/stages/analyzer.py` — `analyze_incremental()` 增加 `layer_context` 参数；**删除** `meta_analyze()` 和 `_format_l2_observe()`；`_format_l1_history()` 重命名为 `format_parent_history()`（公开 API）
- `noa/stages/optimizer.py` — `optimize()` 增加 `layer_context` 参数
- 新增 `tests/test_meta_analyze_equivalence.py` (4 个等价测试)

**Phase 4: 统一 Executor + 递归化**
- `noa/planner/executors.py` — **重写**: 合并 `L1ActionExecutor` + `L2ActionExecutor` → 统一 `ActionExecutor`；新增 `_do_spawn_sublayer()`；向后兼容别名
- `noa/engine.py` — 使用统一 `ActionExecutor`；增加 `layer_context`、`observer_search_roots`
- `noa/orchestrator.py` — **重写**: 简化为 workspace 管理 + 单个 NOptimizer 薄入口；`L1Target` → `SubprocessTarget`；**删除** 硬编码 L1/L2 调度
- `noa/subprocess_runner.py` — `run_l1_subprocess` → `run_layer_subprocess`（向后兼容别名）
- `noa/planner/guardrails.py` — `spawn_sublayer` 校验（can_spawn + no_improve_steps >= 2）
- `noa/planner/reducer.py` — `spawn_sublayer` 分支（重置 trajectories/diagnosis/no_improve_steps）+ `should_stop` 检查 spawn 预算

**Phase 5: 组件 Agentic 化**
- `noa/planner/agent.py` — **重写**: `_PlannerProbe` 工具集（get_history_detail, get_diagnosis_detail, get_budget_status）+ 多轮 tool-calling `_decide_agentic()`；`max_tool_calls=0` 退化为原有行为
- `noa/stages/optimizer.py` — 新增 `optimize_agentic()` + `_OptimizerProbe`（verify_search_block, read_source_file, dry_run_patch）
- `noa/planner/executors.py` — `_do_propose_patch()` 切换到 `optimize_agentic()`；新增 `optimizer_max_tool_calls` 参数

**Phase 6: 权限模型 + 测试 + 文档**
- `noa/diff_utils.py` — `_normalize_diff_path()` 拒绝 `../` 和绝对路径；`extract_diffs()` 解析后规范化；`commit_to_source()` 增加 `layer_context` 写权限校验
- `noa/stages/observer.py` — `_ObservationProbe` 增加 `layer_context`；`get_tool_schemas()` level >= 2 时禁用 `run_experiment_command` 和 `discover_experiment_commands`
- `noa/stages/evaluator.py` — `evaluate()` 增加 `layer_context` 参数，透传给 `commit_to_source()`
- `scripts/run_l1.py` — 新增 `--optimizer-tool-calls`、`--planner-tool-calls`、`--max-depth`、`--max-spawn-calls`；构造 `LayerContext`
- `scripts/run_nested.py` — 新增 `--max-depth`、`--max-spawn-calls`
- **新增测试 (4 个文件, 34 个新测试):**
  - `tests/test_layer_context.py` — LayerContext 方法 + path traversal 防御
  - `tests/test_permission_acl.py` — diff 路径规范化 + commit 写权限校验
  - `tests/test_unified_executor.py` — 统一 executor 行为 + spawn sublayer 边界
  - `tests/test_spawn_sublayer.py` — guardrails 前置条件 + reducer 状态重置

**文件总览:** 新建 5 个 | 重写 3 个 | 修改 16 个 | 测试: 51 通过

### spawn_sublayer 真实逻辑实现 (2026-03-03)
- **问题:** `_do_spawn_sublayer` 是空壳占位符（只增计数器+返回 ok=True），engine 不拦截 spawn 结果，orchestrator 不处理递归。Planner 说 "spawn" → 什么都没发生。
- **设计决策:**
  - L2 在同进程运行（避免跨进程 state 传递复杂度）
  - L2 评估通过 subprocess 运行 mini-L1（确保加载修改后的 noa/）
  - L2 修改 noa/ 后 L1 必须重启（模块缓存过期）
  - 延迟 import `NOptimizer` 避免循环依赖
- **修改文件:**
  - `noa/planner/executors.py` — 重写 `_do_spawn_sublayer()` + 新增 `_run_child_optimizer()`；`__init__` 新增 `noa_dir`, `project_root`, `dataset_pickle_path` 参数；child_target_factory 闭包调 `run_layer_subprocess` 运行 mini-L1
  - `noa/engine.py` — `NOptimizer.__init__` 新增 `noa_dir`, `dataset_pickle_path`；传参给 `ActionExecutor`；主循环新增 spawn+noa_modified 检测 break；返回 dict 新增 `spawn_restart`
  - `noa/orchestrator.py` — `run()` 改为循环结构：Round 0 in-process L1 → spawn_restart → Round 1+ subprocess L1 重启（加载修改后 noa/）；子进程 L1 不传 noa_dir（无法再 spawn，自然终止递归）
- **不修改的文件:** reducer、guardrails、protocol、subprocess_runner（现有逻辑已够用）

### Planner 信息断裂修复 — pattern 选择 + 增量 observe (2026-03-03)
- **问题 1:** Planner LLM 不知道 `propose_patch` 可传 `pattern_index`，`summary()` 不暴露 pattern 列表 → 始终修同一个 pattern
- **问题 2:** Planner LLM 不知道 `observe` 可传 `n_samples` → 无法增量 observe→analyze 循环构建 failure pool
- **修改文件:**
  - `noa/planner/prompts.py` — 新增 `Action params` 段落（observe 的 n_samples、propose_patch 的 pattern_index 等）；新增第 6、7 条 hard constraints（选未尝试 pattern、小批量增量 observe）
  - `noa/planner/protocol.py` — `PlannerState` 新增 `pool_size: int` 和 `attempted_patterns: list[dict]`；`summary()` 暴露 `pool_size`、`attempted_patterns`、`patterns` 列表（index/name/severity/count/component）
  - `noa/planner/reducer.py` — observe/analyze 分支从 payload 读取 `pool_size`；evaluate_patch 分支将 active_pattern 名称+结果追加到 `attempted_patterns`
  - `noa/planner/executors.py` — `_do_observe` payload 新增 `pool_size` 字段

### Meta Evolve Skill 系统四项修复 (2026-03-19)
- **修复 1:** `_build_observation()` 的 `population_summary` 改为从真实 population 对象（`population.scores()`）计算，含 percentiles/min/max；population 不可用时 fallback 到 trajectory，且去掉了 `score > 0` 过滤
- **修复 2:** `_compute_step_dynamics()` 的 `running_best` 初始值从 `0.0` 改为 `all_scores[0]`，避免负分任务指标失真
- **修复 3:** `SkillEvidence` 新增 `stagnation_levels`、`error_rates`、`co_active_counts` 字段，`record()` 和 `record_activation()` 同步传递上下文，save/load 也已更新
- **修复 4:** `SkillDistiller` 的 `DISTILL_SYSTEM` 提示词 axis 列表从旧的 5 轴（prompt调整/搜索控制/上下文丰富）更新为当前 3 轴（reflection/diagnosis/strategy）
- **修改文件:** `meta_evolve/skill_orchestrator.py`、`meta_evolve/skill.py`、`meta_evolve/skill_library.py`、`meta_evolve/skill_distiller.py`

### ❌ 未完成 / 待解决
1. ~~**检索器替换（高优先级）**~~ → ✅ 已完成 WikiSemanticRetriever
2. **Baseline 评估** — 需要用 WikiSemanticRetriever 跑完整 baseline（100 条），确定真实 F1 基线
3. ~~**L2 Meta-Optimizer 验证**~~ → ✅ Orchestrator 已实现 L1+L2 编排
4. ~~**端到端验证**~~ → ✅ `run_nested.py` 已跑通嵌套优化（修复 6 个 bug 后）
5. ~~**L1 subprocess 首轮超时**~~ → ✅ timeout 已从 600s 增至 1200s
6. **L2 完整验证** — Bug 7+8 已修复，需重新运行验证 L2 能否实际优化 noa/ 代码
