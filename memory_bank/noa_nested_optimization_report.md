# NOA 嵌套优化实验报告

**日期**: 2026-03-04
**配置**: `configs/hotpotqa_nested.json`
**模型**: `claude-haiku-4-5-20251001`
**数据集**: HotpotQA validation, n=50
**运行时长**: ~5 小时（L2 未完成，手动终止）

---

## 总览

| 指标 | 值 |
|------|-----|
| L0 原始 baseline F1 | **8.78** |
| L1 单层优化 best F1 | **33.40** (+280%) |
| L2 spawned L1 最高 F1 | **56.21** (+540%) |
| 所有 56 次 L1 runs 均值 | **32.44** |
| 所有 runs 中位数 | **35.31** |
| L1 第一轮 patch 接受率 | 2/5 (40%) |
| L2 spawned L1 总 patch 接受率 | 58/100 (58%) |

---

## 一、L1 第一轮（直接优化 L0）

**Baseline F1: 8.78 → 最终 F1: 33.40 (+280%)**
**共 14 步, 3 次 analyze, 5 次 propose_patch, 2 次 ACCEPTED**

### 1.1 诊断出的 Patterns

| # | Pattern | 影响组件 | 严重度 | 样本数 |
|---|---------|---------|--------|--------|
| P1 | QuestionRewriter 过度标记 `[AMBIGUOUS]`，导致下游放弃回答转为"请求澄清" | question_rewriter → info_extractor → retriever（级联失败） | high | 3→12→20 |
| P2 | AnswerGenerator 输出冗长、模棱两可的回答，而非直接 yes/no | AnswerGenerator + prompts.py | high | 3→11→15 |
| P3 | QuestionRewriter 输出被 markdown/注释/解释"污染"，InfoExtractor 把噪声当关键词提取 | QuestionRewriter → InfoExtractor | high | 6→18 |
| P4 | InfoExtractor 提取的关键词太泛（如 'NBA player, home arena' 而非 'Jalen Jones'），导致检索失败 | InfoExtractor | medium | 2 |
| P5 | Retriever 对隐式实体关系无法桥接 | Retriever | medium | - |

### 1.2 Patches 详情

#### Patch 1 (Step 3→4): REJECTED 8.78 → 0.00

- **prompts.py - INFO_EXTRACTOR**: 移除 "If the query is marked [AMBIGUOUS], extract alternative keyword interpretations"，改为 "Always extract concrete entity names and proper nouns"
- **失败原因**: diff 格式错误导致文件损坏，评估得分归零

#### Patch 2 (Step 5→6): ACCEPTED 8.78 → 15.35 (+6.57)

- **prompts.py - QUESTION_REWRITER**: 重写 prompt，限制只在"内部逻辑矛盾/语法残缺"时才标 `[AMBIGUOUS]`，不再对"隐含上下文"的问题标记
- **prompts.py - ANSWER_GENERATOR**: 重写 prompt，明确要求 yes/no 问题只回答 'yes'/'no'，禁止模棱两可和请求澄清

#### Patch 3 (Step 8→9): ACCEPTED 15.35 → 33.40 (+18.05)

- **prompts.py - QUESTION_REWRITER**: 强化 "只输出改写后的问题本身，不要 markdown/解释/建议"
- **prompts.py - ANSWER_GENERATOR**: 进一步强化 "不要说信息不足/不确定，禁止 markdown 格式"
- **components.py - QuestionRewriter.forward**: 降低 temperature 0.7→0.3，增加后处理清理 markdown 伪影
- **config.py**: 全局默认 temperature 0.7→0.5

#### Patch 4 (Step 11→12): REJECTED 33.40 → 0.00

- 尝试在 QUESTION_REWRITER prompt 中增加 few-shot examples，强化输出清理正则
- 失败原因: 过度修改导致系统崩溃

#### Patch 5 (Step 13→14): REJECTED 33.40 → 0.00

- 尝试在 AnswerGenerator 增加 yes/no 后处理提取逻辑，以及更激进的 QuestionRewriter 清理
- 失败原因: 多处同时修改导致兼容性问题

---

## 二、L2 优化 L1（Meta-Optimizer）

**L2 Baseline (L1 子优化器均值): 41.36**
**L2 共完成 4 步（observe ×3, analyze ×1），未能推进到 propose_patch 阶段**

### 2.1 L2 诊断出的 Pattern

| # | Pattern | Root Cause | 严重度 |
|---|---------|-----------|--------|
| P1 | L2 子优化器收到伪造数据集，导致所有 patches 被拒 | `planner/executors.py` 的 `_run_child_optimizer` 创建了硬编码的 `SimpleNamespace(question='opt_run_1', answer='100')` 假数据集传给子优化器，子优化器在假数据上评估 patch，自然全部失败 | high |

### 2.2 L2 建议的修复

> 将假数据集替换为从原始 dataset 中采样的真实问题子集，让子优化器在真实数据上评估 patch 效果。

### 2.3 L2 卡住的原因

L2 被 guardrail 拦截在 observe 阶段——"intermediate coverage 0% < 95%"，要求先收集/修复轨迹才能进入 patch 阶段。这导致 L2 反复 spawn L1 收集数据（共 55 个 L1 children），消耗大量算力但无法推进到 propose_patch。

---

## 三、L2 Spawned 的 L1 Children 汇总

**总计 56 次 L1 运行，58 个 ACCEPTED / 42 个 REJECTED（接受率 58%）**

### 3.1 高分代表性 Runs

#### 最高分: F1 = 56.21（run 042633）

**诊断 Patterns:**

1. AnswerGenerator 输出冗长解释而非直接回答 (x3→x8)
2. QuestionRewriter 输出 markdown/解释性注释 (x2→x4)
3. InfoExtractor 对 'who is' 问题无法提取实体 (x1→x4)
4. HintGenerator 输出 markdown 格式影响下游 (x3)

**有效 Patches:**

- **Patch 1 (14.70→37.00)**: 修改 `prompts.py` ANSWER_GENERATOR，强化直接回答要求
- **Patch 2 (37.00→56.21)**: 重写 ANSWER_GENERATOR 为极简规则格式 — `"RULES: 1. Output ONLY the answer. 2. For yes/no: output only 'yes' or 'no'. 3. For factual: output ONLY the specific entity/name/number"`

#### 第二高: F1 = 45.36（run 033052）

**诊断 Patterns:**

1. InfoExtractor 生成泛化关键词而非具体实体 (x1→x5)
2. QuestionRewriter 过度标记 [AMBIGUOUS] (x1→x3)
3. HintGenerator 输出不确定/模棱两可的 hints (x4)

**有效 Patches:**

- **Patch 1 (14.27→41.25)**: 同时修改 QUESTION_REWRITER / INFO_EXTRACTOR / ANSWER_GENERATOR 三个 prompt
- **Patch 2 (41.25→45.36)**: 进一步强化 QUESTION_REWRITER（大写 "ONLY"），强调只在逻辑矛盾时标 [AMBIGUOUS]

#### 第三高: F1 = 45.36（run 055435）

**有效 Patches:**

- **Patch 1 (14.70→37.41)**: 修改 `components.py`，在 QuestionRewriter 和 AnswerGenerator 的 `forward()` 里增加输出清理逻辑（strip markdown / 提取首行）
- **Patch 2 (37.41→45.36)**: 修改 QUESTION_REWRITER prompt 加 "CRITICAL:" 前缀强调

#### 其他高分: F1 = 40.67（run 043601）

**有效 Patches:**

- **Patch 1 (14.70→30.67)**: 修改 `components.py` AnswerGenerator，增加输出后处理（strip markdown，提取纯文本）
- **Patch 2 (30.67→40.67)**: 修改 QUESTION_REWRITER prompt，将 [AMBIGUOUS] 限制为仅"逻辑矛盾"，并在 `components.py` QuestionRewriter.forward 里增加清理逻辑

### 3.2 L1 Children 共性规律

所有高分 run 的核心成功模式都围绕三板斧：

1. **AnswerGenerator prompt 极简化** — 从模糊的"给 hints 回答问题"改为极简规则（yes/no 只输出 yes/no，factual 只输出实体名）→ **贡献最大增量**（平均 +15~20 F1）
2. **QuestionRewriter 限制 [AMBIGUOUS] 标记** — 只对真正逻辑矛盾的标记，隐含上下文不标 → **消除级联失败**（平均 +5~10 F1）
3. **temperature 降低 + 输出后处理** — 降温减少冗余输出，代码层面 strip markdown → **稳定性提升**

### 3.3 尚未解决的 Patterns

以下问题在多数 runs 中被识别但未被有效修复：

- **InfoExtractor 泛化关键词**: 对隐式实体无法推断具体名称（如问 "who wrote about X" 时无法提取作者名）
- **HintGenerator 传递不确定性**: 输出 "retrieved content does not answer the question"，导致 AnswerGenerator 拒绝回答
- **Retriever 多跳桥接失败**: 单次检索无法完成需要多跳推理的问题

---

## 四、数据文件位置

| 文件 | 路径 |
|------|------|
| L1 第一轮 trace | `.noa_cache/planner_traces/l1_hotpotqa_rag_20260304_023657.jsonl` |
| L2 trace | `.noa_cache/planner_traces/l2_noa_20260304_030110.jsonl` |
| L2 spawned L1 traces (55个) | `.noa_cache/planner_traces/l1_hotpotqa_rag_20260304_0*.jsonl` |
| 运行快照 | `.noa_runs/20260304_023651/` |
| 配置文件 | `configs/hotpotqa_nested.json` |

---

## 五、Bug 修复记录

本次运行前修复了两个阻塞性 bug：

1. **`retriever.py` — Wikipedia API 限流导致大量空响应**
   - 增加线程安全的 100ms 最小请求间隔
   - 增加指数退避重试（最多 3 次，0.5s × 2^attempt）
   - 增加 `json.JSONDecodeError` 和 `requests.RequestException` 异常处理

2. **`noa/tools/probe.py:68` — LLM 传入字符串导致 dict() 崩溃**
   - `dict(inputs)` → `inputs if isinstance(inputs, dict) else {"input": inputs}`
