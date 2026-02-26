# NOA (Nested Optimization Agents) 项目进展报告

> **日期**: 2026-02-26
> **项目名称**: NestedEvolve — 基于嵌套优化 Agent 的 AI 系统自动改进框架

---

## 一、项目概述

### 1.1 研究目标

本项目提出 **NOA (Nested Optimization Agents)**，一个两层嵌套优化框架，旨在利用大语言模型 (LLM) **自动分析、诊断和修复 AI 系统的代码缺陷**。核心思想是：

- **L1 优化器**：读取目标系统源码 → 运行并观察失败案例 → LLM 诊断根因 → LLM 生成代码补丁 → 沙箱评估 → 择优保留
- **L2 元优化器**：当 L1 优化饱和时，L2 将 L1 优化器自身作为优化目标，分析 L1 的历史记录，改进优化框架的代码（如 Analyzer 的 prompt、Evaluator 的逻辑等），从而让下一轮 L1 更有效

### 1.2 实验对象

以 **HotpotQA 多跳问答** 的 RAG (Retrieval-Augmented Generation) 管线为目标系统（L0），评估指标为 **Token-level F1 Score**。

### 1.3 核心创新

1. **嵌套优化**：L2 优化 L1 的代码，L1 优化 L0 的代码，形成自我改进闭环
2. **代码级修改**：不同于传统 prompt tuning，NOA 直接生成 SEARCH/REPLACE 代码补丁
3. **沙箱隔离**：所有修改在临时目录中验证，仅在分数提升时才正式提交
4. **自动适配**：通过 LLM 自动生成目标系统的适配器（AutoAdapter），无需手动编写接口代码

---

## 二、系统架构

### 2.1 三层结构

```
┌─────────────────────────────────────────────────────────┐
│                    L2 元优化器 (Orchestrator)             │
│  分析 L1 的优化历史 → 修改 noa/ 框架代码 → 让 L1 更强    │
│  文件: noa/orchestrator.py                               │
├─────────────────────────────────────────────────────────┤
│                    L1 优化器 (NOptimizer)                 │
│  I-O-A-O-E 循环: 读源码→运行→诊断→补丁→评估              │
│  文件: noa/engine.py                                     │
├─────────────────────────────────────────────────────────┤
│                    L0 目标系统 (RAG Pipeline)             │
│  QuestionRewriter → InfoExtractor → Retriever            │
│  → HintGenerator → AnswerGenerator                       │
│  文件: target_systems/hotpotqa_rag/                      │
└─────────────────────────────────────────────────────────┘
```

### 2.2 目录结构

```
NestedEvolve/
├── noa/                            # 核心优化框架
│   ├── core/
│   │   ├── protocol.py             # 数据结构定义 (Trajectory, Diagnosis, DeltaPatch 等)
│   │   └── prompts.py              # 所有 LLM 提示词模板
│   ├── stages/                     # I-O-A-O-E 五个阶段
│   │   ├── initiator.py            # I - 源码内省
│   │   ├── observer.py             # O - 运行观察
│   │   ├── analyzer.py             # A - 失败诊断
│   │   ├── optimizer.py            # O - 补丁生成
│   │   └── evaluator.py            # E - 沙箱评估
│   ├── engine.py                   # L1 优化主循环 (NOptimizer)
│   ├── orchestrator.py             # L1+L2 嵌套编排 (Orchestrator)
│   ├── auto_adapter.py             # LLM 动态生成适配器
│   ├── diff_utils.py               # 补丁解析与应用
│   ├── workspace.py                # 工作空间隔离管理
│   └── subprocess_runner.py        # 子进程隔离执行
├── target_systems/
│   └── hotpotqa_rag/               # L0 目标: HotpotQA RAG 管线
│       ├── pipeline.py             # 主管线 (5 组件串行)
│       ├── components.py           # 各组件实现
│       ├── retriever.py            # 多种检索器后端
│       ├── evaluate.py             # F1 评估函数
│       └── prompts.py              # 管线默认提示词
├── scripts/                        # 入口脚本
│   ├── run_baseline.py             # 基线评估
│   ├── run_l1.py                   # 仅 L1 优化
│   └── run_nested.py               # L1+L2 嵌套优化 (主入口)
├── utils/
│   ├── llm.py                      # LLM 调用封装 (litellm)
│   └── data.py                     # HotpotQA 数据加载
├── configs/
│   └── hotpotqa_default.json       # 管线默认配置
└── results*.json                   # 实验结果 (6 个版本)
```

---

## 三、各组件详解

### 3.1 L0 目标系统：HotpotQA RAG 管线

RAG 管线由 5 个组件串行组成，每个组件接收上一步的输出作为输入：

```
Question
   ↓
┌──────────────────┐
│ QuestionRewriter  │  改写问题，使其更适合检索
│ (gpt-4o-mini)    │  输入: question → 输出: rewritten_query
└──────────────────┘
   ↓
┌──────────────────┐
│ InfoExtractor     │  从问题中提取检索关键词
│ (gpt-4o-mini)    │  输入: rewritten_query → 输出: search_keywords
└──────────────────┘
   ↓
┌──────────────────┐
│ Retriever         │  从外部知识源检索相关段落
│ (wiki_semantic)   │  输入: search_keywords → 输出: retrieve_content
│                   │  后端: WikiSemantic / Local / Wikipedia / ColBERT / Hybrid
└──────────────────┘
   ↓
┌──────────────────┐
│ HintGenerator     │  根据检索内容生成答题提示
│ (gpt-4o-mini)    │  输入: rewritten_query + retrieve_content → 输出: hints
└──────────────────┘
   ↓
┌──────────────────┐
│ AnswerGenerator   │  根据提示生成最终答案
│ (gpt-4o-mini)    │  输入: rewritten_query + hints → 输出: answer
└──────────────────┘
   ↓
Answer (与 ground_truth 计算 F1)
```

**评估方式** (`evaluate.py`):
- `f1_score(prediction, ground_truth)`: Token 级别 F1，对 yes/no 等特殊答案有专门处理
- `normalize_answer()`: 小写化、去冠词/标点
- `evaluate_batch()`: 并行批量评估，返回平均 F1

### 3.2 L1 优化器：NOptimizer (I-O-A-O-E 循环)

NOptimizer (`noa/engine.py`) 实现了 **Initiate-Observe-Analyze-Optimize-Evaluate** 五阶段循环：

#### 阶段 1: Initiate（内省）
- **功能**: LLM 阅读目标系统全部源码，提取系统描述
- **文件**: `noa/stages/initiator.py`
- **输出**: `SystemDescription` (workflow_summary, component_names, source_files, baseline_score)
- **使用 LLM**: 是

#### 阶段 2: Observe（观察）
- **功能**: 从数据集采样 N 条，运行目标系统，收集执行轨迹
- **文件**: `noa/stages/observer.py`
- **输出**: `list[Trajectory]` (question, prediction, ground_truth, f1, intermediate)
- **使用 LLM**: 否（仅执行目标系统）

#### 阶段 3: Analyze（分析）
- **功能**: LLM 阅读源码 + 失败轨迹（含中间输出），诊断根因
- **文件**: `noa/stages/analyzer.py`
- **输出**: `Diagnosis` (failure_patterns: [{pattern, root_cause, severity, affected_component, affected_file, suggested_fix}])
- **关键逻辑**: 以中位 F1 为阈值划分失败/成功样本；展示失败样本 + 少量成功样本做对比
- **使用 LLM**: 是

#### 阶段 4: Optimize（优化）
- **功能**: 对每个失败模式，LLM 生成 SEARCH/REPLACE 代码补丁
- **文件**: `noa/stages/optimizer.py`
- **输出**: `DeltaPatch` (diffs: [DiffBlock(file_path, search, replace)], rationale)
- **关键约束**: SEARCH 必须精确匹配原始代码；优先修改 prompt/配置而非逻辑；每个补丁针对一个问题
- **使用 LLM**: 是

#### 阶段 5: Evaluate（评估）
- **功能**: 在沙箱中应用补丁并测试效果
- **文件**: `noa/stages/evaluator.py`
- **流程**:
  1. 在内存中应用 diff → 修改后的文件列表
  2. 写入临时目录（完整复制 + 覆盖修改）
  3. 从临时目录加载全新目标实例
  4. 在数据子集上评估 F1
  5. 若分数提升 → 接受补丁，写回源码；否则 → 拒绝
- **使用 LLM**: 否（仅执行与比较）

#### L1 完整流程

```
Initiate → SystemDescription
     ↓
Observe → Trajectories + baseline_score
     ↓
╔═══════════════ 循环 (最多 max_iterations 次) ═══════════════╗
║                                                              ║
║  Analyze → Diagnosis (failure_patterns)                      ║
║     ↓                                                        ║
║  对每个 pattern (按严重度排序):                                ║
║     Optimize → DeltaPatch                                    ║
║        ↓                                                     ║
║     Evaluate → 接受 or 拒绝                                  ║
║        ↓                                                     ║
║     若接受: 更新源码, 重新加载目标, 更新 sys_desc              ║
║                                                              ║
║  停止条件:                                                    ║
║  - 达到 max_iterations                                       ║
║  - 连续 2 个循环无接受补丁                                    ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
     ↓
返回: final_score, history, iterations, accepted
```

### 3.3 L2 元优化器：Orchestrator

Orchestrator (`noa/orchestrator.py`) 在 L1 基础上增加了 L2 层：

```
工作空间初始化 (复制源码到 .noa_runs/<id>/workspace)
     ↓
╔═══════════════ L2 轮次循环 (max_l2_rounds) ═══════════════╗
║                                                             ║
║  ① 运行 L1 (子进程)                                         ║
║     ↓                                                       ║
║  ② 快照保存                                                  ║
║     ↓                                                       ║
║  ③ 若 L1 有改进 → 启动 L2 元优化:                             ║
║     ├─ Meta-Analyze: LLM 分析 noa/ 源码 + L1 历史            ║
║     │  (为什么某些诊断失败？为什么补丁无效？)                    ║
║     ├─ 对每个元级问题:                                        ║
║     │   ├─ Optimize: LLM 生成 noa/ 代码补丁                  ║
║     │   └─ Evaluate: 运行完整 L1 (子进程) 作为评估             ║
║     │       若 L1 分数提升 → 接受 noa/ 补丁                   ║
║     └─ 快照保存                                               ║
║                                                             ║
║  停止条件:                                                    ║
║  - 达到 max_l2_rounds                                       ║
║  - 连续 2 轮无进展                                            ║
╚═════════════════════════════════════════════════════════════╝
     ↓
返回: rounds 历史, snapshots, final_score
```

### 3.4 支撑组件

#### AutoAdapter (`noa/auto_adapter.py`)
- LLM 扫描目标系统源码，自动生成 `AutoAdapter` 包装类
- 统一接口: `adapter(question) → AdapterResult(answer, intermediate)`
- 模块隔离: 清理 `sys.modules` 缓存，确保从不同目录加载修改后的代码时不会使用旧版本
- 缓存: 生成后保存为 `_noa_adapter.py`，后续直接加载

#### Diff 工具 (`noa/diff_utils.py`)
- `extract_diffs()`: 从 LLM 输出中解析 SEARCH/REPLACE 块
- `apply_diffs_in_memory()`: 内存中应用补丁（先精确匹配，后行级回退）
- `write_to_temp_dir()`: 写入临时沙箱目录
- `commit_to_source()`: 仅在评估通过后写回原始源码

#### 工作空间管理 (`noa/workspace.py`)
- 将 `noa/` 和目标系统源码复制到 `.noa_runs/<run_id>/workspace`
- 快照功能：每轮 L1/L2 后保存完整副本，支持恢复
- **核心保障**: 原始代码永不被修改

#### 子进程运行器 (`noa/subprocess_runner.py`)
- L1 在独立 Python 子进程中运行
- 避免模块缓存污染（`auto_adapt` 多次加载不同版本代码）
- L2 评估时使用 `isolate_source=True`，将 L0 源码复制到临时目录

---

## 四、Agent 端到端工作流程

以 `run_nested.py` 为入口的完整工作流：

```
1. 加载数据
   load_hotpotqa(n=50)  →  50 条 QA 样本 (固定 seed=42)

2. 生成适配器
   auto_adapt(source_dir)  →  LLM 生成 AutoAdapter 包装类

3. 初始化 Orchestrator
   Orchestrator(source_dir, target_factory, dataset, eval_fn, score_fn,
                l1_max_iterations, l2_max_iterations, max_l2_rounds, ...)

4. 运行嵌套优化
   orchestrator.run()
   ├─ WorkspaceManager.setup()  →  创建隔离工作空间
   ├─ Round 0:
   │   ├─ run_l1_subprocess()   →  L1 优化 (子进程)
   │   ├─ snapshot("after_l1_round_0")
   │   ├─ _run_l2()             →  L2 元优化
   │   │   ├─ meta_analyze()    →  诊断 noa/ 框架问题
   │   │   ├─ optimize()        →  生成 noa/ 代码补丁
   │   │   └─ evaluate (run_l1_subprocess)  →  完整 L1 作为评估
   │   └─ snapshot("after_l2_round_0")
   ├─ Round 1:
   │   ├─ run_l1_subprocess()   →  使用改进后的 noa/ 运行 L1
   │   └─ ...
   └─ 返回结果

5. 保存结果
   → results_v*.json
```

### 关键设计决策

| 决策 | 原因 |
|------|------|
| 子进程执行 L1 | 避免 Python 模块缓存导致加载旧版代码 |
| 工作空间隔离 | 保护原始代码，支持回滚 |
| 内存中应用补丁 | 无副作用，失败可安全丢弃 |
| 补丁逐个评估 | 隔离每个修改的效果，避免相互干扰 |
| 按严重度排序处理 | 优先修复影响最大的问题 |
| 连续 2 轮无进展则停止 | 避免无效迭代浪费资源 |

---

## 五、实验结果与分析

### 5.1 总体结果

| 版本 | Baseline | Final Score | 提升 | 提升率 | 轮次 | L1 接受 | L2 接受 | 运行日期 |
|------|----------|-------------|------|--------|------|---------|---------|----------|
| v1 (results.json) | 38.02 | 44.45 | +6.43 | +16.9% | 1 (L1) | 1 | - | 02-23 |
| v2 | 36.11 | 52.53 | +16.42 | +45.5% | 1 (L1) | 2 | - | 02-24 |
| v3 | 31.91 | 51.47 | +19.56 | +61.3% | 2 (L1+L2) | 1 | 1 | 02-24 |
| v4 | 46.01 | 48.36 | +2.35 | +5.1% | 2 (L1+L2) | 3 | 0 | 02-25 |
| v5 | 27.87 | 45.19 | +17.32 | +62.2% | 2 (L1+L2) | 2 | 1 | 02-25 |
| **v6** | **38.88** | **58.00** | **+19.12** | **+49.2%** | **2 (L1+L2)** | **3** | **1** | **02-26** |

> 注：各版本 Baseline 不同是因为每次运行时 Observe 阶段的采样和 Retriever 返回结果存在随机性。

### 5.2 各版本详细分析

#### Version 1 (results.json) — 首次运行，仅 L1

**最终分数**: 38.02 → **44.45** (+6.43)

运行了 4 轮 L1 迭代，共 1 个补丁被接受：

| 迭代 | 尝试修改 | Before | After | 结果 |
|------|---------|--------|-------|------|
| 1 | Retriever 空内容处理 + AnswerGenerator 简洁 prompt + InfoExtractor 关键词清洗 | 38.02 | 32.94 | 拒绝 |
| 2 | Retriever colbert 后端 + 空 passage 过滤 + prompt 改进 | 38.02 | **44.45** | **接受** |
| 3 | 同类修改（Retriever + prompt） | 44.45 | 41.08 | 拒绝 |
| 4 | AnswerGenerator prompt + InfoExtractor 换行处理 | 44.45 | 43.83 | 拒绝 |

**关键发现**: 第一次成功的修改是对 Retriever 添加了 colbert 后端显式处理和空 passage 过滤。后续迭代尝试进一步优化但均未超过已接受的分数。

---

#### Version 2 (results_v2.json) — L1 调优

**最终分数**: 36.11 → **52.53** (+16.42)

运行了 5 轮 L1 迭代，2 个补丁被接受：

| 迭代 | 关键修改 | Before | After | 结果 |
|------|---------|--------|-------|------|
| 1 | Retriever 空 passage 处理 + InfoExtractor 清洗 + AnswerGenerator prompt | 36.11 | 37.72 | 接受 |
| 2 | Retriever 显式 colbert 后端 | 37.72 | 39.02 | 接受 |
| 3-5 | 各种 prompt 和后处理调整 | - | - | 全部拒绝 |

**进步**: 相比 v1，实现了更大的提升。两次小幅改进累积带来了显著效果。

---

#### Version 3 (results_v3.json) — 首次引入 L2

**最终分数**: 31.91 → **51.47** (+19.56)

这是首次运行 L1+L2 嵌套优化的版本。

**Round 0 - L1**: Baseline 31.91，1 轮迭代，0 个补丁被接受（Retriever 回退策略导致分数下降至 23.78）

**Round 0 - L2**: L2 诊断发现 L1 子进程运行失败的根本原因
- **关键 L2 修复**:
  1. `subprocess_runner.py`: 修复 evaluate.py 导入验证逻辑
  2. `workspace.py`: 确保 evaluate.py 被复制到工作空间
  3. `auto_adapter.py`: 改进错误追踪
- L2 补丁 Before: 28.72 → After: **43.12** (接受)

**Round 1 - L1**: 使用 L2 改进后的框架重新运行
- Baseline 44.55 → 迭代 2 接受 Retriever 空内容处理 → **51.47**

**关键意义**: **L2 首次发挥了关键作用**——发现并修复了框架基础设施层面的 bug（子进程的 evaluate.py 加载失败），这是 L1 自身无法解决的问题。

---

#### Version 4 (results_v4.json) — 多模式尝试

**最终分数**: 46.01 → **48.36** (+2.35)

**Round 0 - L1**: 5 轮迭代，1 个补丁被接受
- InfoExtractor 清除 "Keywords:" 前缀 → 46.01→52.16（接受）
- 后续 Retriever fallback、AnswerGenerator prompt 等均拒绝

**Round 0 - L2**: 2 轮迭代，0 个补丁被接受
- 尝试改进 Analyzer prompt 和 Optimizer diff 约束，但均未带来提升

**Round 1 - L1**: 5 轮迭代，2 个补丁被接受
- Retriever 空内容 fallback + LocalRetriever 返回 top-k → 34.63→45.43（接受）
- InfoExtractor regex 归一化 → 45.43→48.36（接受）
- 后续包括 AnswerGenerator guard clause (20.0!) 等大幅下降

**教训**: 过于激进的修改（如 AnswerGenerator 添加 guard clause 返回默认答案）反而导致严重性能下降，因为很多问题的 hints 虽然不完美但仍然包含有用信息。

---

#### Version 5 (results_v5.json) — L2 改进 Analyzer

**最终分数**: 27.87 → **45.19** (+17.32)

**Round 0 - L1**: 5 轮迭代，1 个补丁被接受
- Retriever 空内容 fallback → 27.87→41.87（接受）
- InfoExtractor 归一化、AnswerGenerator/HintGenerator guard clause 等均拒绝

**Round 0 - L2**: 1 轮迭代，1 个补丁被接受
- **L2 关键修改**: 修改 `stages/analyzer.py` 的 `analyze()` 函数，将失败样本展示从默认的全部改为最多 10 个，成功样本从 3 个增加到 5 个
- Before: 41.87 → After: **49.65**（接受）

**Round 1 - L1**: 4 轮迭代，1 个补丁被接受
- AnswerGenerator prompt 简洁化 + 后处理取第一段 → 41.52→**45.19**（接受）

**关键发现**: L2 通过调整 Analyzer 的样本展示策略，让 L1 在下一轮中获得了更好的诊断结果。

---

#### Version 6 (results_v6.json) — 最佳结果

**最终分数**: 38.88 → **58.00** (+19.12) ✨

**Round 0 - L1**: 5 轮迭代，1 个补丁被接受
- Retriever 添加 fallback_engine (WikipediaRetriever) 用于所有非 Wikipedia 后端 → 38.88→**50.90**（接受）
- HintGenerator/AnswerGenerator guard clause、InfoExtractor 清洗等均拒绝

**Round 0 - L2**: 2 轮迭代，1 个补丁被接受
- **L2 关键修改**: 增强 `ANALYZER_PROMPT`，添加指导语：
  > "Pay special attention to input validation issues, retriever backend errors, and silent fallback behaviors that may cause failures. Include subtle failure patterns related to configuration or data retrieval problems."
- L2 评估时 L1 分数达到 **72.17**（接受）

**Round 1 - L1**: 使用增强后的 Analyzer prompt 运行 L1
- 4 轮迭代，2 个补丁被接受：
  1. Retriever 增强 fallback（为所有后端添加 Wikipedia fallback）→ 44.41→49.95（接受）
  2. HintGenerator 空内容检测 → 49.95→**58.00**（接受）

**成功关键**:
1. L2 改进了 Analyzer 的 prompt，使其能发现更细微的失败模式
2. 增强的 Analyzer 在 Round 1 中诊断出了之前遗漏的 HintGenerator 空内容问题
3. 多层修复叠加：Retriever fallback + HintGenerator guard 的组合有效减少了空内容传播

---

### 5.3 性能趋势

```
Final Score 趋势:

 60 │                                                    ★ v6: 58.0
    │
 55 │
    │              ● v2: 52.53    ● v3: 51.47
 50 │
    │                                        ● v4: 48.36
 45 │  ● v1: 44.45                                       ● v5: 45.19
    │
 40 │
    │
 35 │
    ├──────────────────────────────────────────────────────────────→
      v1          v2          v3          v4          v5          v6
```

---

## 六、核心发现

### 6.1 反复出现的失败模式

**模式 1: Retriever 空内容 (出现频率: 6/6 版本, 最高严重度)**

所有版本都将其识别为首要问题。Retriever 在某些查询上返回空列表，导致下游 HintGenerator 和 AnswerGenerator 在没有任何上下文的情况下生成答案，引发幻觉。

解决方案演进：
- v1-v2: 简单 fallback 消息 `"[No relevant content found]"`
- v3-v4: 添加 WikipediaRetriever 作为 fallback 引擎
- v5: 保留 fallback 消息策略
- v6: **为所有后端统一添加 fallback 引擎** → 效果最好

**模式 2: InfoExtractor 关键词格式化 (出现频率: 6/6 版本, 中等严重度)**

LLM 输出的关键词格式不一致：有时包含 "Keywords:" 前缀、换行符、项目符号、分号等，传给 Retriever 后影响检索质量。

解决方案演进：
- v1: 逗号分割清洗
- v2-v3: 移除 "Keywords:" 前缀
- v4: 正则表达式归一化（换行→逗号，多余空格折叠）
- v5-v6: 行级处理，移除项目符号

**模式 3: AnswerGenerator 幻觉/冗长 (出现频率: 5/6 版本, 中等严重度)**

AnswerGenerator 在 hints 不足时仍然自信地生成错误答案；或生成过于冗长的答案导致 F1 下降。

**关键教训**: 添加 guard clause 返回默认答案（如 "Insufficient information"）**往往导致严重性能下降**（v4 迭代中从 48.36 降到 20.0）。因为即使 hints 不完美，LLM 仍可能从中提取有用信息。更好的策略是改进 prompt 要求简洁回答。

**模式 4: Retriever 后端选择 (出现频率: 4/6 版本, 低严重度)**

`config.backend` 为 "colbert" 时默认 fallback 到 `HybridRetriever`，行为不透明。多次尝试显式处理但效果不一。

### 6.2 L2 元优化的有效性分析

| 版本 | L2 修改内容 | L2 效果 |
|------|------------|---------|
| v3 | 修复 subprocess_runner 和 workspace 的基础设施 bug | **高**: 从 28.72→43.12，解决了框架级别问题 |
| v4 | 尝试改进 Analyzer prompt 和 Optimizer diff 约束 | **无**: 均未被接受 |
| v5 | 调整 Analyzer 的样本展示数量 (failures≤10, successes≤5) | **中**: 间接提升了后续 L1 的诊断质量 |
| v6 | 增强 ANALYZER_PROMPT，引导关注输入验证和后端错误 | **高**: L1 评估达到 72.17，最终帮助 L1 找到 HintGenerator 修复 |

**结论**: L2 在以下场景最有效：
1. **修复框架 bug** (v3): 最直接的效果
2. **改进 Analyzer prompt** (v6): 引导 L1 发现之前遗漏的问题模式
3. **调整算法参数** (v5): 微调但有间接效果

L2 的局限性：
- 对 Optimizer 的补丁生成能力提升有限（v4 中尝试失败）
- L2 生成的 diff 有时本身就是空的或无效的（模板占位符）

### 6.3 补丁接受率分析

跨所有版本的统计：

- **L1 总尝试补丁数**: ~50+
- **L1 接受补丁数**: 12
- **L1 接受率**: ~24%
- **L2 总尝试补丁数**: ~8
- **L2 接受补丁数**: 3
- **L2 接受率**: ~37.5%

大部分被拒绝的补丁是因为修改后分数反而下降，说明 LLM 的代码修改建议有较高的失败率，**沙箱评估机制是保障系统不退化的关键安全网**。

---

## 七、当前进展总结

### 7.1 已完成

1. **框架实现完整**: L0/L1/L2 三层架构全部可用，I-O-A-O-E 循环稳定运行
2. **6 轮实验**: 从纯 L1 (v1-v2) 到 L1+L2 (v3-v6) 的完整实验链
3. **最佳成绩**: v6 达到 F1=58.0，相比 baseline (38.88) 提升 49.2%
4. **L2 验证**: 证明了元优化在修复框架 bug 和改进分析 prompt 上的有效性
5. **关键工程**: 子进程隔离、工作空间管理、快照恢复等基础设施稳定

### 7.2 当前存在的问题

1. **补丁接受率低** (~24%): LLM 生成的代码修改多数不能提高性能，资源浪费较大
2. **Retriever 问题根深蒂固**: 每个版本都在修同一个问题，说明当前修复策略是"打补丁"而非根本解决
3. **评估不稳定性**: 不同运行的 baseline 变化较大 (27.87~46.01)，采样和检索的随机性影响了结果可比性
4. **L2 Optimizer 能力有限**: L2 生成的 noa/ 代码补丁有时是空的或使用模板占位符
5. **Guard clause 陷阱**: 添加输入验证/安全默认值反复导致性能大幅下降，系统需要更好地处理信息不完整的场景
6. **单一评估指标**: 仅依靠 F1 Score 可能不足以全面反映改进效果

### 7.3 下一步方向建议

1. **提高补丁质量**: 探索让 LLM 在生成补丁前先"模拟"修改效果的方法，或引入多候选方案排序
2. **稳定评估**: 增大评估样本量或固定更多随机种子，减少 baseline 波动
3. **深层 Retriever 改进**: 考虑在 L0 层面引入更好的检索策略，而非反复打补丁
4. **扩展目标系统**: 在其他 AI 系统（如代码生成、摘要等）上验证 NOA 的通用性
5. **L2 能力增强**: 给 L2 更多的优化操作空间（如修改超参数、调整循环策略等）
6. **消融实验**: 系统对比纯 L1 vs L1+L2 的效果差异，量化 L2 的贡献

---

## 附录

### A. 运行命令示例

```bash
# 基线评估
python scripts/run_baseline.py --n=50

# 仅 L1 优化
python scripts/run_l1.py --n=50 --iterations=5 --samples=50 --model=gpt-4.1-mini

# L1+L2 嵌套优化 (主入口)
python scripts/run_nested.py \
  --n=50 \
  --l1-iterations=5 \
  --l1-samples=50 \
  --l1-eval-samples=20 \
  --l2-iterations=2 \
  --l2-rounds=2 \
  --model=gpt-4.1-mini \
  --output=results_v6.json
```

### B. 核心数据结构

```python
# 执行轨迹
Trajectory(question, prediction, ground_truth, f1, intermediate)

# 诊断结果
Diagnosis(failure_patterns=[{pattern, root_cause, severity, affected_component, affected_file, suggested_fix}], summary, raw_analysis)

# 代码补丁
DeltaPatch(diffs=[DiffBlock(file_path, search, replace)], rationale)

# 评估结果
EvalResult(before_score, after_score, accepted, patch)
```

### C. 使用的 LLM

- **优化器 LLM**: gpt-4.1-mini (Analyzer, Optimizer, Initiator, AutoAdapter)
- **目标系统 LLM**: gpt-4o-mini (RAG 管线各组件)
