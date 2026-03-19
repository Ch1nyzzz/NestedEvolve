# Gemini Suggestion: Prompt-MAML 重构方案

## Background: 与 EvoX 的本质差异

EvoX 的核心就是两层循环：内层进化 solution，外层在搜索停滞时基于当前 population state 和过去策略表现去生成/替换 search strategy；它强调的是 **task-time、run-time 的在线策略演化**，目标是在固定预算内把当前任务做得更好。

我们真正想做的，和 EvoX 拉开差距的关键，不是"也有一个外层"，而是要把问题改成：

> **跨任务地训练一个可迁移的 evolver prior / optimizer，使它在没见过的 OOD 任务上，用很少的适应预算就能更快收敛或达到更高性能。**

这就更接近 MAML 的精神。MAML 不是只在单个任务里在线改，而是 **在任务分布上训练一个初始化**，让模型在新任务上经过很少几步更新就能快速适应；它优化的是"容易适应"这件事，而不是只把当前训练任务跑好。这是我们的算法想要做的事情。

## 当前痛点

单靠参数化 φ 实现计算，优化之后映射回去来完成 meta optimize 是不现实的：

1. 没有那么多的计算量来计算多个任务百次迭代
2. 最终 φ 的变化不一定大，因为 evolve 策略可能本来就一般，导致最终结果没有太大变化

**转换方向：从参数化 φ 转到 LLM 层面的 prompt。** 通过 LLM 的能力来做到一个 meta 的优化器来提高任意 evolver 的泛化性（提升最高性能、减少运行迭代次数等）。

## 概念映射：传统 MAML → NestedEvolve (Prompt-MAML)

| 传统 MAML 概念 | NestedEvolve (Prompt-MAML) 概念 |
|---|---|
| 初始化参数 θ | **全局进化先验 Φ (Meta-Prompt)**：一段指导内层 LLM "如何去高效搜索/演化"的 System Prompt。包含：高级启发式搜索规则、变异算子库、跨任务避坑经验。 |
| Inner Loop (内层梯度下降) | **Fast Adaptation (极少步数的快速适应)**：给定新任务 T_i 和当前的 Φ，让 LLM 在 **极小的预算内（如 3~5 代）** 快速生成并迭代该任务的专属解。 |
| Loss 函数 / 梯度 | **演化轨迹 (Trajectory) 与文本梯度 (Textual Gradient)**：记录这 3~5 代的"尝试→反馈→再尝试"过程。文本梯度就是 Meta-LLM 对"为什么没能快速涨分"的语义总结。 |
| Outer Loop (外层更新) | **Semantic Meta-Update (语义元反思与重写)**：跨任务收集内层轨迹，用最高阶的 Meta-LLM 分析 Φ 的盲区，直接在文本层面重写并升级 Φ′。 |

---

## 四步执行计划（重构指南）

### 步骤 1：重构"进化器先验" (Redefine the Meta-State)

**涉及文件：** `meta_evolve/state.py`, `meta_evolve/strategy.py`

彻底抛弃浮点数/权重数组。将外层优化的对象 Φ 定义为一份结构化的 Markdown 文本（它本质上是教 LLM 如何成为一个顶尖的算法工程师）。

初始的 Φ_0 应该包含：

- **Role:** "你是一个顶尖的演化优化器，目标是用最少的步数找到最优解。"
- **Core Heuristics (核心搜索法则)：** 例如 "当连续两代适应度停滞时，必须放弃局部词汇微调，触发激进的全局语义重写。"
- **Meta Memory (元经验库)：** 初始为空，留给外层循环慢慢往里面填写跨任务的"避坑指南"。

### 步骤 2：截断内层循环 (Inner Loop - Fast Adaptation)

**涉及文件：** `meta_evolve/evolve_loop.py`, `noa/trajectory_store.py`

这是解决"算力不够"痛点的关键！既然是 MAML，内层绝不需要跑上百代去求极值。

1. **极小预算试跑：**
   - 从训练集抽取一个小 Batch 的任务（如 1 个文本摘要、1 个数学推理）
   - 将 Φ 实例化为内层 Evolver
   - 在每个任务上，严格限制只迭代 K 次（例如 K=3）

2. **收集演化轨迹 (Trajectory)：**
   - 详细记录这 K 步经历了什么：
   ```
   [(Gen1: 使用策略A -> 分数0.2); (Gen2: 发现格式错误，使用策略B修复 -> 分数0.8)]
   ```

3. **计算 Meta-Fitness：**
   - 评估 Φ 好坏的标准不再是绝对得分，而是前 K 步的学习曲线面积（AUC）或 ΔScore
   - 我们要奖励的是"冷启动提分快"

### 步骤 3：外层语义元更新 (Outer Loop - Textual Meta-Update)

**涉及文件：** `meta_evolve/meta_learner.py`, `meta_evolve/semantic_analyzer.py`

这是拉开与传统算法差距的核心护城河！用 LLM 代替求导，进行"文本梯度下降"。

1. **轨迹压缩 (Semantic Analysis)：**
   - 用 LLM 把内层繁杂的 JSON 日志浓缩成精简的失败/成功总结

2. **元反思 (Meta-Reflection)：**
   - 把总结喂给最聪明的模型（如 GPT-4o / Claude-3.5）
   - Prompt 示例：
   > "作为 Meta-Optimizer，这是当前的进化先验 Φ。在任务A上，轨迹显示前3步提分迅速；但在任务B上，它陷入了死循环。请分析 Φ 中缺失了什么通用法则导致了任务B的失败？"

3. **准则进化 (Meta-Rewrite)：**
   - Meta-LLM 会根据反思重写 Φ
   - 例如自动在 Memory 中加上一条："新规则：每次变异前，必须强制校验上一次报错的边界条件是否被解决。"
   - 这种自然语言层面的更新，比把 `mutation_rate` 调高 0.1 带来的行为改变要剧烈且有效一万倍

### 步骤 4：证明 OOD 泛化能力 (Zero-shot Transfer / Meta-Testing)

**涉及文件：** `scripts/run_stark_prime.py`, `target_systems/`

这是要写在 Paper 里的核心 Storyline：

1. **冻结外层：** 经过几十个简单任务的 Outer Loop 洗礼后，你得到了一个饱经风霜、见多识广的终极 Φ*。

2. **降维打击 Baseline：** 将 Φ* 直接用于完全没见过的、极度复杂的真实业务任务（如 OOD 任务 STaRK-Prime 或 PubMedQA）。

3. **见证奇迹：** 对比未经 Meta 训练的普通 LLM 优化器（它可能要像无头苍蝇一样盲目试错 20 步才能摸到门道），你会发现你的 Φ* 就像一个经验丰富的老手，仅仅在第 1、2 步就能精准定位新任务的瓶颈并实现分数垂直跃升。这完美证明了你的算法大大减少了迭代次数和运行成本。

---

## 下一步 Action 建议

1. **大刀阔斧地删减：** 清理掉 `state.py` 和 `surrogate.py` 中用于数值特征提取、高斯映射等不再需要的传统 EA 数学代码。

2. **锁死代数：** 先把 `evolve_loop` 的 `max_generations` 强行设为 3。

3. **跑通最小飞轮：** 挑两个最简单的 Toy 任务（比如几秒钟就能评测的字符串匹配或情绪分类）。手工写好 Meta-Learner Prompt，强行跑通一次闭环：
   > 内层跑3步 → 收集轨迹 → 外层 LLM 反思并重写 Φ → 将新 Φ 喂给下一个任务
