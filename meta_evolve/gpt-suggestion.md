# GPT Suggestion: 从"学参数"到"学干预策略"

## 核心思路转变

把外层从"学参数"改成"学干预策略"，把 prompt 当成 optimizer program，而不是 φ 的自然语言外壳。

## 文献支撑

- **Mass**：prompt 本身是高影响设计变量，适合先做局部 prompt warm-up，再做更大层级的 orchestration 优化
- **GEPA**：用执行 trace / evaluation trace 做自然语言反思，比纯 RL 更省 rollout
- **AdaEvolve**：把外层适配拆成局部搜索强度、全局资源分配、停滞时 meta-guidance 三层
- **Dr. Zero**：proposer-solver 式课程和分组归一化能在预算很小的时候维持有效自演化
- **SKILLRL / OpenSage**：跨任务迁移不能靠存 raw trajectory，而要靠蒸馏后的 skill / memory / tactic bank

---

## 凝练、可落地的计划

### 1. 先把"学什么"定义对

不要直接学一个 free-form prompt，也不要继续只学 φ。学这个：

```
M_ψ: (task_spec, run_summary, tactic_bank, budget) → action
```

其中 action 不是一句话，而是**类型化干预**：

| 干预类型 | 说明 |
|---|---|
| `set_phi` | 调低层 actuator |
| `rewrite_mutator_prompt` | 改 mutator prompt |
| `rewrite_selector_prompt` | 改 selector / evaluator prompt |
| `spawn_island` / `merge` / `reallocate_budget` | 改资源分配 |
| `inject_tactic` | 注入 tactic |
| `rollback_last_change` | 回滚 |

也就是说，现在的 `StrategyParams` 不要废掉，它应当退到执行层 actuator；真正要学的是**"什么时候打哪个 actuator"**。

### 2. 把外层输入做成一个"run summary card"

LLM controller 不该直接看整个 population。要给它一个压缩后的状态卡：

- **任务特征：** 任务 family、输入规模、是否可验证、是否多峰
- **当前群体状态：** best / mean / std、diversity、island 间差异
- **动态特征：** 最近 3~5 次 best 提升斜率、停滞长度、错误类型分布
- **当前策略：** φ、当前 mutator prompt id、最近几次 intervention
- **预算：** 剩余 step、剩余 LLM call、是否到 checkpoint

这一步特别关键。因为你要学的是 **state → action**，不是"根据感觉重写 prompt"。

### 3. 建一个跨任务的 tactic bank，而不是存轨迹

最容易做错的是把所有 run trace 都塞进上下文。那样会又贵又乱。更好的做法是像 SKILLRL / OpenSage 那样，把经验蒸馏成分层 tactic：

**两层 tactic：**

- **General tactics：** 任何 evolver 都能用
  - 例：停滞时增加探索、失败模式集中时先加分析再改代码
- **Family-specific tactics：** 对某类任务有效
  - 例：combinatorial task 先改候选生成；code task 先改 diff/rewrite 比例

**每条 tactic 存 5 个字段：**

1. 适用条件
2. 干预动作
3. 预期改善目标
4. 常见失败模式
5. 历史有效任务簇

这比 raw trajectory 更像"可迁移 prior"。

### 4. 训练时不要做长链反传，做"稀疏干预"的元数据集

不要做"每个任务 100 次 inner iteration × 很多任务 × 再反传"。改成这个数据生成方式：

1. 对每个训练任务，只跑一个很小预算的 inner loop
2. 只在少数 checkpoint 允许 meta-controller 干预：
   - step 0
   - step 5
   - 检测到停滞时
3. 在每个 checkpoint，采样 2~4 个候选干预动作
4. 用短视窗收益打标签：看接下来 3~5 step 的 gain

这样得到的数据是：

```
(state, action, Δbest, Δauc, cost)
```

训练的不是 end-to-end rollout gradient，而是一个 **intervention policy prior**。这会比"跨任务跑很多次 inner loop 再用 ES 调 φ"更像真正的 meta-learning，也便宜得多。

> **Dr. Zero 的启发：** 为了压算力，要做结构化分组归一化，别用一锅端的全局 baseline。应该按 task family / 难度 bucket 统计 intervention 的相对收益。

### 5. 用 GEPA 式方法优化 controller prompt — 本质上是"学 optimizer program"

Controller 本身也别拿梯度硬训。先用两层 prompt：

- **Proposer prompt：** 根据 state card 产出 action
- **Critic/reflection prompt：** 看 action 后的短期结果，反思为什么有效/无效

然后像 GEPA 那样做：

1. 基于 trace 的自然语言反思
2. Pareto / pool 保留多种 controller prompt 变体
3. 用 pairwise 方式比较"哪个 controller 在哪个 task family 更有效"

这样学出来的，不是一个固定 prompt，而是一个**会读状态、会选动作、会调用记忆的 optimizer program**。

> **GEPA 的关键启发：** 不是"直接改提示词"，而是"把 execution trace / evaluation trace 变成 prompt 的学习信号"。

### 6. 运行时采用三层干预，而不是"停滞才换策略"

直接借 AdaEvolve 的结构，但换成我们的 setting：

| 层级 | 干预内容 |
|---|---|
| **Local adaptation** | 改单个 island / population 的探索强度、mutation style |
| **Global adaptation** | 把预算分给更有希望的 island / task branch |
| **Meta-guidance** | 真的停滞时，不是只调温度，而是注入新的 high-level tactic |

> 如果只做"搜索停滞 → 重写 strategy prompt"，还是 EvoX 味道太重。真正拉开差距的，是让外层能在**不同时间尺度做不同层级的 intervention**。

---

## 在当前 Repo 上的最小改造路径

不用大重写，按这个顺序改：

### 第一步：保留 `strategy.py` 里的 16 维 StrategyParams

把它视为低层 actuator。

### 第二步：新增 `meta_controller.py`

输入 `run_summary + tactic_bank`，输出一个严格 JSON：

```json
{
  "action_type": "...",
  "target": "...",
  "new_phi": {...},
  "tactic_id": "...",
  "expected_gain": 0.0,
  "confidence": 0.0
}
```

### 第三步：在 `evolve_loop.py` 加 checkpoint hook

每隔 k 步或检测到 stagnation 时，生成 `run_summary`，调用 `meta_controller`，再应用 action。现在的 inner loop 已经有 island model、streaming workers、population state，这个插点很好加。

### 第四步：把 `prompt_builder.py` 一拆二

现在它只负责 mutation prompt。以后应分成：

- `build_mutation_prompt` — 服务内层 evolver
- `build_controller_prompt` — 服务外层 meta-optimizer

### 第五步：把 `meta_learner.py` 从"ES over φ"改成"prompt evolution over controller"

现在的 `MetaLearner` 明确写的是 ES 直接优化真实 validation score，surrogate 只做 warmup / prescreen / diagnostic。这个可以保留成 baseline，但主线应切到：

1. 用小预算 rollout 采 intervention data
2. 用 GEPA 式反思去演化 controller prompt
3. φ 只作为 controller 的输出空间之一

---

## 最终要证明的三件事

不是"prompt 也能优化"，而是：

1. **OOD 任务上**，在相同 adaptation budget 下，meta-controller 比 static evolver 更快
2. 不仅能提最终 best score，还能提 **AUC(best-so-far)**，也就是更快收敛
3. 学到的是**可解释 tactic prior**，而不是某个 benchmark 的 prompt overfit
