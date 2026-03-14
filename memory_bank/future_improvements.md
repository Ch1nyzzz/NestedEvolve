# 未来改进方向

记录当前不需要修改、但已知存在局限性、未来可能需要改进的设计点。

---

## 1. Initiator 的内省范围过窄

**现状：** Initiator 只分析 `state_dict()` 返回的配置 dict（prompt 模板、temperature、k 等超参数），优化空间仅限于配置字段的值。

**局限：**
- 无法修改组件本身（例如替换检索器实现、增删组件）
- 无法修改 pipeline 拓扑（组件顺序、分支、并行）
- 无法修改组件内部逻辑（例如给 Retriever 加 reranking 步骤）
- 无法引入新的中间步骤（例如在 retriever 后加一个 passage filter）

**可能的演进方向：**
- 将 `state_dict()` 扩展为包含"结构描述"（组件列表 + 可插拔接口），不只是参数值
- MutableField 增加 `"component"` 或 `"topology"` 类型，允许 Optimizer 生成结构性变更
- 引入代码级 patch（LLM 生成代码片段替换组件实现），但需要更强的沙盒验证

---

## 2. Delta Patch 只支持值替换

**现状：** DeltaPatch.changes 是 `{path: new_value}` 的扁平映射，只能改已有字段的值。

**局限：**
- 不能添加新字段
- 不能删除字段
- 不能表达条件逻辑（"当 X 时用 A，否则用 B"）

---

## 3. Observer 的轨迹粒度

**现状：** 只记录各组件的最终输出 (`intermediate` dict)。

**局限：**
- 没有记录 LLM 的原始 prompt 和完整 response（只有处理后的结果）
- 没有记录延迟、token 用量等效率指标
- 多轮推理（如 chain-of-thought）的中间步骤丢失

---

## 4. 评估函数的单一性

**现状：** 只用 F1 一个指标决定 patch 是否接受。

**局限：**
- 没有考虑效率（token 用量、延迟）
- 没有考虑鲁棒性（不同类型问题的表现方差）
- 多目标优化场景（F1 vs 成本 vs 速度）无法表达
