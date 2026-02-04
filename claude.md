# 项目信息：复杂时间推理研究

## 项目基本信息
- **项目名称**：Research on Complex Temporal Reasoning
- **研究方向**：自然语言处理（NLP）、大语言模型（LLM）、时间推理（Temporal Reasoning）、模型评测与工作流设计
- **指导教师**：Zhen Jia（贾震）
- **项目类型**：本科毕业设计

---

## 核心研究问题

大语言模型在复杂时间推理上的真实能力如何？有哪些系统性弱点？能否通过"工作流 + 外部时间推理模块"在现有基准（尤其是 TRAM）上取得实质性提升，而不是仅仅换一个更大的黑盒模型？

---

## 一、总体目标

### 1.1 系统评估多种模型在时间推理基准上的表现

**核心基准**：TRAM（Temporal Reasoning for Large Language Models）
- 综合时间推理数据集，涵盖 **10 个任务、38 个子任务**
- 题型包括：
  - 顺序（ordering）、频率（frequency）、时长（duration）、典型时间（typical time）
  - 时间歧义解析、时间算术（temporal arithmetic）
  - 时间关系（relation）、时间 NLI、因果（causality）、故事时间线（storytelling）等

**对比模型**：
- 英文开源 LLM：LLaMA-3-8B-Instruct 等
- 中文/多语种 LLM：DeepSeek 系列、通义千问/Qwen 系列
- 传统预训练模型：RoBERTa-large（用于时间关系任务）

### 1.2 研究 Prompt 与模型配置对时间推理表现的影响

**四种提示方式**：
1. **SP-0S**：Standard Prompting, Zero-shot
2. **SP-5S**：Standard Prompting, 5-shot
3. **CoT-0S**：Chain-of-Thought Prompting, Zero-shot
4. **CoT-5S**：Chain-of-Thought Prompting, 5-shot

**研究问题**：
- CoT 是否真正改善时间推理能力？
- few-shot 示例带来的增益有多大？
- 中文 LLM 在英文任务下对这些 prompt 的敏感度如何？

### 1.3 设计面向 LLM 的复杂时间推理工作流（Workflow）

**工作流设计思路**（不是重新发明模型结构）：
1. 从输入问题（及必要上下文）中识别时间表达、事件与实体
2. 将时间表达归一化为时间点/区间/周期，构建简要时间线
3. 对需要计算的子问题（如时间差、跨时区、时长比较等）交给符号/工具模块（Python 日期库）进行精确运算
4. 利用 LLM 对整个问题进行解释性推理与最终答案生成

**对比实验**：
- 纯 LLM + prompt 端到端 vs LLM + workflow 的表现差异
- 针对 TRAM 论文中已知的错误模式（calculation slips、implicit oversights 等）

---

## 二、当前进展（文献与任务理解 + 实验设计阶段）

### 2.1 TRAM 论文研究

**已完成**：
- 通读 TRAM 论文，总结了：
  - TRAM 的整体结构、10 个任务的定义与数据来源
  - 各类任务的输入形式、输出形式和评价指标
  - 作者在 GPT-4、GPT-3.5、LLaMA-2、BERT/RoBERTa/RST 上的实验设置与主要结果
  - 错误分析中的高频错误类型（假设偏差、计算失误、忽略隐含时间线等）

### 2.2 模型与任务子集选定

**暂定模型集合**：
- **DeepSeek 系列**：中文/多语种 LLM 主力代表
- **通义千问/Qwen 系列**：主流中文 LLM，对比 DeepSeek 和英文模型
- **LLaMA-3-8B-Instruct**：英文开源小模型基线（或等价 7/8B 指令模型）
- **RoBERTa-large**：传统预训练模型，用于 Relation / Temporal NLI 等结构化任务的监督对照

**首批重点任务**：
1. **Duration（时长）**：结合常识 + 简单算术
2. **Arithmetic（时间算术）**：集中测试模型的时间计算能力
3. **Relation（时间关系标签）**：如 BEFORE/AFTER/INCLUDES 等
4. **后续扩展**：Storytelling / Causality 等更高层任务

### 2.3 Prompt 模板与小规模 Pilot 实验

**已规划**：
- 四种 prompt 框架（SP-0S、SP-5S、CoT-0S、CoT-5S）
- 准备对应的 few-shot 示例模板（从 TRAM dev 集中选出）

**近期目标**：
- 在 2–3 个任务上，用统一的 prompt 在 2–3 个模型上跑小样本（每任务 30–50 题）
- 生成"模型 × prompt × 任务"的性能表
- 为后续 workflow 设计和大规模实验打基础

---

## 三、后续技术路线

### 3.1 完成 TRAM 子集的多模型、多 Prompt 基线评测

**实现方案**：
- 使用 Python/HuggingFace + 各模型 API
- 统一评测脚本：
  - 从 TRAM JSON/CSV 读取数据
  - 根据任务类型构造对应 prompt
  - 调用不同模型（本地或远程）并解析返回结果
  - 计算 Accuracy/F1 等指标并生成对比表

### 3.2 设计与实现 Temporal Reasoning Workflow 的最小可行版本

**Step 1**：基于现有工具实现
- 时间表达解析 + 时间归一化 + 时间算术模块
- 使用 Python 的日期库、自写规则

**Step 2**：为 Arithmetic / Duration 类任务设计 Pipeline
- 先由 LLM 解析意图和参数
- 工具模块算数
- LLM 输出答案

**Step 3**：为 Relation 类任务设计小型 Workflow
- 抽取事件与候选时间点/区间
- 符号层比较
- LLM 解释

### 3.3 对比实验 & 错误分析

**对比维度**：
- 端到端 LLM + prompt
- LLM + workflow

**错误分析框架**：
- 结合 TRAM 提供的错误分类框架
- 对实验产生的错误进行分类和手工分析
- 统计各类错误（假设偏差、算术错误、关系混淆等）在两种方案中的占比变化
- 迭代改进 workflow 设计

### 3.4 最后写作阶段

**总结内容**：
- 不同模型在各类时间推理任务上的能力谱
- Prompt 组合对时间推理的影响规律
- Workflow 方案在具体任务上的提升效果与局限
- 完整毕设：背景综述、方法设计、实验结果、误差分析与未来工作

---

## 关键文献与资源

- **TRAM 论文**：Temporal Reasoning for Large Language Models（核心基准）
- **模型来源**：
  - HuggingFace Hub（LLaMA、RoBERTa 等）
  - 官方 API（DeepSeek、Qwen 等）
- **工具**：Python datetime、dateutil、自定义时间推理模块

---

## 工作进度追踪

| 阶段 | 任务 | 状态 | 备注 |
|------|------|------|------|
| 文献理解 | TRAM 论文深入阅读与总结 | ✅ 完成 | 已掌握核心内容 |
| 实验设计 | 模型与任务子集选定 | ✅  完成 | 暂定模型与首批任务已确定 |
| 实验设计 | Prompt 模板设计 | 🔄 进行中 | 四种框架已规划，示例准备中 |
| 基线评测 | Pilot 实验（小样本） | ⏳ 待开始 | 目标：2–3 个任务，30–50 题/任务 |
| 基线评测 | 完整基线评测 | ⏳ 待开始 | 多模型、多 prompt、多任务 |
| 方法设计 | Workflow 最小可行版本 | ⏳ 待开始 | 分三个 step 逐步实现 |
| 对比实验 | LLM vs LLM+Workflow 对比 | ⏳ 待开始 | 基于基线评测结果 |
| 错误分析 | 系统性错误分析 | ⏳ 待开始 | 结合 TRAM 错误分类框架 |
| 写作 | 毕设论文撰写 | ⏳ 待开始 | 最后阶段 |

---

## 重要笔记

- **核心创新点**：不是换更大的黑盒模型，而是通过 workflow + 符号推理模块来改善时间推理能力
- **关键挑战**：
  - 时间表达的多样性与歧义性
  - 隐含时间信息的识别
  - 跨语言（中英文）的时间推理差异
  - 工作流设计的通用性与可扩展性
- **预期贡献**：
  - 系统的模型性能评估
  - 对 LLM 时间推理弱点的深入理解
  - 可行的工作流方案，展示符号推理的价值








