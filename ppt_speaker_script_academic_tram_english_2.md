# Concise Speaker Script for `academic_tram_prompt_routing_english 2.pptx`

**Topic:** Classifier-Driven Prompt Routing for Complex Temporal Reasoning on TRAM  
**Presenter:** Nie Wenhao  
**Style:** Short, oral, figure-focused. Do not read the slide word by word.  
**Suggested length:** 6-8 minutes.

---

## Slide 1 — Title

**EN:** Good morning. I’m Nie Wenhao. My project is about complex temporal reasoning on TRAM. The main idea is to use a small classifier before the LLM, so the system can first identify the task type and then choose a better prompt.

**CN:** 早上好，我是聂文豪。我的项目研究 TRAM 上的复杂时间推理。核心思路是在大模型前面加一个小分类器，先判断题目类型，再选择更合适的提示词。

---

## Slide 2 — Contents

**EN:** I will go through five parts: background, method, experiments, error analysis, and conclusion. The key storyline is simple: why routing is needed, how I built it, and what the results show.

**CN:** 我会按五部分讲：背景、方法、实验、错误分析和结论。主线很简单：为什么需要路由、我是怎么做的、结果说明了什么。

---

## Slide 3 — Motivation: Temporal Reasoning Is Not a Single Skill

**EN:** Temporal reasoning is not one single ability. Date calculation, hour adjustment, and time-zone conversion look similar, but they need different reasoning steps.

**CN:** 时间推理不是一种单一能力。日期计算、小时调整、时区转换表面相似，但内部步骤不同。

**EN:** This figure shows several TRAM task types. It motivates my design: instead of using one prompt for everything, the system should first understand what type of time problem it is.

**CN:** 这张图展示了 TRAM 的多种时间问题类型。它说明我的设计动机：不要所有题都用同一个提示词，而是先判断题目类型。

---

## Slide 4 — Research Question and Thesis Claim

**EN:** My research question is: can a small classifier predict the task type and help choose a better prompt for the LLM?

**CN:** 我的研究问题是：一个小分类器能不能判断题目类型，并帮助大模型选择更合适的提示词？

**EN:** The figure here shows the core logic: classify the question, estimate confidence, route to a prompt, and use fallback when confidence is low.

**CN:** 这张图展示核心逻辑：先分类，再估计置信度，再路由到提示词；如果置信度低，就回退。

---

## Slide 5 — Contributions

**EN:** My work has three main parts. First, I built a classifier-driven prompt router. Second, I added a scoring normalization rule. Third, I analyzed the remaining real errors after normalization.

**CN:** 我的工作主要有三部分。第一，做了分类器驱动的提示词路由。第二，加入了评分规范化规则。第三，分析了规范化之后真正剩下的错误。

**EN:** The important point is that this is not only a prompt experiment. It is a small system with training, routing, evaluation, and error diagnosis.

**CN:** 重点是，这不只是调提示词，而是一个包含训练、路由、评测和错误分析的小系统。

---

## Slide 6 — Classifier Overview

**EN:** This is the key system module. The input question goes into my lightweight classifier, which predicts the temporal category and confidence.

**CN:** 这是系统的核心模块。输入问题先进入轻量分类器，分类器输出时间问题类别和置信度。

**EN:** The LLM itself is not fine-tuned. My trainable part is the classifier, and my system contribution is how it controls prompt selection.

**CN:** 大模型本身没有微调。我自己训练的是分类器，系统贡献是用它控制提示词选择。

---

## Slide 7 — Evaluation Protocol: ruleset v1.1

**EN:** This slide explains why I changed the scoring layer. Some answers were semantically correct, but counted wrong because the format was different.

**CN:** 这一页解释为什么要修正评分层。有些答案语义上是对的，但因为格式不同被判错。

**EN:** ruleset v1.1 only normalizes answers during scoring. It does not change the original dataset or gold labels.

**CN:** ruleset v1.1 只在评分时做答案规范化，不修改原始数据和标准答案。

---

## Slide 8 — Main Results: Corrected Accuracy

**EN:** This is the main result. Under corrected evaluation, Router plus Fallback reaches 0.7750, which is the best observed workflow.

**CN:** 这是主结果。在修正评测下，Router 加 Fallback 达到 0.7750，是当前观察到最好的工作流。

**EN:** Also, all methods use one LLM call per question. So the gain is not from calling the LLM more times, but from better prompt selection and fallback.

**CN:** 而且所有方法每题都是一次大模型调用。所以提升不是靠多调用，而是靠更好的提示词选择和回退机制。

---

## Slide 9 — Base vs Corrected: What Changed?

**EN:** This table shows the difference between strict scoring and corrected scoring. The corrected scores are higher because equivalent formats are now recognized.

**CN:** 这张表展示严格评分和修正评分的差别。修正后分数更高，是因为等价格式现在能被识别。

**EN:** I want to stress that this is not a new model run. The outputs are the same; only the evaluation rule is more reasonable.

**CN:** 我想强调，这不是重新跑模型。模型输出没变，只是评分规则更合理了。

---

## Slide 10 — Category-Level Findings

**EN:** This page is important because different categories behave very differently. Date Computation improves a lot after normalization, but it is still the hardest category.

**CN:** 这一页很重要，因为不同类别表现差异很大。Date Computation 修正后提升很多，但仍然是最难的类别。

**EN:** Time Zone Conversion is no longer zero after correction, but it is still weak. Year Shift is already near perfect, so it is not the main bottleneck.

**CN:** Time Zone Conversion 修正后不再是零，但仍然较弱。Year Shift 基本接近满分，所以不是主要瓶颈。

---

## Slide 11 — Remaining Corrected Errors

**EN:** After normalization, 376 errors remain. Most of them are Date Computation errors, followed by Time Zone Conversion and Hour24.

**CN:** 规范化之后还剩 376 个错误。大部分来自 Date Computation，其次是 Time Zone Conversion 和 Hour24。

**EN:** This tells me where to improve next. The next work should not be random prompt tuning, but targeted improvement on these categories.

**CN:** 这告诉我下一步该改哪里。后续不应该随便调提示词，而应该针对这些类别优化。

---

## Slide 12 — Error Sample Evidence Panel

**EN:** This slide shows real error samples from the system. I use it to show that the analysis is traceable, not just based on summary numbers.

**CN:** 这一页展示系统里的真实错误样本。我用它说明分析是可追溯的，不只是看汇总数字。

**EN:** The key columns are gold answer, predicted answer, corrected match, and error type. These help explain why each case is still wrong.

**CN:** 关键字段是标准答案、预测答案、修正匹配结果和错误类型。它们能解释每个样本为什么仍然错。

---

## Slide 13 — Why Fallback Helps

**EN:** Fallback is a risk-control mechanism. If the classifier confidence is high, the system routes to a category prompt. If confidence is low, it uses a safer fallback prompt.

**CN:** Fallback 是一种风险控制机制。如果分类器置信度高，就使用类别提示词；如果置信度低，就使用更安全的回退提示词。

**EN:** The figure shows this confidence boundary. The goal is to avoid harmful routing decisions without adding extra LLM calls.

**CN:** 图中展示了这个置信度边界。目标是在不增加调用次数的情况下，避免有风险的路由决策。

---

## Slide 14 — Limitations

**EN:** There are still limitations. The evaluation slice is 400 questions, so larger-scale testing would make the conclusion stronger.

**CN:** 目前还有局限。评估切片是 400 题，如果后续扩大测试规模，结论会更强。

**EN:** Also, ruleset v1.1 is manually designed. It works for this project, but whether it generalizes to other benchmarks still needs testing.

**CN:** 另外，ruleset v1.1 是人工设计的。它适用于本项目，但能否泛化到其他 benchmark 还需要验证。

---

## Slide 15 — Future Work

**EN:** Future work follows the error distribution. First, improve Date Computation prompts. Second, add stronger direction constraints for Time Zone Conversion. Third, improve carry and borrow handling for Hour24.

**CN:** 后续工作直接跟着错误分布走。第一，优化日期计算提示词。第二，加强时区转换的方向约束。第三，改进 Hour24 的进位和借位处理。

---

## Slide 16 — Conclusion

**EN:** To conclude, temporal reasoning is heterogeneous, so task-aware routing is meaningful. In my experiment, Router plus Fallback is the strongest observed workflow under corrected evaluation.

**CN:** 总结来说，时间推理具有异质性，所以任务感知路由是有意义的。在我的实验中，Router 加 Fallback 是修正评测下表现最好的工作流。

**EN:** More importantly, after removing format artifacts, the real bottlenecks become clear: Date Computation, Time Zone Conversion, and Hour24.

**CN:** 更重要的是，去掉格式误判之后，真正的瓶颈变得清楚：日期计算、时区转换和 Hour24。

**EN:** Thank you. I’m happy to answer questions.

**CN:** 谢谢大家，欢迎提问。

---

## Slide 17 — References

**EN:** These are the main references. They support the benchmark, prompting methods, confidence-based fallback, and the TF-IDF plus Logistic Regression classifier.

**CN:** 这些是主要参考文献，分别支持 benchmark、提示词方法、基于置信度的回退机制，以及 TF-IDF 加逻辑回归分类器。

---

# Quick Q&A Notes

## Why not only use the LLM?

**EN:** Because the classifier gives the system an explicit task-recognition layer. It makes prompt selection more controllable.

**CN:** 因为分类器提供了明确的任务识别层，让提示词选择更可控。

## Why is classifier accuracy high but QA accuracy lower?

**EN:** Classification only predicts the task type. The LLM still has to do the actual temporal reasoning, and that is where many errors happen.

**CN:** 分类只判断题目类型，真正的时间计算还是由大模型完成，很多错误发生在这一步。

## Did ruleset v1.1 change the data?

**EN:** No. It only changes scoring. The original gold answers stay unchanged.

**CN:** 没有。它只改变评分方式，原始标准答案不变。

## What is the biggest remaining problem?

**EN:** Date Computation. It has the most remaining errors after correction.

**CN:** 最大问题是日期计算，因为修正后它仍然有最多错误。
