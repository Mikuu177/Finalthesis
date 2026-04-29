# PPT Reference Audit

Source PPT: `academic_tram_prompt_routing_english 2.pptx`

## Overall Judgment

Most references on the final slide are relevant to the presentation. They cover four necessary foundations:

1. temporal reasoning benchmarks,
2. prompting baselines,
3. confidence / fallback motivation,
4. classical text classification.

However, two references should be handled carefully:

- `TIMEBENCH` is relevant as related benchmark background, but it is not directly used as the dataset.
- `LIBLINEAR` is not ideal unless the implementation explicitly uses the `liblinear` solver. The current training script uses scikit-learn `LogisticRegression(max_iter=1200)` with default settings, so a scikit-learn reference is more accurate.

## Reference-by-Reference Check

| PPT Reference | Reasonable? | Actually Used? | Where It Supports the PPT | Recommendation |
|---|---:|---:|---|---|
| Wang & Zhao (2024), TRAM | Yes | Yes | Dataset / benchmark source. Supports slides about TRAM, temporal task categories, and evaluation setting. | Keep. This is the most important reference. |
| Chu et al. (2024), TIMEBENCH | Yes, but indirect | Partly | Related work showing temporal reasoning is an active LLM evaluation topic. | Keep if you mention it as related benchmark, not as your dataset. |
| Wei et al. (2022), Chain-of-Thought Prompting | Yes | Yes | Supports the CoT Prompt baseline. | Keep. |
| Kojima et al. (2022), Large Language Models are Zero-Shot Reasoners | Mostly yes | Partly | Supports zero-shot / direct prompting background. | Keep only if you describe Fixed Prompt as a zero-shot prompting baseline. Otherwise optional. |
| Geifman & El-Yaniv (2017), Selective Classification | Yes | Yes | Supports fallback as selective risk control. | Keep. Useful for explaining confidence-based fallback. |
| Guo et al. (2017), On Calibration of Modern Neural Networks | Yes | Yes | Supports caution around confidence values and probability calibration. | Keep. Do not claim your classifier is perfectly calibrated. |
| Manning et al. (2008), Introduction to Information Retrieval | Yes | Yes | Supports TF-IDF feature extraction. | Keep. |
| Fan et al. (2008), LIBLINEAR | Weak / implementation mismatch | Not directly | Would support linear classification if liblinear solver were used. But current script does not specify `solver="liblinear"`. | Replace with scikit-learn paper or mention only as background for linear classification. |

## Current Implementation Evidence

The training script uses:

```python
Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50000)),
    ('lr', LogisticRegression(max_iter=1200, n_jobs=None)),
])
```

So the safest citation for implementation is:

- Pedregosa et al. (2011), scikit-learn: Machine Learning in Python.

The TF-IDF idea can still be supported by:

- Manning, Raghavan & Schutze (2008), Introduction to Information Retrieval.

## Recommended Final PPT References

I recommend the final slide use this cleaned list:

1. Wang, Y. & Zhao, Y. (2024). TRAM: Benchmarking Temporal Reasoning for Large Language Models.
2. Chu, Z. et al. (2024). TIMEBENCH: A Comprehensive Evaluation of Temporal Reasoning Abilities in LLMs.
3. Wei, J. et al. (2022). Chain-of-Thought Prompting Elicits Reasoning in Large Language Models.
4. Kojima, T. et al. (2022). Large Language Models are Zero-Shot Reasoners.
5. Geifman, Y. & El-Yaniv, R. (2017). Selective Classification for Deep Neural Networks.
6. Guo, C. et al. (2017). On Calibration of Modern Neural Networks.
7. Manning, C. D., Raghavan, P. & Schutze, H. (2008). Introduction to Information Retrieval.
8. Pedregosa, F. et al. (2011). scikit-learn: Machine Learning in Python.

## How to Explain This Slide Orally

Do not read every citation. Say:

> These references support four parts of the project: TRAM and TimeBench support temporal reasoning benchmarks; CoT and Zero-Shot Reasoning support prompting baselines; Selective Classification and Calibration support the confidence-based fallback design; and Information Retrieval plus scikit-learn support the TF-IDF and Logistic Regression classifier implementation.

Chinese meaning:

> 这些文献分别支撑项目的四个部分：TRAM 和 TimeBench 支撑时间推理 benchmark；CoT 和 Zero-Shot Reasoning 支撑提示词基线；Selective Classification 和 Calibration 支撑基于置信度的 fallback 设计；Information Retrieval 和 scikit-learn 支撑 TF-IDF 与逻辑回归分类器实现。

## Important Defense Boundary

If the teacher asks whether every reference is directly used, answer:

> Not every reference is a dataset source. TRAM is the dataset source. TimeBench is related benchmark background. CoT and zero-shot prompting support the baseline design. Selective classification and calibration support the fallback mechanism. TF-IDF and scikit-learn references support the lightweight classifier implementation.

Chinese meaning:

> 不是每一篇文献都是数据来源。TRAM 是本项目的数据来源；TimeBench 是相关 benchmark 背景；CoT 和 zero-shot prompting 支撑 baseline 设计；选择性分类和校准文献支撑 fallback 机制；TF-IDF 和 scikit-learn 文献支撑小分类器实现。
