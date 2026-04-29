# Thesis Figure Redesign Plan

Target folder for new replacement images:

`figures/thesis_redesign/`

## Purpose

This plan inventories the existing thesis images and separates them into four groups:

1. **Keep as real evidence**: screenshots that prove the system and analysis are real.
2. **Regenerate with GPT**: conceptual/method diagrams that can be made more polished.
3. **Generate with chart/code**: numerical charts that must use exact experimental values.
4. **Do not use**: old or confusing images that do not support the final thesis logic.

The final visual logic should be:

**Conceptual diagrams explain the method -> screenshots prove implementation and evidence -> charts summarize results.**

---

## 1. Existing Image Inventory

### 1.1 Root `figures/` Images

| File | Size | Type | Recommendation |
|---|---:|---|---|
| `FLOW-Data-Flow.png` | 2200x550 | old flow diagram | Do not use; too thin and less aligned with final thesis logic |
| `FLOW-System-Architecture.png` | 1500x900 | old architecture diagram | Do not use; contains unrelated widget/mock detector labels |
| `G1.png` | 1672x941 | GPT concept figure | Regenerate if needed; current one is more PPT-like |
| `G3.png` | 1447x1087 | GPT framework figure | Regenerate cleaner if used |
| `G4.png` | 1672x941 | GPT contribution figure | Good for PPT, not necessary in thesis |
| `G11.png` | 1672x941 | confidence/fallback chart | Regenerate with clearer thesis style if used |
| `G12.png` | 1448x1086 | validity diagram | Optional; use only in limitations if space allows |
| `G13.png` | 2244x701 | future work roadmap | Optional; use only in conclusion/future work |
| `MODEL-Classifier-Architecture.png` | 1800x1100 | model architecture diagram | Keep or regenerate with GPT/vector style |
| `MODEL-System-Pipeline.png` | 2200x720 | runtime pipeline diagram | Keep or regenerate with GPT/vector style |
| `MODEL-Training-Pipeline.png` | 2200x900 | training pipeline diagram | Keep or regenerate with GPT/vector style |
| `MODEL-Training-Results.png` | 1700x1000 | classifier result figure | Prefer regenerate as clean chart/table, not GPT |
| `S2.png` | 1144x555 | real system screenshot | Keep as evidence |
| `S5.png` | 1130x58 | very thin screenshot | Do not use |
| `S6.png` | 921x195 | scoring policy screenshot | Keep as evidence |
| `S7.png` | 638x360 | workflow table screenshot | Keep as evidence, but small; screenshot again if possible |
| `S8-1.png` | 1144x556 | corrected audit screenshot | Keep as evidence |
| `S8-2.png` | 1141x565 | corrected audit screenshot | Keep as evidence |
| `S9.png` | 559x829 | category chart screenshot | Keep only if needed; formal chart is better |
| `S10-1.png` | 1141x562 | remaining error screenshot | Keep as evidence |
| `S10-2.png` | 1148x561 | remaining error screenshot | Keep as evidence |
| `S10-3.png` | 1146x558 | remaining error screenshot | Keep as evidence |
| `S14.png` | 1043x309 | analysis center screenshot | Keep as evidence |

### 1.2 Existing `figures/final_en/`

| File | Recommendation |
|---|---|
| `fig01_system_pipeline.png` | Superseded by stronger closed-loop / runtime figures |
| `fig02_classifier_architecture.png` | Usable, but regenerate with more polished GPT/vector style |
| `fig03_training_pipeline.png` | Usable, but regenerate if consistent style is needed |
| `fig04_training_cache.png` | Keep or regenerate; concept is useful |
| `fig05_classifier_results.png` | Prefer code-generated clean chart/table |
| `fig06_main_results.png` | Keep as formal chart |
| `fig07_category_results.png` | Keep as formal chart |
| `fig08_remaining_errors.png` | Keep as formal chart |

### 1.3 Existing `figures/template_report/`

| File | Current Use | Recommendation |
|---|---|---|
| `fig01_closed_loop.png` | Figure 1.1 | Regenerate with GPT |
| `fig02_preliminary_exploration.png` | Figure 1.2 | Keep or regenerate with chart/code, not GPT |
| `fig03_classifier_processing.png` | Figure 2.1 | Regenerate with GPT |
| `fig04_training_pipeline.png` | Figure 2.3 | Regenerate with GPT |
| `fig05_training_cache.png` | Figure 2.4 | Regenerate with GPT |
| `fig06_classifier_results.png` | Figure 4.1 | Regenerate as chart/code, not GPT |
| `fig07_main_results.png` | Figure 4.2 | Keep or regenerate as chart/code |
| `fig08_category_results.png` | Figure 4.4 | Keep or regenerate as chart/code |
| `fig09_remaining_errors.png` | Figure 5.3 | Keep or regenerate as chart/code |
| `fig10_problem_types_screenshot.png` | Figure 1.3 | Keep as real screenshot |
| `fig11_classifier_architecture.png` | Figure 2.2 | Regenerate with GPT |
| `fig12_runtime_pipeline.png` | Figure 3.1 | Regenerate with GPT |
| `fig13_evidence_panel.png` | Figure 3.2 | Keep as real screenshot |
| `fig14_workflow_table_screenshot.png` | Figure 4.3 | Keep as real screenshot, but screenshot again if possible |
| `fig15_scoring_policy_screenshot.png` | Figure 5.1 | Keep as real screenshot |
| `fig16_corrected_audit_examples.png` | Figure 5.2 | Keep as real evidence composite |
| `fig17_remaining_error_samples.png` | Figure 5.4 | Keep as real evidence composite |

---

## 2. Recommended Final Thesis Figure Set

### 2.1 GPT-Regenerated Concept / Method Figures

Place these new images in `figures/thesis_redesign/`:

1. `fig01_closed_loop_research_logic.png`
2. `fig02_classifier_input_processing.png`
3. `fig03_classifier_architecture.png`
4. `fig04_training_pipeline.png`
5. `fig05_training_cache_reproducibility.png`
6. `fig06_runtime_pipeline.png`
7. `fig07_future_work_roadmap.png` optional

### 2.2 Chart / Code Figures

Do not generate these with GPT because the numbers must be exact:

8. `fig08_preliminary_exploration_results.png`
9. `fig09_classifier_training_results.png`
10. `fig10_main_workflow_results.png`
11. `fig11_categorywise_results.png`
12. `fig12_remaining_error_distribution.png`

### 2.3 Real Screenshots / Evidence Figures

Keep or re-capture these as authentic system evidence:

| Target filename | Source |
|---|---|
| `fig13_problem_type_examples_screenshot.png` | `figures/S2.png` |
| `fig14_analysis_center_evidence_screenshot.png` | `figures/S14.png` |
| `fig15_workflow_table_screenshot.png` | `figures/S7.png` or re-capture at higher resolution |
| `fig16_scoring_policy_screenshot.png` | `figures/S6.png` |
| `fig17_corrected_audit_examples.png` | compose from `S8-1.png` and `S8-2.png` |
| `fig18_remaining_error_samples.png` | compose from `S10-1.png`, `S10-2.png`, `S10-3.png` |

---

## 3. Unified GPT Style Prefix

Use this prefix before every GPT image prompt:

```text
Create a clean, high-end academic thesis figure. White background, black and dark-gray text, thin vector lines, one restrained blue accent color (#2F5F8F), no gradients, no 3D, no decorative AI dashboard style, no cartoon characters, no fake logos, no random text. Use precise spacing, strong alignment, and readable labels. The output should look like a professional computer science dissertation diagram.
```

---

## 4. GPT Image Prompts

### 4.1 Closed-Loop Research Logic

Output filename:

`figures/thesis_redesign/fig01_closed_loop_research_logic.png`

```text
Create a clean, high-end academic thesis figure. White background, black and dark-gray text, thin vector lines, one restrained blue accent color (#2F5F8F), no gradients, no 3D, no decorative AI dashboard style, no cartoon characters, no fake logos, no random text. Use precise spacing, strong alignment, and readable labels. The output should look like a professional computer science dissertation diagram.

Draw a closed-loop research pipeline with EXACTLY 7 connected stages from left to right:

1. TRAM Temporal QA Problem
2. Preliminary Multi-Model Exploration
3. Observed Category / Model Differences
4. Lightweight Classifier Motivation
5. Prompt Routing + Fallback
6. Corrected Evaluation
7. Remaining Error Diagnosis

Design details:
- Use rounded rectangular nodes with thin borders.
- Use solid arrows between stages.
- Stage 4 and Stage 5 should use the blue accent because they are the project contribution.
- Add a subtle feedback arrow from stage 7 back to stage 4 labeled “targeted improvement”.
- Add one small subtitle under the pipeline: “The classifier is introduced as a response to observed temporal-task heterogeneity.”
- Aspect ratio 16:5.
- No extra text beyond the labels above.
```

### 4.2 Classifier Input Processing

Output filename:

`figures/thesis_redesign/fig02_classifier_input_processing.png`

```text
Create a clean, high-end academic thesis figure. White background, black and dark-gray text, thin vector lines, one restrained blue accent color (#2F5F8F), no gradients, no 3D, no decorative AI dashboard style, no cartoon characters, no fake logos, no random text. Use precise spacing, strong alignment, and readable labels. The output should look like a professional computer science dissertation diagram.

Draw a left-to-right data processing diagram with EXACTLY 7 modules:

1. Raw Question
2. Optional Context
3. Text Concatenation
4. TF-IDF Vectorizer
5. 1-2 Gram Sparse Vector
6. Logistic Regression
7. Category Probabilities

Then place a small decision block below module 7:
- Confidence = max probability
- Predicted category = argmax probability

Draw an arrow from the decision block to:
Prompt Router

Design details:
- Show “question text” and “context” as two document cards merging into one text stream.
- Use blue accent only for the final decision block and arrow into Prompt Router.
- Include labels: “input”, “feature extraction”, “classification”, “routing signal”.
- Aspect ratio 16:6.
- No formulas except “max probability” and “argmax probability”.
```

### 4.3 Lightweight Classifier Architecture

Output filename:

`figures/thesis_redesign/fig03_classifier_architecture.png`

```text
Create a clean, high-end academic thesis figure. White background, black and dark-gray text, thin vector lines, one restrained blue accent color (#2F5F8F), no gradients, no 3D, no decorative AI dashboard style, no cartoon characters, no fake logos, no random text. Use precise spacing, strong alignment, and readable labels. The output should look like a professional computer science dissertation diagram.

Draw a model architecture diagram with three vertical panels:

Panel A title: Input Representation
Contents:
- Question + context
- Token patterns
- Unigram / bigram features

Panel B title: Lightweight Classifier
Contents:
- TF-IDF matrix
- Multiclass Logistic Regression
- L2 regularization

Panel C title: Output for Routing
Contents:
- Predicted category
- Probability distribution
- Confidence score
- Fallback signal

Connections:
- Arrow from Panel A to Panel B.
- Arrow from Panel B to Panel C.
- Dashed arrow from “Fallback signal” to a small box labeled “conservative prompt”.

Design details:
- Panel B should be highlighted with the blue accent.
- Include a tiny note at the bottom: “The classifier predicts task type; the LLM still generates the final answer.”
- Aspect ratio 4:3.
```

### 4.4 Classifier Training Pipeline

Output filename:

`figures/thesis_redesign/fig04_training_pipeline.png`

```text
Create a clean, high-end academic thesis figure. White background, black and dark-gray text, thin vector lines, one restrained blue accent color (#2F5F8F), no gradients, no 3D, no decorative AI dashboard style, no cartoon characters, no fake logos, no random text. Use precise spacing, strong alignment, and readable labels. The output should look like a professional computer science dissertation diagram.

Draw a training pipeline with EXACTLY 8 steps:

1. TRAM-derived Data
2. Category Filtering
3. Strict Split
4. Train Set 6709
5. TF-IDF Fit
6. Logistic Regression Training
7. Dev Evaluation 958
8. Independent Test 1918

Add output artifacts on the right:
- task_clf.joblib
- classifier_report.json
- eval_predictions.csv

Design details:
- The strict split step should branch visually into train/dev/test.
- The test path should be visually separated from training to emphasize no leakage.
- Use blue accent for “Independent Test 1918”.
- Aspect ratio 16:6.
```

### 4.5 Training Cache and Reproducibility

Output filename:

`figures/thesis_redesign/fig05_training_cache_reproducibility.png`

```text
Create a clean, high-end academic thesis figure. White background, black and dark-gray text, thin vector lines, one restrained blue accent color (#2F5F8F), no gradients, no 3D, no decorative AI dashboard style, no cartoon characters, no fake logos, no random text. Use precise spacing, strong alignment, and readable labels. The output should look like a professional computer science dissertation diagram.

Draw a reproducibility cache mechanism:

Left side inputs:
- Split hash
- Training config
- Category whitelist
- Random seed

These merge into:
Cache Key

From Cache Key, split into two paths:
Path 1: Cache Hit -> Load existing model -> Load report
Path 2: Cache Miss -> Train classifier -> Evaluate -> Store artifacts

Artifacts box:
- task_clf.joblib
- classifier_report.json
- confusion_matrix.csv
- eval_predictions.csv

Design details:
- Use a diamond for Cache Key decision.
- Use blue accent for the cache hit/miss decision.
- Add bottom note: “Same split + same config = reproducible classifier artifact.”
- Aspect ratio 16:7.
```

### 4.6 Runtime Prompt Routing Pipeline

Output filename:

`figures/thesis_redesign/fig06_runtime_pipeline.png`

```text
Create a clean, high-end academic thesis figure. White background, black and dark-gray text, thin vector lines, one restrained blue accent color (#2F5F8F), no gradients, no 3D, no decorative AI dashboard style, no cartoon characters, no fake logos, no random text. Use precise spacing, strong alignment, and readable labels. The output should look like a professional computer science dissertation diagram.

Draw the runtime workflow from left to right:

Input Question
-> Task Classifier
-> Predicted Category + Confidence
-> Decision: confidence >= 0.95?
-> if yes: Category-Specific Prompt
-> if no: Fallback Prompt
Both prompt paths merge into:
Single LLM Call
-> Parsed Answer
-> ruleset v1.1 Evaluation
-> Final Correct / Incorrect

Design details:
- Use a diamond for the confidence decision.
- Use solid arrow for normal route and dashed arrow for fallback route.
- Highlight “Single LLM Call” and “ruleset v1.1 Evaluation” with blue accents.
- Include one small note: “Fallback does not add an extra LLM call.”
- Aspect ratio 16:6.
```

### 4.7 Optional Future Work Roadmap

Output filename:

`figures/thesis_redesign/fig07_future_work_roadmap.png`

```text
Create a clean, high-end academic thesis figure. White background, black and dark-gray text, thin vector lines, one restrained blue accent color (#2F5F8F), no gradients, no 3D, no decorative AI dashboard style, no cartoon characters, no fake logos, no random text. Use precise spacing, strong alignment, and readable labels. The output should look like a professional computer science dissertation diagram.

Draw a three-stage future work roadmap:

Stage 1:
Date Prompt Bank
305 remaining errors
Focus: cross-month / cross-year reasoning

Stage 2:
Time Zone Direction Constraints
60 remaining errors
Focus: direction and offset control

Stage 3:
Hour24 Carry/Borrow Constraints
11 remaining errors
Focus: minute-first calculation

Design details:
- Use three horizontally aligned cards connected by arrows.
- Stage 1 should be emphasized as highest priority.
- Add a small title: “Future work follows the corrected error surface.”
- Aspect ratio 16:5.
```

---

## 5. Chart / Code Figure Specifications

These should not be generated by GPT because they need exact numbers.

### 5.1 `fig08_preliminary_exploration_results.png`

Use exact values:

- DeepSeek early run: `0.9100`
- Blind multi-model: `0.8767`
- Hour24 DeepSeek: `0.9400`
- Hour24 Router: `0.9800`
- TZ DeepSeek: `0.3200`
- TZ Router: `0.4000`

### 5.2 `fig09_classifier_training_results.png`

Use exact values:

- train samples: `6709`
- dev samples: `958`
- test samples: `1918`
- dev accuracy: `1.0000`
- dev macro-F1: `1.0000`
- test accuracy: `1.0000`
- test macro-F1: `1.0000`
- test correct: `1918/1918`

### 5.3 `fig10_main_workflow_results.png`

Use exact values:

- Fixed Prompt: `0.7575`
- CoT Prompt: `0.7700`
- Classifier Router: `0.7575`
- Classifier Router + Fallback: `0.7750`

### 5.4 `fig11_categorywise_results.png`

Use exact corrected values:

- Date Computation: Fixed `0.7008`, CoT `0.7159`, Router `0.7045`, Router+Fallback `0.7235`
- Hour Adjustment (24h): Fixed `0.9545`, CoT `0.9773`, Router `0.8864`, Router+Fallback `0.9318`
- Time Zone Conversion: Fixed `0.2000`, CoT `0.2000`, Router `0.3000`, Router+Fallback `0.3000`
- Year Shift: all `1.0000`

### 5.5 `fig12_remaining_error_distribution.png`

Use exact values:

- Date Computation: `305`
- Time Zone Conversion: `60`
- Hour Adjustment (24h): `11`
- Total: `376`

---

## 6. Screenshot Figures to Keep or Re-capture

These should not be GPT-generated because their value is authenticity.

| Target filename | Source |
|---|---|
| `fig13_problem_type_examples_screenshot.png` | `figures/S2.png` |
| `fig14_analysis_center_evidence_screenshot.png` | `figures/S14.png` |
| `fig15_workflow_table_screenshot.png` | `figures/S7.png` or re-capture at higher resolution |
| `fig16_scoring_policy_screenshot.png` | `figures/S6.png` |
| `fig17_corrected_audit_examples.png` | compose from `figures/S8-1.png` and `figures/S8-2.png` |
| `fig18_remaining_error_samples.png` | compose from `figures/S10-1.png`, `figures/S10-2.png`, `figures/S10-3.png` |

---

## 7. Acceptance Checklist

- Every GPT-generated figure has only the requested labels and no hallucinated text.
- All chart/code figures use exact numeric values.
- Screenshots remain readable at Word page width.
- The final figure sequence supports this logic:
  1. problem and motivation,
  2. classifier method,
  3. implementation evidence,
  4. final results,
  5. scoring correction,
  6. remaining errors.
- Old unrelated figures are not reused, especially `FLOW-System-Architecture.png` and `S5.png`.
