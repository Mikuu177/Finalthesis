# academic_tram_prompt_routing_english.pptx 图像制作执行清单

源文件：`/Volumes/Mikuu-ultra/Git/Finalthesis/academic_tram_prompt_routing_english.pptx`  
总图片占位：**15 个（每页 1 个）**

---

## 一、截图类：怎么截、在哪截、截什么

> 统一建议：浏览器缩放 `100%`，窗口宽度 >= 1440，截图后按 PPT 占位框比例裁切。  
> 优先截你自己的系统前端（这是答辩最有说服力的证据）。

### S2（Slide 2 Context）TRAM 子任务异构性
- 目标占位：Slide 2 右侧图
- 页面：前端 `B. Testing Center`
- 操作步骤：
1. 打开 `B. Testing Center`
2. 滚到 `Temporal Problem Types`
3. 保证五类卡片都在一屏：`Date / Hour24 / TZ / Year / Month`
4. 截整个卡片网格区域（不要截顶部地址栏）
- 截图内容要点：五类定义 + why_hard/risk 信息同时可见

### S5（Slide 5 Method）系统流程截图
- 目标占位：Slide 5 右上图
- 页面：系统首页（A/B/C 标签上方 Hero 区域）
- 操作步骤：
1. 打开首页
2. 保证顶部 pipeline 全链路可见：`Input -> Classifier -> Confidence -> Router -> Fallback -> LLM -> Output`
3. 让 A/B/C 标签也露出（证明是系统而非静态图）
4. 截取 Hero + Pipeline + Tabs 区域

### S6（Slide 6 Measurement）评分口径截图
- 目标占位：Slide 6 右侧图
- 页面：前端 `C. Analysis Center`
- 操作步骤：
1. 切到 `C. Analysis Center`
2. 截 `Scoring Policy (Base vs Corrected)` 卡片
3. 同时让 `Ruleset Version: v1.1` 和 Base/Corrected 文案可见
4. 若空间允许，把下面 Workflow 表顶部一并截入

### S7（Slide 7 Results）主结果表截图
- 目标占位：Slide 7 右下图
- 页面：前端 `C. Analysis Center`
- 操作步骤：
1. 截 `Workflow Table (Base + Corrected)`
2. 确保四组 corrected 数值清晰：
   - Fixed `0.7575`
   - CoT `0.7700`
   - Router `0.7575`
   - Router+Fallback `0.7750`
3. 截图里保留表头（Base/Corrected/Delta）

### S8（Slide 8 Interpretation）纠偏样例截图
- 目标占位：Slide 8 左下大图
- 页面：前端 `C. Analysis Center`
- 操作步骤：
1. 下拉到 `Error Samples`
2. 设置筛选：
   - `Corrected Match = true`
   - `Category = Time Zone Conversion`（先截一张）
3. 截到表格中至少 2 条修复样例
4. 再把 `Category = Date Computation` 再截一张（可二选一拼图）

### S9（Slide 9 Category）类别变化图截图
- 目标占位：Slide 9 右下图
- 页面：前端 `C. Analysis Center`
- 操作步骤：
1. 截 `Category-wise Boundary (Corrected Highlight)` 柱图
2. 确保 `Time Zone Conversion` 行可见（Base 到 Corrected 的变化）
3. 若柱图不清晰，再补截下方 `Category Table (Base -> Corrected)`

### S10（Slide 10 Error）剩余错误样例截图
- 目标占位：Slide 10 右下图
- 页面：前端 `C. Analysis Center`
- 操作步骤：
1. 在 `Error Samples` 依次筛选并各截一张：
   - `Category = Date Computation`, `Corrected Match = false`
   - `Category = Time Zone Conversion`, `Corrected Match = false`
   - `Category = Hour Adjustment (24h)`, `Corrected Match = false`
2. 把 3 张拼成 1 张（左右或上下均可）
3. 每个子图保留 `sample_id + gold + pred_norm`

### S14（Slide 14 Takeaways）收尾总览截图
- 目标占位：Slide 14 右侧图
- 页面：前端 `C. Analysis Center`
- 操作步骤：
1. 截 `Scoring Policy` + `Main Workflow Results` 同屏区域
2. 确保 `ruleset v1.1` 和 corrected 柱图同框
3. 用作“结论页证据图”

### S15（Slide 15 References）可选
- 建议：不放图，直接删占位
- 若老师要求：放项目仓库/结果目录二维码截图（小图）

---

## 二、GPT 作图：高约束提示词（可直接复制）

> 统一风格前缀（每条都可加在最前面）：  
`Design a clean academic figure for a thesis defense slide. White background, grayscale palette with one blue accent (#3B82F6), thin vector lines, no watermark, no logo, no photorealism, no decorative gradients, high legibility.`

> 统一负向约束（每条都建议追加）：  
`Do not generate random UI text, fake equations, lorem ipsum, stock-photo style people, or unrelated icons. Keep geometry precise and minimal.`

### G1（Slide 1 Cover）封面概念图
```text
Design a clean academic figure for a thesis defense slide. White background, grayscale palette with one blue accent (#3B82F6), thin vector lines.
Create ONE cover workflow illustration with EXACTLY 7 connected modules from left to right:
1) Input Question card
2) Task Classifier block
3) Confidence Score meter (gauge value shown as 0.95)
4) Prompt Router block
5) Fallback branch node (dashed arrow down then back)
6) LLM Answer block
7) Final Output card
Layout constraints:
- Horizontal composition, centered.
- Main arrow is solid; fallback branch is dashed.
- Add tiny icon above each module (question mark, classifier grid, gauge, switch, shield, chat bubble, checkmark).
- No title text inside image.
- Keep all labels short in English only (exact labels above).
Visual constraints:
- Flat vector style, no shadows, no 3D.
- Blue accent only for arrows and confidence gauge.
- Aspect ratio 16:9.
Do not generate random text paragraphs.
```

### G3（Slide 3 Formulation）三段式框架图
```text
Create a 3-panel conceptual framework diagram (left-to-right) with EXACT panel titles:
Panel A: Task Type Recognition
Panel B: Prompt Routing Decision
Panel C: Confidence-based Risk Control
Under each panel include exactly 2 bullet points:
Panel A bullets:
- TF-IDF features
- Logistic Regression probabilities
Panel B bullets:
- Category-specific prompt bank
- Single-call LLM inference
Panel C bullets:
- If confidence < tau then fallback
- Robustness over aggressive routing
Connection constraints:
- Solid arrows A->B->C
- One dashed fallback arrow from C back to B (labeled fallback)
Visual constraints:
- Academic infographic, monochrome + blue accent.
- 4:3 ratio.
- Clean typographic hierarchy, no extra decorative shapes.
```

### G4（Slide 4 Positioning）三贡献图标条
```text
Create a horizontal “3 contributions” strip with EXACTLY 3 columns.
Column 1 title: Classifier-driven Routing
Icon: small model block + arrow switch
Subtitle: Lightweight classifier selects prompts

Column 2 title: Auditable Normalization
Icon: checklist + compare arrows
Subtitle: ruleset v1.1 scoring correction

Column 3 title: Corrected Error Surface
Icon: magnifier over bar chart
Subtitle: Remaining bottlenecks after correction

Layout constraints:
- Equal-width 3 columns, separated by thin vertical lines.
- Title (bold) + icon + one subtitle line per column.
- White background, grayscale with blue accent only on key icon strokes.
- No logos, no rounded cartoon style, no gradients.
```

### G11（Slide 11 Discussion）置信度-回退示意图
```text
Create a calibration-style chart with explicit numeric structure:
- X-axis: confidence from 0.0 to 1.0
- Y-axis: sample count
- Draw histogram bars (approximately bell-like, centered near 0.9)
- Draw a vertical threshold line at tau = 0.95 (blue)
- Shade left area label: Fallback Region
- Shade right area label: Routed Region
Add two callout boxes:
1) “low confidence -> conservative fallback”
2) “high confidence -> category prompt”
Include a tiny legend with 3 entries: Distribution, Threshold tau, Decision regions.
Visual constraints:
- Academic plot style, white background, thin axis lines.
- No random symbols, no unrelated formulas.
- 16:9 ratio.
```

### G12（Slide 12 Validity）有效性检查图
```text
Create a “Threats to Validity” diagram with ONE center node and FOUR surrounding nodes.
Center node text: Threats to Validity
Four outer nodes (exact text):
1) Corrected Oracle Missing
2) Incomplete Classifier Logs
3) Limited Evaluation Slice
4) Rule Transferability Risk
Each outer node must include one short mitigation line:
1) “compute corrected oracle next”
2) “export full training metadata”
3) “expand categories and seeds”
4) “cross-benchmark validation”
Layout constraints:
- Center in middle, four nodes around (top-left, top-right, bottom-left, bottom-right)
- Straight connectors from center to each node
- Keep text concise and readable at slide scale
- 4:3 ratio, white background, monochrome + blue accent
```

### G13（Slide 13 Next Steps）路线图图
```text
Create a one-row roadmap with 3 sequential stages and explicit counts.
Stage 1 box text:
Date Prompt Bank
305 remaining errors

Stage 2 box text:
TZ Directional Constraints
60 remaining errors

Stage 3 box text:
Hour24 Carry/Borrow Constraints
11 remaining errors

Connection constraints:
- Thick arrow from Stage 1 -> Stage 2 -> Stage 3
- Stage 1 highlighted slightly (primary focus)
- Include small milestone dots beneath each stage

Visual constraints:
- Wide banner ratio (about 16:5)
- Minimal academic infographic style
- Grayscale with blue accent on arrows only
- No extra decoration, no unrelated text
```

---

## 三、每页占位对应总表（快速核对）

| Slide | 来源 | 你要放的图 |
|---|---|---|
| 1 | GPT | 封面概念图（G1） |
| 2 | 截图 | Testing Center 的 `Temporal Problem Types` |
| 3 | GPT | 三段式框架图（G3） |
| 4 | GPT | 三贡献图标条（G4） |
| 5 | 截图 | 系统 Pipeline + A/B/C 标签 |
| 6 | 截图 | Scoring Policy（v1.1） |
| 7 | 截图 | Workflow Table（含 corrected 四组） |
| 8 | 截图 | corrected_match=true 的审计样例 |
| 9 | 截图 | Category-wise Boundary 图 |
| 10 | 截图 | 三类 remaining error 拼图 |
| 11 | GPT | 置信度阈值与 fallback 图（G11） |
| 12 | GPT | Validity checklist（G12） |
| 13 | GPT | Next steps roadmap（G13） |
| 14 | 截图 | Analysis 总览（policy + results） |
| 15 | 可选 | 建议删占位，或放二维码 |
