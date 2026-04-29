# Short Speech Script for PPT  
**Title:** *Classifier-Driven Prompt Routing for Complex Temporal Reasoning on TRAM*  
**Style:** Short, fast, presentation-led

## Slide 1 — Title
Good morning. I’m Nie Wenhao. Today I will present my project on classifier-driven prompt routing for complex temporal reasoning on TRAM.

## Slide 2 — Background
The starting point is simple: temporal reasoning is not one single skill.  
Tasks such as date computation, hour adjustment, and time-zone conversion have different structures, so one universal prompt is unlikely to work equally well for all of them.

## Slide 3 — Research Question
So my question is: can a lightweight classifier identify the task type, route the question to a better prompt, and use fallback when confidence is low, in order to improve final QA accuracy?

## Slide 4 — Contributions
This project has three main contributions:
1. a classifier-driven routing framework,
2. an auditable evaluation normalization protocol, ruleset v1.1,
3. a corrected error analysis that identifies the real bottlenecks after format artifacts are removed.

## Slide 5 — System Overview
The pipeline is straightforward.  
A question is converted into TF-IDF features, classified by a Logistic Regression model, routed to a category-specific prompt, and then answered by the LLM.  
If classifier confidence is low, the system uses a fallback prompt instead.

## Slide 6 — Evaluation Protocol
A key part of the project is ruleset v1.1.  
It only changes the scoring layer, not the original data.  
The goal is to normalize equivalent answer formats, so corrected accuracy reflects real task performance more faithfully than strict string matching.

## Slide 7 — Main Results
Under corrected evaluation, the results are:
- Fixed Prompt: 0.7575
- CoT Prompt: 0.7700
- Router: 0.7575
- Router + Fallback: 0.7750

So Router + Fallback is the strongest observed workflow in the current setting.

## Slide 8 — Base vs Corrected
This page explains why the corrected scores are higher.  
The gain comes from evaluation normalization, not from retraining the model.  
So corrected results are the formal reporting standard, and base results are kept for audit only.

## Slide 9 — Category-Level Findings
The category-level pattern is clear:
- Date Computation improves after normalization, but remains the main hard category.
- Time Zone Conversion is no longer zero after correction, but it is still difficult.
- Year Shift is already near ceiling.
- Hour24 errors mainly reflect real reasoning issues, not formatting issues.

## Slide 10 — Remaining Errors
After normalization, 376 errors remain:
- 305 in Date Computation
- 60 in Time Zone Conversion
- 11 in Hour24

So the remaining error surface is now much cleaner and more useful for guiding future optimization.

## Slide 11 — Why Fallback Helps
Fallback is a risk-control mechanism.  
When classifier confidence is low, the system avoids a potentially harmful routing decision and switches to a safer prompt.  
So fallback reduces error propagation without increasing the number of LLM calls.

## Slide 12 — Limitations
The current limitations are:
- corrected oracle is still missing,
- classifier training logs need fuller export,
- evaluation is based on a frozen 400-question slice,
- and ruleset v1.1 is manually engineered.

## Slide 13 — Future Work
The next three priorities are:
1. improve the Date prompt bank,
2. add stronger directional constraints for Time Zone Conversion,
3. and add carry/borrow constraints for Hour24.

## Slide 14 — Conclusion
I will end with three points:
1. temporal reasoning is heterogeneous enough to justify routing,
2. Router + Fallback is the strongest observed workflow under the corrected protocol,
3. and after removing format artifacts, the main structural bottlenecks are now clear: Date, Time Zone, and Hour24.

Thank you.

## Slide 15 — References
These are the main references supporting the benchmark, prompting strategy, routing design, and evaluation protocol.
